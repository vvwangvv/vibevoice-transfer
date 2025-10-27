import io
import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from vibevoice.utils import make_pad_mask


def _resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    return wav


# Lightweight HF-style dataset wrapper (optional). Trainer can also pass raw HF datasets directly.
class VibeVoiceDataset:
    def __init__(
        self,
        dataset: Any,
        text_column: str = "text",
        audio_column: str = "audio",
        voice_prompts_column: Optional[str] = "voice_prompts",
        force_voice_prompts: bool = False,
    ) -> None:
        self.dataset = dataset
        self.text_column = text_column
        self.audio_column = audio_column
        self.voice_prompts_column = voice_prompts_column
        self.force_voice_prompts = force_voice_prompts

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        data: Dict[str, Any] = {}
        data["text"] = item[self.text_column]
        data["audio"] = item[self.audio_column]

        user_provided_prompt = None
        if self.voice_prompts_column and self.voice_prompts_column in item:
            user_provided_prompt = item[self.voice_prompts_column]

        if user_provided_prompt:
            # A prompt was provided in the dataset, so we use it.
            if not isinstance(user_provided_prompt, list):
                data["voice_prompts"] = [user_provided_prompt]
            else:
                data["voice_prompts"] = user_provided_prompt
        else:
            # FALLBACK: No prompt provided, so we auto-generate one from the target audio.
            try:
                target_sr = 24000
                wav_array = _load_audio_to_24k(item[self.audio_column], target_sr=target_sr)
                audio_len_seconds = len(wav_array) / target_sr

                min_len_sec = min(5.0, audio_len_seconds / 4.0)
                max_len_sec = min(15.0, audio_len_seconds / 2.0)

                if min_len_sec > max_len_sec:
                    min_len_sec = max_len_sec
                max_len_sec = min(max_len_sec, audio_len_seconds)

                if max_len_sec > 0.1:
                    prompt_len_sec = random.uniform(min_len_sec, max_len_sec)
                    prompt_len_samples = int(prompt_len_sec * target_sr)

                    max_start_sample = len(wav_array) - prompt_len_samples
                    start_sample = random.randint(0, max_start_sample)

                    prompt_crop = wav_array[start_sample : start_sample + prompt_len_samples]

                    data["voice_prompts"] = [prompt_crop]
                else:
                    data["voice_prompts"] = None

            except Exception as e:
                warnings.warn(f"Could not create voice prompt for item {idx}: {e}")
                data["voice_prompts"] = None
        data["generation_task_prob"] = item.get("generation_task_prob", 1.0)
        return data


def _apply_silence_with_crossfade(
    wav: torch.Tensor,
    *,
    sample_rate: int,
    pre_silence_sec: float = 0.25,
    pre_crossfade_sec: float = 0.25,
    post_crossfade_sec: float = 0.25,
    post_silence_sec: float = 0.75,
) -> torch.Tensor:
    """Pad audio with leading/trailing silence and apply crossfades.

    Structure: [pre_silence][pre_crossfade][audio_body][post_crossfade][post_silence]
    Crossfades blend the audio with silence linearly to avoid hard edges.
    """

    start_sil_samples = int(round(pre_silence_sec * sample_rate))
    end_sil_samples = int(round(post_silence_sec * sample_rate))
    pre_crossfade_samples = int(round(pre_crossfade_sec * sample_rate))
    post_crossfade_samples = int(round(post_crossfade_sec * sample_rate))

    total_len = wav.size(0)
    if total_len == 0:
        pieces = []
        if start_sil_samples > 0:
            pieces.append(torch.zeros(start_sil_samples, dtype=torch.float32))
        if end_sil_samples > 0:
            pieces.append(torch.zeros(end_sil_samples, dtype=torch.float32))
        return torch.cat(pieces) if pieces else wav

    start_len = min(pre_crossfade_samples, total_len)
    remaining_after_start = max(total_len - start_len, 0)
    end_len = min(post_crossfade_samples, remaining_after_start)
    middle_end_idx = total_len - end_len

    start_segment = wav[:start_len]
    middle_segment = wav[start_len:middle_end_idx]
    end_segment = wav[middle_end_idx:]

    def _linear_fade(num_samples: int, start: float, end: float) -> torch.Tensor:
        if num_samples <= 0:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.linspace(start, end, num_samples, dtype=torch.float32)

    start_crossfade = start_segment * _linear_fade(start_len, 0.0, 1.0)
    end_crossfade = end_segment * _linear_fade(end_segment.size(0), 1.0, 0.0)

    pieces = []
    if start_sil_samples > 0:
        pieces.append(torch.zeros(start_sil_samples, dtype=torch.float32))
    if start_crossfade.size(0) > 0:
        pieces.append(start_crossfade)
    if middle_segment.size(0) > 0:
        pieces.append(middle_segment)
    if end_crossfade.size(0) > 0:
        pieces.append(end_crossfade)
    if end_sil_samples > 0:
        pieces.append(torch.zeros(end_sil_samples, dtype=torch.float32))

    return torch.cat(pieces)


def _load_audio_to_24k(
    audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]],
    *,
    target_sr: int = 24000,
    augment_with_silence: bool = False,
) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        wav_out = torch.from_numpy(audio).float()
    elif isinstance(audio, torch.Tensor):
        wav_out = audio.detach().cpu().float()
    elif isinstance(audio, str):
        if ":" not in audio:
            wav, sr = torchaudio.load(audio)
        else:
            wav, sr = _load_audio_from_tar(audio)
        wav = wav.mean(dim=0)
        wav_out = _resample_if_needed(wav, sr, target_sr)
    elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        wav = torch.from_numpy(np.asarray(audio["array"]).float())
        sr = int(audio["sampling_rate"])
        wav_out = _resample_if_needed(wav, sr, target_sr)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    if augment_with_silence:
        wav_out = _apply_silence_with_crossfade(wav_out, sample_rate=target_sr)

    return wav_out


def _load_audio_from_tar(entry):
    tar_path, offset, size = entry.split(":")
    offset, size = int(offset), int(size)

    with open(tar_path, "rb") as f:
        f.seek(offset)
        audio_bytes = f.read(size)
    try:
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
    except:
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes), format="mp3")

    return audio, sr


@dataclass
class VibeVoiceCollator:
    processor: Any  # VibeVoiceProcessor
    max_length: Optional[int] = None
    speech_compress_ratio: int = 3200

    text_field: str = "text"
    audio_field: str = "audio"
    voice_prompts_field: str = "voice_prompts"
    voice_prompt_drop_rate: float = 0.0

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        sample_input_ids: List[List[int]] = []
        sample_attention_masks: List[List[int]] = []

        sample_acoustic_input_masks: List[List[bool]] = []
        sample_semantic_input_masks: List[List[bool]] = []
        sample_labels: List[List[bool]] = []
        all_speech_waveforms: List[np.ndarray] = []
        all_speech_latent_lengths: List[int] = []
        per_segment_is_target: List[bool] = []

        for ex in features:
            text: str = ex.get(self.text_field, "")
            voice_prompts: Optional[List[Union[str, np.ndarray, torch.Tensor]]] = ex.get(self.voice_prompts_field)
            audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]] = ex.get(self.audio_field)
            generation_task_prob: float = ex.get("generation_task_prob", 1.0)
            task = "generation" if random.random() < generation_task_prob else "understanding"

            proc = self.processor(
                text=[text],
                voice_samples=(
                    [voice_prompts]
                    if voice_prompts is not None and random.random() >= self.voice_prompt_drop_rate
                    else None
                ),
                audio=[audio],
                task=[task],
                padding=False,
                truncation=False,
                max_length=self.max_length,
                return_tensors="pt",
            )

            ids = proc["input_ids"][0].tolist()
            attn = proc.get("attention_mask", torch.ones_like(proc["input_ids"]))[0].tolist()
            speech_input_mask = proc.get("speech_input_mask", None)
            if speech_input_mask is None:
                speech_input_mask = torch.zeros_like(proc["input_ids"], dtype=torch.bool)
            else:
                speech_input_mask_list = speech_input_mask[0].tolist()

            if proc["speech_tensors"] is not None:
                all_speech_waveforms.extend([voice_speech for voice_speech in proc["speech_tensors"]])
                all_speech_latent_lengths.extend([voice_mask.sum() for voice_mask in proc["speech_masks"]])
                per_segment_is_target.extend([False] * len(proc["speech_tensors"]))

            prompt_len = len(ids)
            if task == "generation":
                wav_target = _load_audio_to_24k(audio, target_sr=24000, augment_with_silence=True)
                target_latent_len = max(1, int(math.ceil(len(wav_target) / self.speech_compress_ratio)))
                all_speech_waveforms.append(wav_target)
                all_speech_latent_lengths.append(target_latent_len)
                per_segment_is_target.append(True)

                ids = (
                    ids
                    + [self.processor.tokenizer.speech_diffusion_id] * target_latent_len
                    + [self.processor.tokenizer.speech_end_id]
                )
                attn = attn + [1] * target_latent_len + [1]
                acoustic_input_mask = speech_input_mask_list + [True] * target_latent_len + [False]
                semantic_input_mask = [False] * len(speech_input_mask_list) + [True] * target_latent_len + [False]
            else:
                text_target_ids = self.processor.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                ids = ids + text_target_ids + [self.processor.tokenizer.text_end_id]
                attn = attn + [1] * len(text_target_ids) + [1]
                acoustic_input_mask = speech_input_mask_list + [False] * len(text_target_ids) + [False]
                semantic_input_mask = [False] * len(speech_input_mask_list) + [False] * len(text_target_ids) + [False]
            # Ensure text decoding sees an explicit end-of-sequence token after speech output.
            eos_token_id = getattr(self.processor.tokenizer, "eos_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            assert eos_token_id is not None and eos_token_id >= 0, "Tokenizer has no eos_token_id;"

            ids.append(eos_token_id)
            attn.append(1)
            acoustic_input_mask.append(False)
            semantic_input_mask.append(False)
            labels = [-100] * prompt_len + ids[prompt_len:]

            if self.max_length is not None and len(ids) > self.max_length:
                continue

            sample_input_ids.append(torch.LongTensor(ids))
            sample_attention_masks.append(torch.LongTensor(attn))
            sample_acoustic_input_masks.append(torch.BoolTensor(acoustic_input_mask))
            sample_semantic_input_masks.append(torch.BoolTensor(semantic_input_mask))
            sample_labels.append(torch.LongTensor(labels))

        tok = self.processor.tokenizer
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = getattr(tok, "eos_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            raise ValueError("Tokenizer has no pad_token_id or eos_token_id; please set one or pass a valid pad id.")

        input_ids_tensor = pad_sequence(sample_input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask_tensor = pad_sequence(sample_attention_masks, batch_first=True, padding_value=0)
        acoustic_input_mask_tensor = pad_sequence(sample_acoustic_input_masks, batch_first=True, padding_value=False)
        semantic_input_mask_tensor = pad_sequence(sample_semantic_input_masks, batch_first=True, padding_value=False)
        sample_labels_tensor = pad_sequence(sample_labels, batch_first=True, padding_value=-100)

        # is_target waveforms should compute semantic features
        semantic_speech_waveforms = [
            speech_waveform
            for is_target, speech_waveform in zip(per_segment_is_target, all_speech_waveforms)
            if is_target
        ]
        if len(semantic_speech_waveforms) == 0:
            semantic_speech_mask = semantic_speech = None
        else:
            semantic_speech_mask = make_pad_mask(
                [len_ for (len_, is_target) in zip(all_speech_latent_lengths, per_segment_is_target) if is_target]
            )
            semantic_speech = pad_sequence(semantic_speech_waveforms, batch_first=True, padding_value=0.0)

        # all waveforms compute acoustic features
        if len(all_speech_waveforms) == 0:
            acoustic_speech_mask = acoustic_speech = acoustic_speech_loss_mask = None
        else:
            acoustic_speech_mask = make_pad_mask(all_speech_latent_lengths)
            acoustic_speech = pad_sequence(all_speech_waveforms, batch_first=True, padding_value=0.0)
            acoustic_speech_loss_mask = torch.zeros_like(acoustic_speech_mask, dtype=torch.bool)
            for i, is_target in enumerate(per_segment_is_target):
                if is_target:
                    acoustic_speech_loss_mask[i] = acoustic_speech_mask[i]

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "acoustic_speech": acoustic_speech,
            "semantic_speech": semantic_speech,
            "acoustic_speech_mask": acoustic_speech_mask,
            "semantic_speech_mask": semantic_speech_mask,
            "acoustic_input_mask": acoustic_input_mask_tensor,
            "semantic_input_mask": semantic_input_mask_tensor,
            "acoustic_speech_loss_mask": acoustic_speech_loss_mask,
            "diffusion_loss_mask": semantic_input_mask_tensor,  # loss are computed on semantic positions
            "labels": sample_labels_tensor,
        }
