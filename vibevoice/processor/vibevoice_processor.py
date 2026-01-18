import math
import os
import random
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import (
    AudioInput,
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType, logging

from vibevoice.utils import make_pad_mask

from .vibevoice_tokenizer_processor import AudioNormalizer

logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT_FOR_GENERATION = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
DEFAULT_SYSTEM_PROMPT_FOR_UNDERSTANDING = " Transform the speech provided by various speakers into text output, recognizing the distinct voice of each respective speaker.\n"


class VibeVoiceProcessor:
    r"""
    Constructs a VibeVoice processor which wraps a VibeVoice tokenizer and audio processor into a single processor.

    [`VibeVoiceProcessor`] offers all the functionalities of [`VibeVoiceTokenizer`] and [`VibeVoiceTokenizerProcessor`].
    See the [`~VibeVoiceProcessor.__call__`] and [`~VibeVoiceProcessor.decode`] for more information.

    Args:
        tokenizer (`VibeVoiceTextTokenizer` or `VibeVoiceTextTokenizerFast`):
            The tokenizer for text processing.
        audio_processor (`VibeVoiceTokenizerProcessor`):
            The audio processor for speech processing.
        speech_tok_compress_ratio (`int`, *optional*, defaults to 3200):
            The compression ratio for speech tokenization.
        db_normalize (`bool`, *optional*, defaults to True):
            Whether to apply decibel normalization to audio inputs.
    """

    def __init__(
        self,
        tokenizer=None,
        audio_processor=None,
        speech_tok_compress_ratio=3200,
        db_normalize=True,
        system_prompt_for_generation=None,
        system_prompt_for_understanding=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None

        if system_prompt_for_generation is None:
            self.system_prompt_for_generation = DEFAULT_SYSTEM_PROMPT_FOR_GENERATION
        else:
            self.system_prompt_for_generation = system_prompt_for_generation

        if system_prompt_for_understanding is None:
            self.system_prompt_for_understanding = DEFAULT_SYSTEM_PROMPT_FOR_UNDERSTANDING
        else:
            self.system_prompt_for_understanding = system_prompt_for_understanding

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibeVoiceProcessor from a pretrained VibeVoice processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained model
                - a path to a *directory* containing processor config

        Returns:
            [`VibeVoiceProcessor`]: The processor object instantiated from pretrained model.
        """
        import json
        import os

        from transformers.utils import cached_file

        from vibevoice.modular.modular_vibevoice_text_tokenizer import (
            VibeVoiceTextTokenizer,
            VibeVoiceTextTokenizerFast,
        )

        from .vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor

        # Try to load from local path first, then from HF hub
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None

        if os.path.exists(config_path):
            # Local path exists
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Try to load from HF hub
            try:
                config_file = cached_file(pretrained_model_name_or_path, "preprocessor_config.json", **kwargs)
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load preprocessor_config.json from {pretrained_model_name_or_path}: {e}")
                logger.warning("Using default configuration")
                config = {
                    "speech_tok_compress_ratio": 3200,
                    "db_normalize": True,
                }

        # Extract main processor parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer - try from model path first, then fallback to Qwen
        language_model_pretrained_name = config.get("language_model_pretrained_name", None) or kwargs.pop(
            "language_model_pretrained_name", "Qwen/Qwen2.5-1.5B"
        )
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        if "qwen" in language_model_pretrained_name.lower():
            tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(language_model_pretrained_name, **kwargs)
        else:
            raise ValueError(
                f"Unsupported tokenizer type for {language_model_pretrained_name}. Supported types: Qwen, Llama, Gemma."
            )

        # Load audio processor
        if "audio_processor" in config:
            # Create audio processor from config
            audio_config = config["audio_processor"]
            audio_processor = VibeVoiceTokenizerProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
                eps=audio_config.get("eps", 1e-6),
            )
        else:
            # Create default audio processor
            audio_processor = VibeVoiceTokenizerProcessor()

        # Create and return the processor
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a processor to a directory, so that it can be re-loaded using the
        [`~VibeVoiceProcessor.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor will be saved.
        """
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save processor configuration
        processor_config = {
            "processor_class": "VibeVoiceProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "VibeVoiceTokenizerProcessor",
                "sampling_rate": getattr(self.audio_processor, "sampling_rate", 24000),
                "normalize_audio": getattr(self.audio_processor, "normalize_audio", True),
                "target_dB_FS": getattr(self.audio_processor, "target_dB_FS", -25),
                "eps": getattr(self.audio_processor, "eps", 1e-6),
            },
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w") as f:
            json.dump(processor_config, f, indent=2)

        logger.info(f"Processor configuration saved in {config_path}")

    def __call__(
        self,
        text: Optional[
            Union[str, List[str], TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        audio: Optional[Union[np.ndarray, str, List[Union[np.ndarray, str]]]] = None,
        speaker: Optional[
            Union[str, List[str], TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        task: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
        all_speakers: Optional[Set[str]] = None,
        multiple_choice_version: int = 2,
        num_choices: int = 4,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to process one or more podcast scripts with optional voice samples.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process. Can be:
                - A single script string
                - A list of script strings for batch processing
                - A path to a .json or .txt file
                - A list of paths
            audio (`str`, `np.ndarray`, `List[Union[str, np.ndarray]]`, *optional*):
                Audio inputs for each script. Can be:
                - A single audio input for a single script
                - A list of samples for a single script
            task (`str`, `List[str]` - "generation" or "understanding"):
                The task(s) to perform. Can be:
                - A single task string
                - A list of task strings for batch processing
            padding (`bool`, `str` or `PaddingStrategy`, defaults to `True`):
                Whether to pad sequences to the same length
            truncation (`bool`, `str` or `TruncationStrategy`, defaults to `False`):
                Whether to truncate sequences
            return_tensors (`str` or `TensorType`, *optional*):
                If set, will return tensors of a particular framework
            return_attention_mask (`bool`, defaults to `True`):
                Whether to return the attention mask

        Returns:
            `BatchEncoding`: A BatchEncoding with the following fields:
                - **input_ids** -- List of token id sequences or tensor
                - **attention_mask** -- List of attention masks or tensor
                - **speech_tensors** -- Padded speech inputs
                - **speech_masks** -- Speech masks
                - **speech_input_mask** -- Boolean masks indicating speech token positions
        """
        # Handle single vs batch input
        if not isinstance(text, list) and not isinstance(audio, list):
            # Single input
            texts, audios, speakers, tasks = [text], [audio], [speaker], [task]
        else:
            texts, audios, speakers, tasks = text, audio, speaker, task

        # Process each input
        all_encodings = []
        for text_input, audio_input, speaker, task in zip(texts, audios, speakers, tasks):
            if task == "generation":
                encoding = self._process_single_for_generation(text_input, speaker)
            else:
                encoding = self._process_single_for_understanding(
                    audio_input, speaker, all_speakers, multiple_choice_version, num_choices
                )
            all_encodings.append(encoding)

        # Combine batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )
        batch_encoding["multiple_choice_answer"] = [enc.get("multiple_choice_answer", None) for enc in all_encodings]
        return batch_encoding

    def _process_single_for_generation(
        self,
        text: str,
        speaker: str,
    ) -> Dict[str, Any]:
        """Process a single podcast script."""
        # Determine if text is a file path or direct script

        # Create system prompt
        # system_tokens = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
        system_tokens = self.tokenizer.encode(self.system_prompt_for_generation)

        # Build full token sequence
        full_tokens = system_tokens
        speech_input_mask = [False] * len(system_tokens)

        # Add text input section
        full_tokens += self.tokenizer.encode(" Text input:\n", add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(" Text input:\n", add_special_tokens=False))

        speaker_text_tokens = self.tokenizer.encode(f" Speaker {speaker}:{text}\n", add_special_tokens=False)
        full_tokens += speaker_text_tokens
        speech_input_mask += [False] * len(speaker_text_tokens)

        # Add speech output section
        full_tokens += self.tokenizer.encode(" Speech output:\n", add_special_tokens=False) + [
            self.tokenizer.speech_start_id
        ]
        speech_input_mask += [False] * (len(self.tokenizer.encode(" Speech output:\n", add_special_tokens=False)) + 1)

        return {
            "input_ids": full_tokens,
            "speech_inputs": None,
            "speech_input_mask": speech_input_mask,
        }

    def _process_single_for_understanding(
        self,
        audio: Union[str, AudioInput],
        speaker: Optional[str] = None,
        all_speakers: Optional[Set[str]] = None,
        multiple_choice_version: int = 2,
        num_choices: int = 4,
    ) -> Dict[str, Any]:
        """
        multiple_choice_version = 1: choice
        multiple_choice_version = 2: choice + speaker name
        """

        assert audio is not None, "Audio input is required for understanding task."
        """Process a single podcast script."""

        system_tokens = self.tokenizer.encode("Listen to the following speech and answer the question.\n")

        # Build full token sequence
        full_tokens = system_tokens
        speech_input_mask = [False] * len(system_tokens)

        # Add speech input section
        full_tokens += self.tokenizer.encode(" Speech input:\n", add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(" Speech input:\n", add_special_tokens=False))

        speech_tokens, wav, speech_mask = self.prepare_speech_input(audio)
        full_tokens += speech_tokens
        speech_input_mask += speech_mask

        choice_tokens, multiple_choice_answer = self._create_multi_choice_prompt(
            all_speakers, answer=speaker, num_choices=num_choices
        )
        full_tokens += choice_tokens
        speech_input_mask += [False] * len(choice_tokens)

        if multiple_choice_version == 2:
            multiple_choice_answer += f".{speaker}"

        return {
            "input_ids": full_tokens,
            "speech_inputs": [wav],
            "speech_input_mask": speech_input_mask,
            "multiple_choice_answer": multiple_choice_answer,
        }

    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
    ) -> BatchEncoding:
        """Combine multiple encodings into a batch with padding."""
        # Extract input_ids and create attention_mask
        input_ids_list = [enc["input_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]

        # Determine padding strategy
        if isinstance(padding, bool):
            padding_strategy = PaddingStrategy.LONGEST if padding else PaddingStrategy.DO_NOT_PAD
        elif isinstance(padding, str):
            padding_strategy = PaddingStrategy(padding)
        else:
            padding_strategy = padding

        # Apply padding to input_ids
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            max_len = max(len(ids) for ids in input_ids_list)

            # Pad sequences
            padded_input_ids = []
            attention_masks = []
            padded_speech_input_masks = []

            for input_ids, speech_mask in zip(input_ids_list, speech_input_masks_list):
                # Truncate if needed
                if truncation and len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    speech_mask = speech_mask[:max_len]

                # Pad
                padding_length = max_len - len(input_ids)
                # padded_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                padded_ids = [self.tokenizer.pad_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + [1] * len(input_ids)
                padded_speech_mask = [False] * padding_length + speech_mask

                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
                padded_speech_input_masks.append(padded_speech_mask)

            input_ids_list = padded_input_ids
            speech_input_masks_list = padded_speech_input_masks
        else:
            # No padding, just create attention masks
            attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None

        # Process speech inputs
        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True

        # Prepare batch encoding
        batch_encoding = BatchEncoding()

        # Handle tensor conversion
        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list

        # Process speech tensors if present
        if has_speech:
            speech_dict = self.prepare_speech_inputs(
                all_speech_inputs,
                return_tensors=return_tensors,
            )
            batch_encoding["speech_tensors"] = speech_dict["padded_speeches"]
            batch_encoding["speech_masks"] = speech_dict["speech_masks"]
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None

        return batch_encoding

    def _create_transcribe_prompt(self) -> List[int]:
        question = "What did the previous speech say? The transcription is: "
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        return question_tokens

    def _create_speaker_prompt(self) -> List[int]:
        """
        Create multi-choice prompt tokens.

        Returns:
            tuple: (choice_tokens, choice_speech_inputs, choice_speech_masks)
        """
        question = "Who spoke the previous speech? "
        question_full_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        return question_full_tokens

    def _create_multi_choice_prompt(self, speakers: Set[str], answer: str, num_choices: int = 4) -> List[int]:
        """
        Create multi-choice prompt tokens.

        Returns:
            tuple: (choice_tokens, choice_speech_inputs, choice_speech_masks)
        """
        question = "Who spoke the previous speech? Choices: "

        choices = random.sample(list(speakers - {answer}), k=num_choices - 1)
        choices.append(answer)
        random.shuffle(choices)

        answer_choice = None
        indices = "ABCDEFGHIJK"
        for index, speaker in zip(indices, choices):
            question += f"{index}.{speaker}; "
            if speaker == answer:
                answer_choice = index

        question += "The answer is "
        # A.spkid
        # A

        question_full_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        return question_full_tokens, answer_choice

    def prepare_speech_input(self, speech: Union[str, np.ndarray]) -> Tuple[List[int], List[np.ndarray], List[bool]]:
        """
        Create speech tokens and process speech samples for a single speech input.

        Returns:
            tuple: (audio_tokens, voice_speech_inputs, voice_speech_masks)
        """
        vae_token_id = self.tokenizer.speech_diffusion_id

        if isinstance(speech, str):
            wav = self.audio_processor._load_audio_from_path(speech)
        elif isinstance(speech, dict):
            # Handle dict format with 'array' or 'audio' key
            if "array" in speech:
                wav = np.array(speech["array"], dtype=np.float32)
            elif "audio" in speech:
                wav = np.array(speech["audio"], dtype=np.float32)
            else:
                raise ValueError(f"Dictionary audio input must have 'array' or 'audio' key, got: {speech.keys()}")
        else:
            wav = np.array(speech, dtype=np.float32)

        # Apply normalization if needed
        if self.db_normalize and self.audio_normalizer:
            wav = self.audio_normalizer(wav)

        vae_tok_len = math.ceil(wav.shape[0] / self.speech_tok_compress_ratio)

        # Build tokens and masks
        speech_tokens = (
            [self.tokenizer.speech_start_id]
            + [vae_token_id] * vae_tok_len
            + [self.tokenizer.speech_end_id]
            + self.tokenizer.encode("\n", add_special_tokens=False)
        )

        speech_masks = [False] + [True] * vae_tok_len + [False] + [False]
        speech_input = wav

        return speech_tokens, speech_input, speech_masks

    def prepare_speech_inputs(
        self,
        speech_inputs: List[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare speech inputs for model consumption.

        Args:
            speech_inputs: List of speech arrays
            return_tensors: Output tensor type

        Returns:
            Dictionary with padded_speeches and speech_masks
        """
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}

        # Calculate sequence lengths
        vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs]
        speech_masks = make_pad_mask(vae_tok_seqlens)
        padded_speeches = pad_sequence(
            [torch.from_numpy(speech_input) for speech_input in speech_inputs], batch_first=True
        )

        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }

        if return_tensors == "np":
            result["padded_speeches"] = padded_speeches.numpy()
            result["speech_masks"] = speech_masks.numpy()

        return result

    def _convert_json_to_script(self, json_file: str) -> str:
        """
        Convert JSON format to script format.
        Expected JSON format:
        [
            {"speaker": "1", "text": "Hello everyone..."},
            {"speaker": "2", "text": "Great to be here..."}
        ]
        """
        import json

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of speaker entries")

        script_lines = []
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict entry: {item}")
                continue

            speaker = item.get("speaker")
            text = item.get("text")

            if speaker is None or text is None:
                logger.warning(f"Skipping entry missing speaker or text: {item}")
                continue

            # Ensure speaker ID is valid
            try:
                speaker_id = int(speaker)
            except (ValueError, TypeError):
                logger.warning(f"Invalid speaker ID: {speaker}, skipping entry")
                continue

            # Clean up text
            text = text.strip()
            if text:
                script_lines.append(f"Speaker {speaker_id}: {text}")

        if not script_lines:
            raise ValueError("No valid entries found in JSON file")

        return "\n".join(script_lines)

    def _convert_text_to_script(self, text_file: str) -> str:
        """
        Convert text file to script format.
        Handles multiple formats:
        1. Already formatted as "Speaker X: text"
        2. Plain text (assigns to Speaker 1)

        Handles edge cases like multiple colons in a line.
        """
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        script_lines = []
        current_speaker = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as "Speaker X: text" format
            # Use regex to be more robust
            speaker_match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)

            if speaker_match:
                speaker_id = int(speaker_match.group(1))
                text = speaker_match.group(2).strip()
                if text:
                    script_lines.append(f"Speaker {speaker_id}: {text}")
            else:
                # Treat as plain text - assign to current speaker
                script_lines.append(f"Speaker {current_speaker}: {line}")

        if not script_lines:
            raise ValueError("No valid content found in text file")

        return "\n".join(script_lines)

    def _merge_inputs(self, text_inputs: BatchEncoding, audio_inputs: Dict) -> BatchEncoding:
        """Merge text and audio inputs into a single BatchEncoding."""
        # Start with text inputs
        merged = BatchEncoding(text_inputs)

        # Add audio-specific fields
        if "audio" in audio_inputs:
            merged["speech_inputs"] = audio_inputs["audio"]
        if "streaming" in audio_inputs:
            merged["streaming"] = audio_inputs["streaming"]

        return merged

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibeVoiceTextTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Return the list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names + audio_processor_input_names + ["speech_inputs", "speech_input_mask"])
        )

    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ) -> str:
        """
        Save audio data to a file.
        Args:
            audio (Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]):
                The audio data to save. Can be a single tensor/array or a list of them.
            output_path (str, optional): Path to save the audio file. Defaults to "output.wav".
            sampling_rate (int, optional): Sampling rate for the audio. If None, uses the processor's default.
            normalize (bool, optional): Whether to normalize the audio before saving. Defaults to False.
            batch_prefix (str, optional): Prefix for batch audio files. Defaults to "audio_".
        Returns:
            str: The path to the saved audio file.
        """
        return self.audio_processor.save_audio(
            audio, output_path=output_path, sampling_rate=sampling_rate, normalize=normalize, batch_prefix=batch_prefix
        )


__all__ = [
    "VibeVoiceProcessor",
]
