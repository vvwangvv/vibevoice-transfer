import argparse
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.utils import logging

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5b",
        help="Path to the HuggingFace model directory",
    )

    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="path to meta file",
    )
    parser.add_argument(
        "--n-choices",
        type=int,
        default=4,
        help="Number of choices for multiple choice task",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save output audio files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    load_dtype = torch.bfloat16
    attn_impl_primary = "flash_attention_2"
    # Load model with device-specific logic
    try:
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=load_dtype,
            device_map="cuda",
            attn_implementation=attn_impl_primary,
        )
    except Exception as e:
        if attn_impl_primary == "flash_attention_2":
            print(f"[ERROR] : {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print(
                "Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality."
            )
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation="sdpa",
            )
        else:
            raise e

    model.eval()

    if hasattr(model.model, "language_model"):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    all_speakers = []
    with args.meta.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            uid, text, audio_path, _ = line.strip().split("|")
            all_speakers.append(uid.rsplit("_", 1)[0])

    correct_choice = correct_speaker = total = 0
    with args.meta.open("r", encoding="utf-8") as f, args.output.open("w", encoding="utf-8") as out_f:
        for line in tqdm(f):
            if not line.strip():
                continue

            uid, text, audio_path, _ = line.strip().split("|")
            speaker_ref = uid.rsplit("_", 1)[0]

            # Prepare inputs for the model
            inputs = processor(
                audio=audio_path,
                task="understanding",
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
                all_speakers=set(all_speakers),
                speaker=speaker_ref,
                multiple_choice_version=1,
                num_choices=args.n_choices,
            )

            # Move tensors to target device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to("cuda")

            # Generate audio
            start_time = time.time()
            for key in ["parsed_scripts", "all_speakers_list"]:
                inputs.pop(key, None)
            answer = inputs.pop("multiple_choice_answer", None)

            dtype = next(model.parameters()).dtype
            inputs["acoustic_input_mask"] = inputs.pop("speech_input_mask")
            inputs["semantic_input_mask"] = torch.zeros_like(inputs["acoustic_input_mask"]).bool()
            inputs["acoustic_speech"] = inputs.pop("speech_tensors").to(dtype)
            inputs["acoustic_speech_mask"] = inputs.pop("speech_masks")

            suppress_tokens = [
                processor.tokenizer.speech_diffusion_id,
                processor.tokenizer.speech_end_id,
                processor.tokenizer.speech_start_id,
            ]
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                eos_token_id=processor.tokenizer.text_end_id,
                suppress_tokens=suppress_tokens,
                tokenizer=processor.tokenizer,
                output_logits=True,
                return_dict_in_generate=True,
            )
            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.2f} seconds")

            # Calculate token metrics
            input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
            generated_tokens = outputs.sequences[0][input_tokens:]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            try:
                choice_answer, speaker_answer = generated_text.split("Bit")[0].split(".")
            except:
                choice_answer = generated_text.split("Bit")[0]
                speaker_answer = ""

            choice_ref = answer[0]

            if choice_answer == choice_ref:
                correct_choice += 1
            if speaker_answer.strip() == speaker_ref:
                correct_speaker += 1
            out_f.write(f"{uid} {choice_ref} {choice_answer} {speaker_ref} {speaker_answer}\n")
            total += 1
    print(f"Choice Accuracy: {correct_choice}/{total} = {correct_choice/total:.4f}")
    print(f"Speaker Accuracy: {correct_speaker}/{total} = {correct_speaker/total:.4f}")


if __name__ == "__main__":
    main()
