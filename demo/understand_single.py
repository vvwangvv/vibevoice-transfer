import argparse
import os
import re
import time
import traceback
from pathlib import Path

import torch
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
        "--audio",
        type=Path,
        default=Path(
            "/mnt/bn/wangwei-nas-lq-01/mlx/users/wangwei.0/workspace/VibeVoice/data/manifest/emilia_spk30_en/test1/wavs/EN_B00001_S09397_W000608.wav"
        ),
        help="Path to the audio file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
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

    # Prepare inputs for the model
    inputs = processor(
        audio=args.audio.as_posix(),
        task="understanding",
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to target device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda")

    # Generate audio
    start_time = time.time()
    for key in ["parsed_scripts", "all_speakers_list"]:
        inputs.pop(key, None)

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
        max_new_tokens=None,
        max_length=256,
        tokenizer=processor.tokenizer,
        output_logits=True,
        return_dict_in_generate=True,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    breakpoint()

    # Calculate token metrics
    input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
    output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
    generated_tokens = output_tokens - input_tokens

    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save output (processor handles device internally)
    output_path = os.path.join(args.output_dir, f"debug_generated.wav")
    os.makedirs(args.output_dir, exist_ok=True)

    processor.save_audio(
        outputs.speech_outputs[0],  # First (and only) batch item
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
    main()
