import argparse
import os
import re
import time
import traceback
from typing import Any, Dict, List, Tuple, Union

import torch
from transformers.utils import logging

from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
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
        "--text",
        type=str,
        default="Speaker EN_B00001_S09397: From really early on, I defined my own identity as an artist. This was before photography.",
        help="Text input for the script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--disable_prefill",
        action="store_true",
        help="Disable speech prefill (voice cloning) by setting is_prefill=False during generation",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 1.3)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize potential 'mpx' typo to 'mps'
    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"

    # Validate mps availability if requested
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    load_dtype = torch.bfloat16
    attn_impl_primary = "flash_attention_2"
    print(f"Using device: {args.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
    # Load model with device-specific logic
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
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
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
        else:
            raise e

    if args.disable_prefill:
        print("Voice cloning disabled: running generation with is_prefill=False")
    else:
        print("Voice cloning enabled: running generation with is_prefill=True")

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    if hasattr(model.model, "language_model"):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    # Prepare inputs for the model
    inputs = processor(
        text=args.text,  # Wrap in list for batch processing
        task="generation",
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to target device
    target_device = args.device if args.device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Starting generation with cfg_scale: {args.cfg_scale}")

    # Generate audio
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        is_prefill=not args.disable_prefill,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Calculate audio duration and additional metrics
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        # Assuming 24kHz sample rate (common for speech synthesis)
        sample_rate = 24000
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")

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

    # Print summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Output file: {output_path}")
    print(f"Speaker names: {args.speaker_names}")
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")

    print("=" * 50)


if __name__ == "__main__":
    main()
