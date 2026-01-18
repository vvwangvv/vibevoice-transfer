# train_vibevoice_lora.py
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import VerificationMode, load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser, Trainer, TrainerCallback
from transformers import TrainingArguments as HfTrainingArguments
from transformers import set_seed

from vibevoice.finetune.data_vibevoice import (
    DynamicBatchSampler,
    VibeVoiceCollator,
    VibeVoiceDataset,
)
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger(__name__)

# ================== SAMPLE CALLBACK UTILS ==================

import copy

import torch
from transformers import TrainerCallback


class EmaCallback(TrainerCallback):
    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        """
        attr_path: where the head lives under self.model (Trainer wraps your VibeVoiceForConditionalGeneration)
        decay:     EMA decay (0.999 ~ stable, 0.9999 ~ very smooth, slower to adapt)
        """
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None  # store non-EMA weights when we swap

    def _get_module(self, model):
        # Resolve dotted path like "model.prediction_head"
        mod = model
        for name in self.attr_path.split("."):
            mod = getattr(mod, name)
        return mod

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        head = self._get_module(model)
        self.shadow = {k: p.detach().to(self.device).clone() for k, p in head.state_dict().items()}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        head = self._get_module(model)
        with torch.no_grad():
            for k, v in head.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach().to(self.device), alpha=(1.0 - self.decay))

    # ---- Swap helpers ----
    def _swap_in_ema(self, model):
        head = self._get_module(model)
        self._orig = copy.deepcopy(head.state_dict())
        head.load_state_dict(self.shadow, strict=False)

    def _swap_back(self, model):
        if self._orig is None:
            return
        head = self._get_module(model)
        head.load_state_dict(self._orig, strict=False)
        self._orig = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # use EMA during eval
        self._swap_in_ema(model)

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_save(self, args, state, control, model=None, **kwargs):
        # temporarily swap to EMA, let Trainer save, then swap back
        self._swap_in_ema(model)

    def on_save_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # final checkpoint: persist EMA
        self._swap_in_ema(model)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to VibeVoice base model with config.json"}
    )
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to processor dir (preprocessor_config.json). Defaults to model path."}
    )
    cache_dir: Optional[str] = field(default=None)
    freeze_acoustic_tokenizer: bool = field(default=True)
    freeze_semantic_tokenizer: bool = field(default=True)
    train_diffusion_head: bool = field(
        default=False, metadata={"help": "Train diffusion prediction head (full fine-tune)"}
    )
    train_connectors: bool = field(
        default=False, metadata={"help": "Train acoustic/semantic connectors (full fine-tune)"}
    )
    layers_to_freeze: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated indices of diffusion head layers to freeze (e.g., '0,1,5,7,8')."},
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "HF dataset name or 'json' with --train_jsonl for local files"}
    )
    dataset_config_name: Optional[str] = field(default=None)
    train_split_name: str = field(default="train")
    eval_split_name: Optional[str] = field(default="validation")
    text_column_name: str = field(default="text")
    audio_column_name: str = field(default="audio")
    voice_prompts_column_name: Optional[str] = field(default="voice_prompts")
    eval_split_size: float = field(default=0.0)
    ignore_verifications: bool = field(default=False)
    train_jsonl: Optional[Path] = field(
        default=None, metadata={"help": "Path to local train JSONL with {text, audio, [voice_prompts]}"}
    )
    validation_jsonl: Optional[Path] = field(default=None, metadata={"help": "Optional path to local validation JSONL"})
    voice_input_use_semantic: bool = field(
        default=False,
        metadata={
            "help": "the priority is lower than generation_use_semantic_only and understanding_use_semantic_only"
        },
    )
    voice_prompt_drop_rate: float = field(
        default=0.0,
        metadata={
            "help": "Probability to drop conditioning voice prompt during training (0.0 keep always, 1.0 drop always)."
        },
    )
    multiple_choice_version: int = field(default=2, metadata={"help": "version 1: only ABCD; version 2: A <Speaker>;"})
    num_choices: int = field(default=4, metadata={"help": "number of choices for multiple choice task"})
    understanding_use_semantic_only: bool = field(
        default=False, metadata={"help": "Use mismatched acoustic/semantic representation for voice prompts"}
    )
    generation_use_semantic_only: bool = field(
        default=False, metadata={"help": "Use mismatched acoustic/semantic representation for voice prompts"}
    )


@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    ddpm_batch_mul: int = field(default=1)
    ce_loss_weight: float = field(default=1.0)
    diffusion_loss_weight: float = field(default=1.0)
    per_device_train_max_samples: Optional[int] = field(default=None)
    per_device_train_max_tokens: Optional[int] = field(default=None)
    gradient_clipping: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient clipping using max_grad_norm (set via --max_grad_norm, default 1.0). When False, disables clipping by forcing max_grad_norm=0.0."
        },
    )


class VibeVoiceTrainer(Trainer):

    def compute_loss(
        self,
        model: VibeVoiceForConditionalGeneration,
        inputs: Dict[str, Any],
        return_outputs=False,
        num_items_in_batch: Optional[int] = None,
    ):
        labels = inputs.get("labels")

        # Use custom training forward pass with new diffusion loss
        outputs = model(ddpm_batch_mul=self.args.ddpm_batch_mul, **inputs)

        # CE Loss
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Diffusion loss
        diffusion_loss = (
            outputs.diffusion_loss if outputs.diffusion_loss is not None else torch.tensor(0.0, device=ce_loss.device)
        )
        total = self.args.ce_loss_weight * ce_loss + self.args.diffusion_loss_weight * diffusion_loss

        prefix = "train" if model.training else "eval"
        self.log(
            {
                f"{prefix}/batch_size": logits.size(0),
                f"{prefix}/ce_loss": ce_loss.detach().item(),
                f"{prefix}/diffusion_loss": diffusion_loss.detach().item(),
            }
        )
        if hasattr(self, "optimizer") and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            lr_val = self.optimizer.param_groups[0].get("lr", None)
            if lr_val is not None:
                self.log({"train/learning_rate_real": float(lr_val)})

        return (total, outputs) if return_outputs else total

    def get_train_dataloader(self):
        self.accelerator.even_batches = False
        batch_sampler = DynamicBatchSampler(
            self.train_dataset,
            self.args.per_device_train_max_tokens,
            max_samples=self.args.per_device_train_max_samples,
            random_seed=self.args.seed,
            drop_residual=False,
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            batch_sampler=batch_sampler,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)
        return train_dataloader


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # Configure gradient clipping
    if not getattr(training_args, "gradient_clipping", False):
        if hasattr(training_args, "max_grad_norm"):
            training_args.max_grad_norm = 0.0
            logger.info("Gradient clipping disabled (set max_grad_norm=0.0). Use --gradient_clipping to enable.")
    else:
        if (
            (not hasattr(training_args, "max_grad_norm"))
            or training_args.max_grad_norm is None
            or training_args.max_grad_norm <= 0
        ):
            training_args.max_grad_norm = 1.0
        logger.info(f"Gradient clipping enabled: max_grad_norm={training_args.max_grad_norm}")

    # Load processor
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError("--model_name_or_path (or --processor_name_or_path) must be provided")
    processor: VibeVoiceProcessor = VibeVoiceProcessor.from_pretrained(processor_path)

    # Required special tokens
    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id", "text_start_id", "text_end_id"]:
        if not hasattr(tok, required) or getattr(tok, required) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    # Load model
    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required to load VibeVoice base model")
    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        dtype = torch.float16
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
    )

    if data_args.generation_use_semantic_only:
        model.model.prediction_head.noisy_images_proj = nn.Linear(
            model.config.semantic_tokenizer_config.vae_dim, model.config.decoder_config.hidden_size, bias=False
        )
        model.model.prediction_head.final_layer.linear = nn.Linear(
            model.config.decoder_config.hidden_size, model.config.semantic_tokenizer_config.vae_dim, bias=False
        )

    # Diagnostics: LM head tie
    try:
        in_emb_mod = model.get_input_embeddings()
        out_emb_mod = model.get_output_embeddings()
        in_w = getattr(in_emb_mod, "weight", None)
        out_w = getattr(out_emb_mod, "weight", None)
        shared_ptr = bool(in_w is not None and out_w is not None and in_w.data_ptr() == out_w.data_ptr())
        values_equal = False
        if in_w is not None and out_w is not None and in_w.shape == out_w.shape:
            try:
                values_equal = bool(torch.allclose(in_w, out_w))
            except Exception:
                values_equal = False
        try:
            tie_cfg = getattr(getattr(model.config, "decoder_config", model.config), "tie_word_embeddings", None)
        except Exception:
            tie_cfg = getattr(model.config, "tie_word_embeddings", None)
        logger.info(
            f"LM head diagnostics -> shared_params={shared_ptr}, values_equal={values_equal}, tie_word_embeddings={tie_cfg}"
        )
        if out_w is not None:
            logger.info(f"LM head requires_grad before freeze: {bool(out_w.requires_grad)}")
    except Exception as e:
        logger.warning(f"LM head tie diagnostics failed: {e}")

    # Hard-tie LM head
    try:
        emb_module = model.get_input_embeddings()
        head_module = model.get_output_embeddings()
        if hasattr(emb_module, "weight") and hasattr(head_module, "weight"):
            if (
                emb_module.weight.shape == head_module.weight.shape
                and emb_module.weight.data_ptr() != head_module.weight.data_ptr()
            ):
                with torch.no_grad():
                    head_module.weight = emb_module.weight
                logger.info("Force-tied LM head weight to input embeddings (pointer share).")
    except Exception as e:
        logger.warning(f"Force-tie of LM head failed: {e}")

    # Validate special IDs (info logs only)
    special_names = ["speech_start_id", "speech_diffusion_id", "speech_end_id", "text_start_id", "text_end_id"]
    try:
        vocab_size = int(getattr(model.config.decoder_config, "vocab_size", 0))
    except Exception:
        vocab_size = 0
    in_emb_mod = model.get_input_embeddings()
    out_emb_mod = model.get_output_embeddings()
    in_w = getattr(in_emb_mod, "weight", None)
    out_w = getattr(out_emb_mod, "weight", None)
    for name in special_names:
        val = getattr(tok, name, None)
        exists = val is not None
        in_range = exists and isinstance(val, int) and 0 <= val < vocab_size
        equal_row = None
        if in_range and in_w is not None and out_w is not None and in_w.shape == out_w.shape and in_w.size(0) > val:
            try:
                equal_row = bool(torch.allclose(in_w[val], out_w[val]))
            except Exception:
                equal_row = False
        decoded_str = None
        if exists and isinstance(val, int):
            try:
                decoded_str = tok.decode([val])
            except Exception:
                try:
                    decoded_str = tok.convert_ids_to_tokens(val)
                except Exception:
                    decoded_str = "<decode_failed>"
        logger.info(
            f"Special token check -> {name}={val}, decoded='{decoded_str}', exists={exists}, in_vocab_range={in_range}, emb_vs_head_row_equal={equal_row}"
        )

    # Quick tokenizer diagnostics (optional)
    try:
        logger.info("=== TOKENIZER DIAGNOSTICS ===")
        logger.info(f"Tokenizer class: {type(tok).__name__}")
        logger.info(f"Tokenizer vocab_size: {tok.vocab_size}")
        # tiny CE smoke test
        with torch.no_grad():
            simple_text = "The cat sat on the mat."
            simple_ids = torch.tensor([tok.encode(simple_text, add_special_tokens=True)], device=model.device)
            simple_mask = torch.ones_like(simple_ids)
            x = model.get_input_embeddings()(simple_ids)
            outputs = model.model(inputs_embeds=x, attention_mask=simple_mask, return_dict=True)
            logits = model.lm_head(outputs.last_hidden_state)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = simple_ids[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean"
            )
            logger.info(f"Simple text CE loss: {ce_loss.item():.4f}")
    except Exception as e:
        logger.warning(f"Tokenizer diagnostics failed: {e}")

    # Disable cache during training
    if hasattr(model.config, "use_cache") and training_args.do_train:
        model.config.use_cache = False

    # Freeze tokenizers
    if model_args.freeze_acoustic_tokenizer and hasattr(model.model, "acoustic_tokenizer"):
        for p in model.model.acoustic_tokenizer.parameters():
            p.requires_grad = False
    if model_args.freeze_semantic_tokenizer and hasattr(model.model, "semantic_tokenizer"):
        for p in model.model.semantic_tokenizer.parameters():
            p.requires_grad = False

    model.tie_weights()

    # Connectors
    if not getattr(model_args, "train_connectors", False):
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = False
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = False

    # Freeze embedding + head
    emb = model.get_input_embeddings()
    if hasattr(emb, "weight"):
        emb.weight.requires_grad_(False)
    head = model.get_output_embeddings()
    if head is not None and hasattr(head, "weight"):
        head.weight.requires_grad_(False)

    # Diagnostics
    def _sum_params(named_iter):
        return sum(p.numel() for _, p in named_iter if p.requires_grad)

    lm_lora = (
        _sum_params(model.model.language_model.named_parameters()) if hasattr(model.model, "language_model") else 0
    )
    pred_head_train = (
        _sum_params(model.model.prediction_head.named_parameters()) if hasattr(model.model, "prediction_head") else 0
    )
    ac_conn_train = (
        _sum_params(model.model.acoustic_connector.named_parameters())
        if hasattr(model.model, "acoustic_connector")
        else 0
    )
    se_conn_train = (
        _sum_params(model.model.semantic_connector.named_parameters())
        if hasattr(model.model, "semantic_connector")
        else 0
    )
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Trainable by block -> LLM-LoRA: {lm_lora:,} | diff_head: {pred_head_train:,} | ac_conn: {ac_conn_train:,} | se_conn: {se_conn_train:,}"
    )
    logger.info("TOTAL trainable: %s", f"{total_trainable:,}")

    # Datasets
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.train_jsonl is not None:
        raw = {"train": [], "validation": []}
        with data_args.train_jsonl.open("r") as f:
            for line in tqdm(f, desc="Loading train JSONL"):
                if line.strip():
                    raw["train"].append(json.loads(line))
        if data_args.validation_jsonl is not None:
            with data_args.validation_jsonl.open("r") as f:
                for line in tqdm(f, desc="Loading validation JSONL"):
                    if line.strip():
                        raw["validation"].append(json.loads(line))
    else:
        if data_args.dataset_name is None:
            raise ValueError(
                "Provide --dataset_name (HF datasets) or use --train_jsonl/--validation_jsonl for local files."
            )
        raw = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            verification_mode=verification_mode,
            cache_dir=model_args.cache_dir,
        )
    train_ds = raw[data_args.train_split_name]
    eval_ds = None
    if training_args.do_eval:
        if data_args.eval_split_name and data_args.eval_split_name in raw:
            eval_ds = raw[data_args.eval_split_name]
        elif data_args.eval_split_size and data_args.eval_split_size > 0 and len(train_ds) > 1:
            split = train_ds.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_ds, eval_ds = split["train"], split["test"]

    train_dataset = VibeVoiceDataset(
        train_ds,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        voice_prompts_column=data_args.voice_prompts_column_name,
    )
    eval_dataset = None
    if eval_ds is not None:
        eval_dataset = VibeVoiceDataset(
            eval_ds,
            text_column=data_args.text_column_name,
            audio_column=data_args.audio_column_name,
            voice_prompts_column=data_args.voice_prompts_column_name,
        )

    # Ratios/dims from processor+model
    speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
    data_collator = VibeVoiceCollator(
        processor=processor,
        speech_compress_ratio=speech_compress_ratio,
        voice_prompt_drop_rate=data_args.voice_prompt_drop_rate,
        voice_input_use_semantic=data_args.voice_input_use_semantic,
        speakers=train_dataset.speakers,
        multiple_choice_version=data_args.multiple_choice_version,
        num_choices=data_args.num_choices,
        generation_use_semantic_only=data_args.generation_use_semantic_only,
        understanding_use_semantic_only=data_args.understanding_use_semantic_only,
    )

    ema_cb = EmaCallback(attr_path="model.prediction_head", decay=0.999, device="cpu")

    setattr(model, "tokenizer", processor.tokenizer)

    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            ema_cb,
        ],
    )

    # Optional debug pre-training save
    if getattr(training_args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.do_eval and eval_dataset is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
