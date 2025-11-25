from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import modeling_utils
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice import VibeVoiceConfig
from .modeling_vibevoice import VibeVoiceModel, VibeVoicePreTrainedModel
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from .modular_vibevoice_text_tokenizer import (
    VibeVoiceTextTokenizer,
    VibeVoiceTextTokenizerFast,
)

# from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel
from .modular_vibevoice_tokenizer import (
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
)
from .streamer import AsyncAudioStreamer, AudioStreamer

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences.
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """

    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for valid tokens
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0

        # Apply mask to scores
        scores = scores + mask
        return scores


class VibeVoiceForConditionalGenerationInference(VibeVoicePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)

        # Initialize the base model
        self.model = VibeVoiceModel(config)

        # LM head for text generation
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)

        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if not getattr(self.config, "tie_word_embeddings", False):
            return

        if hasattr(self, "lm_head") and hasattr(self.model.language_model, "embed_tokens"):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        """Process speech inputs through tokenizers and connectors."""
        with torch.no_grad():
            if speech_type == "audio":
                # Encode audio to acoustic latents
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]

                # Apply scaling and bias
                acoustic_features = (
                    acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)
                ) * self.model.speech_scaling_factor.to(acoustic_latents.device)

                # Connect to language model space
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]

                return acoustic_features, acoustic_connected
            elif speech_type == "pt":
                encoder_output = VibeVoiceTokenizerEncoderOutput(
                    mean=speech_tensors, std=self.acoustic_tokenizer.config.fix_std
                )
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]

                # Apply scaling and bias
                acoustic_features = (
                    acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)
                ) * self.model.speech_scaling_factor.to(acoustic_latents.device)

                # Connect to language model space
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]

                return acoustic_features, acoustic_connected
            else:
                raise NotImplementedError(f"Speech type {speech_type} not implemented")

    # @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            speech_tensors (`torch.FloatTensor`, *optional*):
                Input speech waveforms for voice cloning or speech understanding.
            speech_masks (`torch.BoolTensor`, *optional*):
                Masks indicating valid speech frames.
            speech_input_mask (`torch.BoolTensor`, *optional*):
                Positions in the input sequence where speech embeddings should be inserted.

        Returns:
            `VibeVoiceCausalLMOutputWithPast` or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process speech inputs if provided
        if speech_tensors is not None and speech_masks is not None:
            acoustic_features, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def _build_generate_config_model_kwargs(
        self, generation_config, inputs, tokenizer, return_processors=False, **kwargs
    ):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config,
            True,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            **kwargs,
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = self.device

        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
        model_kwargs["cache_position"] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)

        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=StoppingCriteriaList()
            )

            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids


AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGenerationInference)

__all__ = [
    "VibeVoiceForConditionalGenerationInference",
]
