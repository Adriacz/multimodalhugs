"""
SiameseMultiModalEmbedderModel

A two-pathway model for audio+video sign language translation pretraining.

Phase 1 — forward(input_frames=..., input_audio=..., labels=...):
    Video path:  FeatureExtractor → MultimodalMapper → merge_modalities → Backbone (MT loss)
    Audio path:  AudioFeatureExtractor (frozen) → AudioProjection
    Alignment:   OT loss between video_repr and audio_repr in d_model space.
    Total loss:  MT_loss + ot_lambda * OT_loss

Phase 2 — forward(input_frames=..., input_audio=None, labels=...):
    Pure video forward, identical to the parent MultiModalEmbedderModel.

Generation always runs in pure-video mode (input_audio is not passed).
"""
import logging
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel
from multimodalhugs.models.siamese_multimodal_embedder.configuration_siamese_multimodal_embedder import (
    SiameseMultiModalEmbedderConfig,
)
from multimodalhugs.modules import FeatureExtractor
from multimodalhugs.modules.sinkhorn import batch_sinkhorn_loss
from multimodalhugs.modules.utils import set_module_parameters, merge_modalities
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model("siamese_multimodal_embedder")
class SiameseMultiModalEmbedderModel(MultiModalEmbedderModel):
    """
    Siamese multimodal model with OT-based audio↔video alignment.

    Inherits the full video pipeline from MultiModalEmbedderModel.
    When ``input_audio`` is provided in forward(), the audio branch runs in
    parallel and contributes an OT alignment loss.  When ``input_audio`` is
    None the forward call is identical to the parent class.
    """

    config_class = SiameseMultiModalEmbedderConfig

    def __init__(self, config: SiameseMultiModalEmbedderConfig):
        super().__init__(config)
        self._init_audio_branch(config)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_audio_branch(self, config: SiameseMultiModalEmbedderConfig):
        if config.audio_feature_extractor_type is None:
            self.audio_feature_extractor = None
            self.audio_projection = None
            return

        self.audio_feature_extractor = FeatureExtractor(
            feature_extractor_type=config.audio_feature_extractor_type,
            pretrained_module=config.pretrained_audio_feature_extractor,
        )
        set_module_parameters(
            self.audio_feature_extractor,
            freeze=config.freeze_audio_feature_extractor,
        )

        # Project audio encoder output into the backbone's embedding space.
        if config.audio_feat_dim != config.d_model:
            self.audio_projection = nn.Linear(config.audio_feat_dim, config.d_model)
        else:
            self.audio_projection = nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_frames: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        input_audio: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        Forward pass.

        When ``input_audio`` is None, delegates entirely to the parent class
        (pure video mode — identical behaviour to MultiModalEmbedderModel).

        When ``input_audio`` is provided (shape [B, n_mels, T_mel]):
            1. Runs the full video pipeline up to (and including) the mapper to
               obtain video_repr [B, T_v, D].
            2. Runs the frozen audio encoder + projection to obtain
               audio_repr [B, T_a, D].
            3. Computes the Sinkhorn OT loss between video_repr and audio_repr.
            4. Continues the video path through merge_modalities + backbone to
               obtain the MT loss.
            5. Returns total_loss = MT_loss + ot_lambda * OT_loss.
        """
        # ── Pure-video fallback ────────────────────────────────────────────
        if input_audio is None or self.audio_feature_extractor is None:
            return super().forward(
                input_frames=input_frames,
                encoder_prompt=encoder_prompt,
                encoder_prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ── Audio+video training path ──────────────────────────────────────
        if encoder_outputs is None:
            if labels is not None:
                decoder_input_ids = None
                decoder_attention_mask = None

            # --- Video pipeline (FeatureExtractor + MultimodalMapper) ---
            if inputs_embeds is None and input_frames is not None:
                if self.feature_extractor is None:
                    inputs_embeds = input_frames
                else:
                    inputs_embeds = self.feature_extractor(input_frames)
                    # Whisper always emits fixed-length output; reset mask.
                    if attention_mask is not None:
                        B, T = inputs_embeds.shape[:2]
                        attention_mask = torch.ones(
                            (B, T), dtype=attention_mask.dtype, device=attention_mask.device
                        )

            if self.multimodal_mapper is not None and inputs_embeds is not None:
                inputs_embeds, attention_mask = self.multimodal_mapper(inputs_embeds, attention_mask)
            # inputs_embeds: [B, T_v, D], attention_mask: [B, T_v]

            # --- Audio pipeline (frozen encoder + projection) ---
            with torch.no_grad() if self.config.freeze_audio_feature_extractor else torch.enable_grad():
                audio_repr = self.audio_feature_extractor(input_audio)  # [B, T_a, D_audio]

            # Whisper encoder always outputs fixed-length; mask is all-ones.
            B_a, T_a = audio_repr.shape[:2]
            audio_mask = torch.ones(
                (B_a, T_a), dtype=torch.long, device=audio_repr.device
            )

            audio_repr = self.audio_projection(audio_repr)  # [B, T_a, D]

            # --- Sinkhorn OT loss ---
            ot_loss = batch_sinkhorn_loss(
                x=inputs_embeds,
                y=audio_repr,
                x_mask=attention_mask,
                y_mask=audio_mask,
                epsilon=self.config.sinkhorn_epsilon,
                max_iter=self.config.sinkhorn_max_iter,
            )

            # --- merge_modalities + backbone (video path continues) ---
            if inputs_embeds is None:
                inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
                input_ids = None

            inputs_embeds, attention_mask = merge_modalities(
                x=inputs_embeds,
                padding_mask=attention_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )
        else:
            # encoder_outputs provided (e.g. cached) — use parent's mask correction.
            if self.multimodal_mapper is not None:
                attention_mask = self.multimodal_mapper.mask_correction(attention_mask)
            from multimodalhugs.modules.utils import merge_modalities_mask_correction
            attention_mask = merge_modalities_mask_correction(
                padding_mask=attention_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )
            # No OT loss when encoder_outputs are cached.
            ot_loss = torch.tensor(0.0, device=attention_mask.device if attention_mask is not None else torch.device("cpu"))

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds if encoder_outputs is None else None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        mt_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=ot_loss.device)
        total_loss = mt_loss + self.config.ot_lambda * ot_loss

        if not (return_dict if return_dict is not None else True):
            output = (outputs.logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # ------------------------------------------------------------------
    # prepare_inputs_for_generation — generation uses pure video mode
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # input_audio is not relevant during generation; strip it if present.
        kwargs.pop("input_audio", None)
        kwargs.pop("audio_attention_mask", None)
        return super().prepare_inputs_for_generation(*args, **kwargs)
