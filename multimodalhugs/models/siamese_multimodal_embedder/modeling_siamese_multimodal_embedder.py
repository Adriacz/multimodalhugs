"""
SiameseMultiModalEmbedderModel

A two-pathway model for audio+video sign language translation pretraining.

Phase 1 — forward(input_frames=..., input_audio=..., labels=...):
    Video path:  FeatureExtractor → MultimodalMapper → merge_modalities → Backbone (MT loss)
    Audio path:  AudioFeatureExtractor → AudioMultimodalMapper
    Alignment:   OT loss between video_repr and audio_repr in d_model space.
    Total loss:  MT_loss + ot_lambda * OT_loss

Phase 2a / video inference — forward(input_frames=..., input_audio=None, labels=...):
    Pure video forward, identical to the parent MultiModalEmbedderModel.

Phase 2b / audio inference — forward(input_frames=None, input_audio=..., labels=...):
    Audio-only forward: AudioFE → AudioMapper → merge_modalities → Backbone (MT loss only).
    Used for the separate audio-only eval pass during training.
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
from multimodalhugs.modules import FeatureExtractor, MultimodalMapper
from multimodalhugs.modules.sinkhorn import batch_sinkhorn_loss
from multimodalhugs.modules.utils import set_module_parameters, merge_modalities, merge_modalities_mask_correction
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model("siamese_multimodal_embedder")
class SiameseMultiModalEmbedderModel(MultiModalEmbedderModel):
    """
    Siamese multimodal model with OT-based audio↔video alignment.

    Inherits the full video pipeline from MultiModalEmbedderModel.
    When ``input_audio`` is provided in forward(), the audio branch runs
    in parallel and contributes a Sinkhorn OT alignment loss.
    When ``input_audio`` is None the call is identical to the parent class.
    """

    config_class = SiameseMultiModalEmbedderConfig

    def __init__(self, config: SiameseMultiModalEmbedderConfig):
        super().__init__(config)
        self._init_audio_branch(config)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Speech checkpoint warm-start
    # ------------------------------------------------------------------

    @classmethod
    def build_model(cls, **kwargs):
        """Build model then optionally load backbone + audio weights from a speech checkpoint."""
        model = super().build_model(**kwargs)
        if getattr(model.config, "pretrained_speech_checkpoint", None):
            model._load_from_speech_checkpoint(model.config.pretrained_speech_checkpoint)
        return model

    def _load_from_speech_checkpoint(self, path: str) -> None:
        """Load backbone + audio branch weights from a MultiModalEmbedder speech2text checkpoint.

        Key remapping applied to the speech checkpoint state dict:
          feature_extractor.*  → audio_feature_extractor.*
          multimodal_mapper.*  → audio_mapper.*
          backbone.*           → backbone.*   (unchanged)

        Uses strict=False so keys absent from this model (e.g. audio branch keys
        when audio_feature_extractor_type is None) are silently ignored.
        """
        from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel

        logger.info("Loading weights from speech checkpoint: %s", path)
        speech_model = MultiModalEmbedderModel.from_pretrained(path)
        raw_state = {k: v.clone() for k, v in speech_model.state_dict().items()}
        del speech_model

        remapped: dict = {}
        for key, val in raw_state.items():
            if key.startswith("feature_extractor."):
                remapped["audio_feature_extractor." + key[len("feature_extractor."):]] = val
            elif key.startswith("multimodal_mapper."):
                remapped["audio_mapper." + key[len("multimodal_mapper."):]] = val
            else:
                remapped[key] = val

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        n_loaded = len(remapped) - len(unexpected)
        logger.info(
            "Speech checkpoint loaded — %d weights applied, %d missing, %d unexpected",
            n_loaded, len(missing), len(unexpected),
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_audio_branch(self, config: SiameseMultiModalEmbedderConfig):
        if config.audio_feature_extractor_type is None:
            self.audio_feature_extractor = None
            self.audio_mapper = None
            return

        self.audio_feature_extractor = FeatureExtractor(
            feature_extractor_type=config.audio_feature_extractor_type,
            pretrained_module=config.audio_pretrained_feature_extractor,
        )
        set_module_parameters(
            self.audio_feature_extractor,
            freeze=config.audio_freeze_feature_extractor,
        )

        if config.audio_multimodal_mapper_type is not None:
            self.audio_mapper = MultimodalMapper(
                feat_dim=config.audio_feat_dim,
                output_dim=config.d_model,
                mapping_layer_type=config.audio_multimodal_mapper_type,
                layer_norm_before=config.audio_multimodal_mapper_layer_norm_before,
                adapter_factor=config.audio_multimodal_mapper_factor,
                p_dropout=config.audio_multimodal_mapper_dropout,
                layer_norm=config.audio_multimodal_mapper_layer_norm,
                activation=config.audio_multimodal_mapper_activation,
            )
            set_module_parameters(
                self.audio_mapper,
                freeze=config.audio_freeze_multimodal_mapper,
            )
        else:
            # No mapper config → plain linear projection as fallback.
            self.audio_mapper = (
                nn.Linear(config.audio_feat_dim, config.d_model)
                if config.audio_feat_dim != config.d_model
                else nn.Identity()
            )

    # ------------------------------------------------------------------
    # Audio-only forward (eval / inference)
    # ------------------------------------------------------------------

    def _forward_audio_only(
        self,
        input_audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_prompt: Optional[torch.LongTensor],
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor],
        decoder_input_ids: Optional[torch.LongTensor],
        decoder_attention_mask: Optional[torch.LongTensor],
        head_mask: Optional[torch.Tensor],
        decoder_head_mask: Optional[torch.Tensor],
        cross_attn_head_mask: Optional[torch.Tensor],
        encoder_outputs: Optional[Tuple],
        past_key_values: Optional[Tuple],
        decoder_inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.LongTensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """Audio-only forward: AudioFE → AudioMapper → merge_modalities → Backbone (pure MT loss)."""
        inputs_embeds = None

        if encoder_outputs is None:
            if labels is not None:
                decoder_input_ids = None
                decoder_attention_mask = None

            with torch.no_grad() if self.config.audio_freeze_feature_extractor else torch.enable_grad():
                audio_repr = self.audio_feature_extractor(input_audio)

            B_a, T_a = audio_repr.shape[:2]
            audio_mask = torch.ones((B_a, T_a), dtype=torch.long, device=audio_repr.device)

            if isinstance(self.audio_mapper, MultimodalMapper):
                audio_repr, audio_mask = self.audio_mapper(audio_repr, audio_mask)
            elif self.audio_mapper is not None:
                audio_repr = self.audio_mapper(audio_repr)

            inputs_embeds, attention_mask = merge_modalities(
                x=audio_repr,
                padding_mask=audio_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )
        else:
            # Cached encoder outputs: reconstruct attention_mask to full length.
            if attention_mask is None:
                B = encoder_outputs[0].shape[0]
                T = encoder_outputs[0].shape[1]
                attention_mask = torch.ones(
                    (B, T), dtype=torch.long, device=encoder_outputs[0].device
                )
            else:
                if isinstance(self.audio_mapper, MultimodalMapper):
                    attention_mask = self.audio_mapper.mask_correction(attention_mask)
                attention_mask = merge_modalities_mask_correction(
                    padding_mask=attention_mask,
                    prompt=encoder_prompt,
                    prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                    embeddings_module=self.get_backbone_encoder.embed_tokens,
                    pad_idx=self.pad_token_id,
                    eos_idx=self.eos_token_id,
                )

        outputs = self.backbone(
            input_ids=None,
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

        if not (return_dict if return_dict is not None else True):
            output = (outputs.logits,) + outputs[1:]
            return (outputs.loss,) + output if outputs.loss is not None else output

        return Seq2SeqLMOutput(
            loss=outputs.loss,
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
        When ``input_audio`` is None → pure video mode (identical to parent).
        When ``input_frames`` is None and ``input_audio`` is provided → audio-only mode (MT loss only).
        When both are provided → Siamese training mode (OT + MT loss).
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

        # ── OT disabled: skip audio computation in training, fall back to pure video ──
        # Only skip when input_frames is present (training); audio-only calls
        # (input_frames=None) must fall through to _forward_audio_only below.
        if self.config.ot_lambda == 0.0 and self.config.ot_lambda_encoder == 0.0 and input_frames is not None:
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

        # ── Audio-only mode (eval / inference) ────────────────────────────
        if input_frames is None:
            return self._forward_audio_only(
                input_audio=input_audio,
                attention_mask=attention_mask,
                encoder_prompt=encoder_prompt,
                encoder_prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ── Audio + video training path ────────────────────────────────────
        if encoder_outputs is None:
            ot_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            ot_mapper_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            ot_encoder_loss = torch.tensor(0.0, device=next(self.parameters()).device)

            if labels is not None:
                decoder_input_ids = None
                decoder_attention_mask = None

            # --- Video: FeatureExtractor ---
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

            # --- Video: MultimodalMapper ---
            if self.multimodal_mapper is not None and inputs_embeds is not None:
                inputs_embeds, attention_mask = self.multimodal_mapper(inputs_embeds, attention_mask)
            # inputs_embeds: [B, T_v, D],  attention_mask: [B, T_v]

            # --- Audio: FeatureExtractor ---
            with torch.no_grad() if self.config.audio_freeze_feature_extractor else torch.enable_grad():
                audio_repr = self.audio_feature_extractor(input_audio)  # [B, T_a, D_audio]

            # Whisper always outputs a fixed-length sequence regardless of input;
            # the incoming audio_attention_mask reflects mel padding, not encoder
            # output length.  Reset to all-ones in encoder output space.
            B_a, T_a = audio_repr.shape[:2]
            audio_mask = torch.ones(
                (B_a, T_a), dtype=torch.long, device=audio_repr.device
            )

            # --- Audio: Mapper (MultimodalMapper or plain Linear) ---
            if isinstance(self.audio_mapper, MultimodalMapper):
                audio_repr, audio_mask = self.audio_mapper(audio_repr, audio_mask)
            elif self.audio_mapper is not None:
                audio_repr = self.audio_mapper(audio_repr)

            if self.config.ot_position == "encoder":
                # ── OT at M2M encoder output ───────────────────────────────
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
                video_enc_out = self.get_backbone_encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )

                with torch.no_grad():
                    audio_inputs_embeds, audio_enc_attn = merge_modalities(
                        x=audio_repr,
                        padding_mask=audio_mask,
                        prompt=encoder_prompt,
                        prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                        embeddings_module=self.get_backbone_encoder.embed_tokens,
                        pad_idx=self.pad_token_id,
                        eos_idx=self.eos_token_id,
                    )
                    audio_enc_out = self.get_backbone_encoder(
                        inputs_embeds=audio_inputs_embeds,
                        attention_mask=audio_enc_attn,
                    )

                ot_loss = batch_sinkhorn_loss(
                    x=video_enc_out.last_hidden_state,
                    y=audio_enc_out.last_hidden_state,
                    x_mask=attention_mask,
                    y_mask=audio_enc_attn,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

                outputs = self.backbone(
                    input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=video_enc_out,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

            elif self.config.ot_position == "both":
                # ── OT at mapper output AND at M2M encoder output ──────────
                ot_mapper_loss = batch_sinkhorn_loss(
                    x=inputs_embeds,
                    y=audio_repr,
                    x_mask=attention_mask,
                    y_mask=audio_mask,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

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
                video_enc_out = self.get_backbone_encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )

                with torch.no_grad():
                    audio_inputs_embeds, audio_enc_attn = merge_modalities(
                        x=audio_repr,
                        padding_mask=audio_mask,
                        prompt=encoder_prompt,
                        prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                        embeddings_module=self.get_backbone_encoder.embed_tokens,
                        pad_idx=self.pad_token_id,
                        eos_idx=self.eos_token_id,
                    )
                    audio_enc_out = self.get_backbone_encoder(
                        inputs_embeds=audio_inputs_embeds,
                        attention_mask=audio_enc_attn,
                    )

                ot_encoder_loss = batch_sinkhorn_loss(
                    x=video_enc_out.last_hidden_state,
                    y=audio_enc_out.last_hidden_state,
                    x_mask=attention_mask,
                    y_mask=audio_enc_attn,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

                outputs = self.backbone(
                    input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=video_enc_out,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

            else:
                # ── OT at mapper output (default) ──────────────────────────
                ot_loss = batch_sinkhorn_loss(
                    x=inputs_embeds,
                    y=audio_repr,
                    x_mask=attention_mask,
                    y_mask=audio_mask,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

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

                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=None,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

        else:
            # Cached encoder outputs — apply mask corrections only; no OT loss.
            if self.multimodal_mapper is not None:
                attention_mask = self.multimodal_mapper.mask_correction(attention_mask)
            attention_mask = merge_modalities_mask_correction(
                padding_mask=attention_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )
            ot_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            ot_mapper_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            ot_encoder_loss = torch.tensor(0.0, device=next(self.parameters()).device)

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
                inputs_embeds=None,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        mt_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)
        if self.config.ot_position == "both":
            total_loss = mt_loss + self.config.ot_lambda * ot_mapper_loss + self.config.ot_lambda_encoder * ot_encoder_loss
        else:
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
    # Generation — video-only or audio-only depending on which input is present
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if kwargs.get("input_frames") is not None:
            # Video mode: strip audio before delegating to parent.
            kwargs.pop("input_audio", None)
            kwargs.pop("audio_attention_mask", None)
            return super().prepare_inputs_for_generation(*args, **kwargs)

        # Audio-only mode: let parent run (won't add input_frames since it's None),
        # then inject input_audio into the returned model_inputs dict.
        input_audio = kwargs.get("input_audio", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if input_audio is not None:
            model_inputs["input_audio"] = input_audio
        return model_inputs
