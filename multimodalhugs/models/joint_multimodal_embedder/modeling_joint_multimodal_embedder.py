"""
JointMultiModalEmbedderModel

Fusion model that feeds BOTH video and audio representations to the M2M backbone.
OT alignment is applied at the mapper output before fusion.

Training (input_frames + input_audio + labels):
    Video path:  FeatureExtractor → MultimodalMapper → video_repr [B, T_v, D]
    Audio path:  AudioFeatureExtractor → AudioMultimodalMapper → audio_repr [B, T_a, D]
    OT:          batch_sinkhorn_loss(video_repr, audio_repr) → ot_loss
    Fusion:      cat([video_repr, audio_repr], dim=1) → fused_repr [B, T_v+T_a, D]
    Backbone:    merge_modalities(fused_repr) → M2M → mt_loss
    Total loss:  mt_loss + ot_lambda * ot_loss

Video-only inference (input_frames, input_audio=None):
    Identical to parent MultiModalEmbedderModel.

Audio-only inference (input_audio, input_frames=None):
    AudioFE → AudioMapper → merge_modalities → M2M (pure MT loss).
"""
import logging
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from multimodalhugs.models.siamese_multimodal_embedder.modeling_siamese_multimodal_embedder import (
    SiameseMultiModalEmbedderModel,
)
from multimodalhugs.models.joint_multimodal_embedder.configuration_joint_multimodal_embedder import (
    JointMultiModalEmbedderConfig,
)
from multimodalhugs.modules import MultimodalMapper
from multimodalhugs.modules.sinkhorn import batch_sinkhorn_loss
from multimodalhugs.modules.utils import merge_modalities
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model("joint_multimodal_embedder")
class JointMultiModalEmbedderModel(SiameseMultiModalEmbedderModel):
    """
    Fusion model: video + audio concatenated along the sequence dim before M2M.

    Inherits from SiameseMultiModalEmbedderModel to reuse:
      - audio branch init (_init_audio_branch)
      - audio mask helper (_audio_valid_mask)
      - audio-only forward (_forward_audio_only)
      - speech checkpoint warm-start (_load_from_speech_checkpoint)

    Overrides:
      - forward()                       — joint fusion path
      - input_to_encoder_outputs()      — fused encoder for generate()
      - prepare_inputs_for_generation() — pass both inputs to the encoder step
    """

    config_class = JointMultiModalEmbedderConfig

    # ------------------------------------------------------------------
    # Encoder routing for generate() — handles video-only, audio-only,
    # and the new joint (fused) mode.
    # ------------------------------------------------------------------

    def input_to_encoder_outputs(
        self,
        input_frames: Optional[torch.Tensor] = None,
        input_audio: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # ── Audio-only ──────────────────────────────────────────────────
        if input_frames is None and input_audio is not None:
            # Reuse parent siamese implementation.
            return super().input_to_encoder_outputs(
                input_frames=None,
                input_audio=input_audio,
                audio_attention_mask=audio_attention_mask,
                encoder_prompt=encoder_prompt,
                encoder_prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ── Video-only ──────────────────────────────────────────────────
        if input_audio is None or self.audio_feature_extractor is None:
            return super().input_to_encoder_outputs(
                input_frames=input_frames,
                encoder_prompt=encoder_prompt,
                encoder_prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ── Joint fused mode (both inputs) ─────────────────────────────
        # Video branch
        if inputs_embeds is None and input_frames is not None:
            if self.feature_extractor is not None:
                inputs_embeds = self.feature_extractor(input_frames)
                if attention_mask is not None:
                    B, T = inputs_embeds.shape[:2]
                    attention_mask = torch.ones(
                        (B, T), dtype=attention_mask.dtype, device=attention_mask.device
                    )
            else:
                inputs_embeds = input_frames

        if self.multimodal_mapper is not None and inputs_embeds is not None:
            inputs_embeds, attention_mask = self.multimodal_mapper(inputs_embeds, attention_mask)

        # Audio branch
        if input_audio.shape[-1] < 3000:
            input_audio = F.pad(input_audio, (0, 3000 - input_audio.shape[-1]))
        with torch.no_grad() if self.config.audio_freeze_feature_extractor else torch.enable_grad():
            audio_repr = self.audio_feature_extractor(input_audio)

        B_a, T_a = audio_repr.shape[:2]
        audio_mask = torch.ones((B_a, T_a), dtype=torch.long, device=audio_repr.device)

        if isinstance(self.audio_mapper, MultimodalMapper):
            audio_repr, audio_mask = self.audio_mapper(audio_repr, audio_mask)
        elif self.audio_mapper is not None:
            audio_repr = self.audio_mapper(audio_repr)

        # Fuse: [B, T_v + T_a, D] and [B, T_v + T_a]
        fused_repr = torch.cat([inputs_embeds, audio_repr], dim=1)
        fused_mask = torch.cat([attention_mask, audio_mask], dim=1)

        fused_embeds, enc_attention_mask = merge_modalities(
            x=fused_repr,
            padding_mask=fused_mask,
            prompt=encoder_prompt,
            prompt_length_padding_mask=encoder_prompt_length_padding_mask,
            embeddings_module=self.get_backbone_encoder.embed_tokens,
            pad_idx=self.pad_token_id,
            eos_idx=self.eos_token_id,
        )

        return self.get_backbone_encoder(
            input_ids=None,
            attention_mask=enc_attention_mask,
            head_mask=head_mask,
            inputs_embeds=fused_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else True,
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
        When input_audio is None  → pure video mode (identical to parent).
        When input_frames is None → audio-only mode (MT loss only, no OT).
        When both are provided    → joint fusion: cat(video, audio) → M2M + OT loss.
        """
        # ── Pure-video fallback ────────────────────────────────────────
        if input_audio is None or self.audio_feature_extractor is None:
            return super(SiameseMultiModalEmbedderModel, self).forward(
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

        # ── Audio-only mode ────────────────────────────────────────────
        if input_frames is None:
            return self._forward_audio_only(
                input_audio=input_audio,
                attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
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

        # ── Joint fusion training path ─────────────────────────────────
        if encoder_outputs is None:
            ot_loss = torch.tensor(0.0, device=next(self.parameters()).device)

            if labels is not None:
                decoder_input_ids = None
                decoder_attention_mask = None

            # --- Video: FeatureExtractor ---
            if inputs_embeds is None and input_frames is not None:
                if self.feature_extractor is None:
                    inputs_embeds = input_frames
                else:
                    inputs_embeds = self.feature_extractor(input_frames)
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
            if input_audio.shape[-1] < 3000:
                input_audio = F.pad(input_audio, (0, 3000 - input_audio.shape[-1]))
            with torch.no_grad() if self.config.audio_freeze_feature_extractor else torch.enable_grad():
                audio_repr = self.audio_feature_extractor(input_audio)  # [B, T_a, D_audio]

            # Whisper always outputs fixed-length; build all-ones mask in encoder output space.
            B_a, T_a = audio_repr.shape[:2]
            audio_mask = torch.ones((B_a, T_a), dtype=torch.long, device=audio_repr.device)

            # --- Audio: Mapper ---
            if isinstance(self.audio_mapper, MultimodalMapper):
                audio_repr, audio_mask = self.audio_mapper(audio_repr, audio_mask)
            elif self.audio_mapper is not None:
                audio_repr = self.audio_mapper(audio_repr)
            # audio_repr: [B, T_a, D],  audio_mask: [B, T_a]

            # --- OT loss (at mapper output, before fusion) ---
            if self.config.ot_lambda > 0.0:
                ot_loss = batch_sinkhorn_loss(
                    x=inputs_embeds,
                    y=audio_repr,
                    x_mask=attention_mask,
                    y_mask=audio_mask,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

            # --- Fuse video + audio along sequence dim ---
            fused_repr = torch.cat([inputs_embeds, audio_repr], dim=1)   # [B, T_v+T_a, D]
            fused_mask = torch.cat([attention_mask, audio_mask], dim=1)  # [B, T_v+T_a]

            # --- merge_modalities: prepend encoder_prompt + EOS ---
            fused_embeds, enc_attention_mask = merge_modalities(
                x=fused_repr,
                padding_mask=fused_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )

            outputs = self.backbone(
                input_ids=None,
                attention_mask=enc_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=None,
                past_key_values=past_key_values,
                inputs_embeds=fused_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        else:
            # Cached encoder outputs (autoregressive decode steps).
            # Reconstruct attention_mask from encoder output length; the mask
            # passed in by transformers may be stale (e.g. prompt length only).
            B = encoder_outputs[0].shape[0]
            T = encoder_outputs[0].shape[1]
            enc_attention_mask = torch.ones(
                (B, T), dtype=torch.long, device=encoder_outputs[0].device
            )
            ot_loss = torch.tensor(0.0, device=next(self.parameters()).device)

            outputs = self.backbone(
                input_ids=None,
                attention_mask=enc_attention_mask,
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
        total_loss = mt_loss + self.config.ot_lambda * ot_loss

        # Expose component losses for WandB logging (read by MultiLingualSeq2SeqTrainer).
        # When label_smoothing > 0 the trainer pops labels, so mt_loss is 0 here and the
        # trainer overwrites _last_mt_loss with the real smoothed MT loss afterwards.
        if self.training:
            self._last_ot_loss = float(ot_loss.detach().item())
            self._last_mt_loss = float(mt_loss.detach().item())

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
    # Generation — keep both inputs until encoder_outputs is computed
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        input_audio = kwargs.get("input_audio", None)
        audio_attention_mask = kwargs.get("audio_attention_mask", None)

        # Strip audio from kwargs so the parent (MultiModalEmbedderModel) doesn't
        # choke on the unknown argument, then re-inject after.
        kwargs.pop("input_audio", None)
        kwargs.pop("audio_attention_mask", None)

        # Call grandparent (MultiModalEmbedderModel) directly — the siamese
        # override would strip audio when input_frames is present.
        from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel
        model_inputs = MultiModalEmbedderModel.prepare_inputs_for_generation(self, *args, **kwargs)

        if input_audio is not None:
            model_inputs["input_audio"] = input_audio
        if audio_attention_mask is not None:
            model_inputs["audio_attention_mask"] = audio_attention_mask

        return model_inputs
