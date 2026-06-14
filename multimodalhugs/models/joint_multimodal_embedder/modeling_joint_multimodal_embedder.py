"""
JointMultiModalEmbedderModel

Asymmetric multi-task model for video SLT assisted by a strong (warm-started)
audio path. Each training step with both modalities is stochastically routed:

    p_audio_only   → Audio-only ASR rehearsal (keeps the warm-started decoder
                     from forgetting; no OT). AudioFE → AudioMapper → M2M.
    p_joint_fusion → Fused step: cat([video, audio]) → M2M (+ OT). Decoder
                     sees both. Off by default (audio-crutch risk).
    remainder      → Video + OT (dominant): video → M2M (MT loss), with
                     OT(video, audio) as auxiliary alignment. The backbone
                     sees video only — this is the path used at inference.

OT is always computed at the mapper output (video_repr vs audio_repr) on the
video/fused modes for logging and transfer pressure.

Generation / eval (controlled by config.generation_mode, default "video"):
    "video" → video-only generation; eval_sacrebleu measures the real goal.
    "audio" → audio-only (also the trainer's separate audio eval pass).
    "joint" → fused video+audio generation.

Total loss (video / fused modes): mt_loss + ot_lambda * ot_loss.
"""
import logging
import random
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

        # ── Video-only (default) ────────────────────────────────────────
        # Fuse only when generation_mode == "joint". Otherwise ignore the audio
        # present in kwargs and encode video only — this is what makes the
        # standard eval pass measure video-only BLEU.
        gen_mode = getattr(self.config, "generation_mode", "video")
        if input_audio is None or self.audio_feature_extractor is None or gen_mode != "joint":
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

        # ── Joint fused mode (both inputs, generation_mode == "joint") ──
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
        Routing:
          - input_audio None  → pure video mode (identical to parent).
          - input_frames None → audio-only mode (ASR, no OT).
          - both present, cached encoder_outputs → joint decode step.
          - both present, training → stochastic multi-task sampler:
                p_audio_only   → audio-only rehearsal (no OT)
                p_joint_fusion → fused video+audio → M2M (+ OT)
                remainder      → video + OT (backbone sees video only)
          - both present, eval loss (not training) → deterministic video + OT.
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

        # ── Audio-only mode (input_frames absent: audio eval pass) ─────
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

        # ── Both modalities present ────────────────────────────────────
        # Decide the routing mode for this step.
        do_audio_only = False
        do_fuse = False
        if encoder_outputs is None and self.training:
            r = random.random()
            if r < self.config.p_audio_only:
                do_audio_only = True
            elif r < self.config.p_audio_only + self.config.p_joint_fusion:
                do_fuse = True
            # else → video + OT (default)

        # ── Audio-only rehearsal step ──────────────────────────────────
        if do_audio_only:
            self._last_ot_loss = 0.0  # no OT on audio steps; keep logging consistent
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
                encoder_outputs=None,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ── Video (+ OT), optionally fused with audio ──────────────────
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

            # --- Audio: FeatureExtractor (always — needed for OT / fusion) ---
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

            # --- OT loss (mapper output, before any fusion) ---
            if self.config.ot_lambda > 0.0:
                ot_loss = batch_sinkhorn_loss(
                    x=inputs_embeds,
                    y=audio_repr,
                    x_mask=attention_mask,
                    y_mask=audio_mask,
                    epsilon=self.config.sinkhorn_epsilon,
                    max_iter=self.config.sinkhorn_max_iter,
                )

            # --- Choose what the backbone sees: video only, or fused ---
            if do_fuse:
                backbone_repr = torch.cat([inputs_embeds, audio_repr], dim=1)
                backbone_mask = torch.cat([attention_mask, audio_mask], dim=1)
            else:
                backbone_repr = inputs_embeds
                backbone_mask = attention_mask

            merged_embeds, enc_attention_mask = merge_modalities(
                x=backbone_repr,
                padding_mask=backbone_mask,
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
                inputs_embeds=merged_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        else:
            # Cached encoder outputs (autoregressive decode steps, joint generation).
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
        gen_mode = getattr(self.config, "generation_mode", "video")

        # Joint generation: keep BOTH inputs so the encoder fuses them.
        # Only when explicitly requested and video frames are present.
        if gen_mode == "joint" and kwargs.get("input_frames") is not None:
            input_audio = kwargs.get("input_audio", None)
            audio_attention_mask = kwargs.get("audio_attention_mask", None)
            kwargs.pop("input_audio", None)
            kwargs.pop("audio_attention_mask", None)
            from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel
            model_inputs = MultiModalEmbedderModel.prepare_inputs_for_generation(self, *args, **kwargs)
            if input_audio is not None:
                model_inputs["input_audio"] = input_audio
            if audio_attention_mask is not None:
                model_inputs["audio_attention_mask"] = audio_attention_mask
            return model_inputs

        # Default: video-only generation (input_frames present) or audio-only
        # generation (input_frames absent, e.g. the audio eval pass). The siamese
        # parent strips audio when frames are present and keeps it otherwise.
        return super().prepare_inputs_for_generation(*args, **kwargs)
