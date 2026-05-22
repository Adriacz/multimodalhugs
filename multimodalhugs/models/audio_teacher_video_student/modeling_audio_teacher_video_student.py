"""
AudioTeacherVideoStudentModel

Video student trained with a frozen audio teacher as a domain anchor.

Architecture
------------
Teacher (frozen, Phase-1 checkpoint — Whisper + AudioMapper + M2M):
    input_audio → FeatureExtractor → MultimodalMapper → [mapper_repr_T]
                                                       → merge_modalities → M2M encoder → [encoder_repr_T]

Student (trainable — CLIP + VideoMapper + M2M, backbone init from Phase-1):
    input_frames → FeatureExtractor → MultimodalMapper → [mapper_repr_S]
                                                        → merge_modalities → M2M encoder → [encoder_repr_S]
                                                                           → M2M decoder → MT loss

OT losses (training only):
    ot_mapper  = Sinkhorn(mapper_repr_S,  mapper_repr_T)   # d_model space, before merge
    ot_encoder = Sinkhorn(encoder_repr_S, encoder_repr_T)  # d_model space, after M2M encoder

Total loss = MT_loss + ot_lambda_mapper * ot_mapper + ot_lambda_encoder * ot_encoder

Evaluation / generation: pure video student path (teacher is not involved).

Training curriculum
-------------------
Stage 1: freeze_feature_extractor=True  — CLIP frozen, VideoMapper+backbone train.
Stage 2: freeze_feature_extractor=False — CLIP unfrozen (lower LR recommended).
"""

import logging
import re
from typing import Optional, Tuple, Union, Dict

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput

from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel
from multimodalhugs.models.audio_teacher_video_student.configuration_audio_teacher_video_student import (
    AudioTeacherVideoStudentConfig,
)
from multimodalhugs.modules import MultimodalMapper
from multimodalhugs.modules.sinkhorn import batch_sinkhorn_loss
from multimodalhugs.modules.utils import merge_modalities, merge_modalities_mask_correction
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model("audio_teacher_video_student")
class AudioTeacherVideoStudentModel(MultiModalEmbedderModel):
    """
    Video student trained against a frozen Phase-1 audio teacher.

    The teacher is loaded from ``config.teacher_model_path`` at init time and
    excluded from the student's saved checkpoint (only student weights are saved).
    Teacher parameters are never updated.
    """

    config_class = AudioTeacherVideoStudentConfig

    # Teacher weights are loaded at runtime from teacher_model_path; they live in
    # self.teacher_model but must not appear in the saved state dict.
    _keys_to_ignore_on_load_missing = [r"^teacher_model\."]

    def __init__(self, config: AudioTeacherVideoStudentConfig):
        super().__init__(config)
        self._load_teacher(config)

    # ------------------------------------------------------------------
    # Teacher initialisation
    # ------------------------------------------------------------------

    def _load_teacher(self, config: AudioTeacherVideoStudentConfig):
        if config.teacher_model_path is None:
            logger.warning(
                "AudioTeacherVideoStudentModel: teacher_model_path is None. "
                "OT losses will be skipped during training."
            )
            self.teacher_model = None
            return

        logger.info("Loading audio teacher from %s ...", config.teacher_model_path)
        self.teacher_model = MultiModalEmbedderModel.from_pretrained(config.teacher_model_path)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        logger.info("Audio teacher loaded and frozen.")

    # ------------------------------------------------------------------
    # Exclude teacher weights from the saved checkpoint
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("teacher_model.")}

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
        Training with audio: OT(mapper) + OT(encoder) + MT loss.
        Eval / no audio / no teacher: pure video student forward (parent).
        """
        use_ot = (
            self.training
            and input_audio is not None
            and self.teacher_model is not None
            and encoder_outputs is None   # cached path skips OT (generation step 2+)
        )

        if not use_ot:
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

        # ── Training path with OT ──────────────────────────────────────

        if labels is not None:
            decoder_input_ids = None
            decoder_attention_mask = None

        # --- Student: video FE ---
        if self.feature_extractor is None:
            video_repr = input_frames
        else:
            video_repr = self.feature_extractor(input_frames)

        B, T_v = video_repr.shape[:2]
        video_mask = torch.ones((B, T_v), dtype=torch.long, device=video_repr.device)

        # --- Student: VideoMapper ---
        if self.multimodal_mapper is not None:
            video_repr, video_mask = self.multimodal_mapper(video_repr, video_mask)
        # video_repr: [B, T_v', d_model]

        # --- Teacher: audio FE + AudioMapper (frozen, no grad) ---
        with torch.no_grad():
            audio_repr = self.teacher_model.feature_extractor(input_audio)
            B_a, T_a = audio_repr.shape[:2]
            audio_mask = torch.ones((B_a, T_a), dtype=torch.long, device=audio_repr.device)

            if isinstance(self.teacher_model.multimodal_mapper, MultimodalMapper):
                audio_repr, audio_mask = self.teacher_model.multimodal_mapper(audio_repr, audio_mask)
            elif self.teacher_model.multimodal_mapper is not None:
                audio_repr = self.teacher_model.multimodal_mapper(audio_repr)
            # audio_repr: [B, T_a', d_model]

        # --- OT at mapper output ---
        ot_mapper = batch_sinkhorn_loss(
            x=video_repr,
            y=audio_repr,
            x_mask=video_mask,
            y_mask=audio_mask,
            epsilon=self.config.sinkhorn_epsilon,
            max_iter=self.config.sinkhorn_max_iter,
        )

        # --- Student: merge_modalities → backbone (full seq2seq) ---
        merged_video, merged_video_mask = merge_modalities(
            x=video_repr,
            padding_mask=video_mask,
            prompt=encoder_prompt,
            prompt_length_padding_mask=encoder_prompt_length_padding_mask,
            embeddings_module=self.get_backbone_encoder.embed_tokens,
            pad_idx=self.pad_token_id,
            eos_idx=self.eos_token_id,
        )

        backbone_out = self.backbone(
            input_ids=None,
            attention_mask=merged_video_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=None,
            past_key_values=past_key_values,
            inputs_embeds=merged_video,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        video_encoder_out = backbone_out.encoder_last_hidden_state  # [B, T_v'', d_model]

        # --- Teacher: merge_modalities → M2M encoder (frozen, no grad) ---
        with torch.no_grad():
            merged_audio, merged_audio_mask = merge_modalities(
                x=audio_repr,
                padding_mask=audio_mask,
                prompt=encoder_prompt,
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.teacher_model.get_backbone_encoder.embed_tokens,
                pad_idx=self.pad_token_id,
                eos_idx=self.eos_token_id,
            )
            teacher_enc_out = self.teacher_model.get_backbone_encoder(
                input_ids=None,
                inputs_embeds=merged_audio,
                attention_mask=merged_audio_mask,
                return_dict=True,
            )
            audio_encoder_out = teacher_enc_out.last_hidden_state  # [B, T_a'', d_model]

        # --- OT at encoder output ---
        ot_encoder = batch_sinkhorn_loss(
            x=video_encoder_out,
            y=audio_encoder_out,
            x_mask=merged_video_mask,
            y_mask=merged_audio_mask,
            epsilon=self.config.sinkhorn_epsilon,
            max_iter=self.config.sinkhorn_max_iter,
        )

        # --- Total loss ---
        mt_loss = (
            backbone_out.loss
            if backbone_out.loss is not None
            else torch.tensor(0.0, device=ot_mapper.device)
        )
        total_loss = (
            mt_loss
            + self.config.ot_lambda_mapper * ot_mapper
            + self.config.ot_lambda_encoder * ot_encoder
        )

        if not (return_dict if return_dict is not None else True):
            output = (backbone_out.logits,) + backbone_out[1:]
            return (total_loss,) + output if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=backbone_out.logits,
            past_key_values=backbone_out.past_key_values,
            decoder_hidden_states=backbone_out.decoder_hidden_states,
            decoder_attentions=backbone_out.decoder_attentions,
            cross_attentions=backbone_out.cross_attentions,
            encoder_last_hidden_state=backbone_out.encoder_last_hidden_state,
            encoder_hidden_states=backbone_out.encoder_hidden_states,
            encoder_attentions=backbone_out.encoder_attentions,
        )

    # ------------------------------------------------------------------
    # Generation — strip audio before delegating to parent
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        kwargs.pop("input_audio", None)
        kwargs.pop("audio_attention_mask", None)
        return super().prepare_inputs_for_generation(*args, **kwargs)
