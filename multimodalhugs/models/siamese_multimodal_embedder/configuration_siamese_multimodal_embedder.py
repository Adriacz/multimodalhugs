from typing import Optional

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


class SiameseMultiModalEmbedderConfig(MultiModalEmbedderConfig):
    """
    Configuration for SiameseMultiModalEmbedderModel.

    Extends MultiModalEmbedderConfig with a second (audio) feature extractor,
    an audio-specific MultimodalMapper, and Optimal Transport alignment params.

    Video pathway (unchanged from parent):
        FeatureExtractor → MultimodalMapper → [merge_modalities →] Backbone

    Audio pathway (added):
        AudioFeatureExtractor → AudioMultimodalMapper → OT loss (then discarded)

    OT can be applied at two positions, controlled by ``ot_position``:
        ``"mapper"``  — between video_repr and audio_repr after their mappers
                        (d_model space, before merge_modalities / backbone encoder).
        ``"encoder"`` — between the M2M encoder outputs of both modalities
                        (backbone encoder is run explicitly for both paths;
                        audio encoder pass uses no_grad since backbone is frozen).

    Args:
        audio_feature_extractor_type: Type string for the audio encoder,
            e.g. ``"whisper"``.  When None the audio branch is disabled.
        audio_pretrained_feature_extractor: HF model ID or local path,
            e.g. ``"openai/whisper-medium"``.
        audio_feat_dim: Output dim of the audio feature extractor.
            Whisper-medium → 1024.
        audio_freeze_feature_extractor: Freeze the audio encoder weights.
        audio_multimodal_mapper_type: Mapper type for the audio branch —
            one of ``{"linear", "adapter", "cnn_adapter"}``.
        audio_multimodal_mapper_layer_norm_before: LayerNorm before mapper.
        audio_multimodal_mapper_layer_norm: LayerNorm inside mapper.
        audio_multimodal_mapper_activation: ReLU at mapper output.
        audio_multimodal_mapper_factor: Overparameterization factor (adapter).
        audio_multimodal_mapper_dropout: Dropout in the audio mapper.
        audio_freeze_multimodal_mapper: Freeze audio mapper weights.
        pretrained_speech_checkpoint: Path to a MultiModalEmbedder speech2text checkpoint.
            When set, backbone weights are loaded from this checkpoint (overriding
            ``pretrained_backbone``), and — if the audio branch is enabled — the
            audio feature extractor and audio mapper weights are also loaded from the
            checkpoint's ``feature_extractor`` and ``multimodal_mapper`` keys respectively.
            Intended for Phase 2 video training that warm-starts from a Phase 1 speech model.
        ot_lambda: Weight for the Sinkhorn OT loss applied at the mapper or encoder output.
            When ``ot_position="both"`` this weight applies to the mapper-level OT term.
        ot_lambda_encoder: Weight for the encoder-level OT term when ``ot_position="both"``.
            Ignored for ``"mapper"`` and ``"encoder"`` positions.
        sinkhorn_epsilon: Entropy regularization for Sinkhorn.
        sinkhorn_max_iter: Number of Sinkhorn iterations.
        ot_position: Where OT is applied — ``"mapper"`` (default), ``"encoder"``, or
            ``"both"`` (mapper + encoder simultaneously).
        ot_pre_norm: If True, apply a shared LayerNorm to both video and audio
            representations immediately before computing mapper-level OT loss.
            Aligns the mean and std of both spaces, which stabilises OT when
            the two mappers have not yet converged to the same scale.
        eval_audio: Run a second audio-only eval pass during training.
    """

    model_type = "siamese_multimodal_embedder"

    def __init__(
        self,
        model_type: str = "siamese_multimodal_embedder",
        # --- Speech checkpoint warm-start ---
        pretrained_speech_checkpoint: Optional[str] = None,
        # --- Audio feature extractor ---
        audio_feature_extractor_type: Optional[str] = None,
        audio_pretrained_feature_extractor: Optional[str] = None,
        audio_feat_dim: int = 1024,
        audio_freeze_feature_extractor: bool = True,
        # --- Audio multimodal mapper ---
        audio_multimodal_mapper_type: Optional[str] = None,
        audio_multimodal_mapper_layer_norm_before: bool = False,
        audio_multimodal_mapper_layer_norm: bool = False,
        audio_multimodal_mapper_activation: bool = False,
        audio_multimodal_mapper_factor: Optional[int] = None,
        audio_multimodal_mapper_dropout: Optional[float] = None,
        audio_freeze_multimodal_mapper: bool = False,
        # --- OT alignment ---
        ot_lambda: float = 1.0,
        ot_lambda_encoder: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        ot_position: str = "mapper",
        ot_pre_norm: bool = False,
        # --- Eval ---
        eval_audio: bool = True,
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)

        self.pretrained_speech_checkpoint = pretrained_speech_checkpoint

        self.audio_feature_extractor_type = audio_feature_extractor_type
        self.audio_pretrained_feature_extractor = audio_pretrained_feature_extractor
        self.audio_feat_dim = audio_feat_dim
        self.audio_freeze_feature_extractor = audio_freeze_feature_extractor

        self.audio_multimodal_mapper_type = audio_multimodal_mapper_type
        self.audio_multimodal_mapper_layer_norm_before = audio_multimodal_mapper_layer_norm_before
        self.audio_multimodal_mapper_layer_norm = audio_multimodal_mapper_layer_norm
        self.audio_multimodal_mapper_activation = audio_multimodal_mapper_activation
        self.audio_multimodal_mapper_factor = audio_multimodal_mapper_factor
        self.audio_multimodal_mapper_dropout = audio_multimodal_mapper_dropout
        self.audio_freeze_multimodal_mapper = audio_freeze_multimodal_mapper

        self.ot_lambda = ot_lambda
        self.ot_lambda_encoder = ot_lambda_encoder
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.ot_position = ot_position
        self.ot_pre_norm = ot_pre_norm
        self.eval_audio = eval_audio
