from typing import Optional

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


class AudioTeacherVideoStudentConfig(MultiModalEmbedderConfig):
    """
    Configuration for AudioTeacherVideoStudentModel.

    Extends MultiModalEmbedderConfig (the video student) with:
    - A frozen audio teacher loaded from ``teacher_model_path`` at init time.
      The teacher is a Phase-1 MultiModalEmbedderModel (Whisper + AudioMapper + M2M).
      Its weights are NOT saved in the student checkpoint.
    - Sinkhorn OT alignment at two points:
        1. Mapper output: VideoMapper output vs AudioMapper output (d_model space).
        2. Encoder output: student M2M encoder last hidden state vs teacher M2M encoder
           last hidden state (after merge_modalities).

    Training curriculum:
        Stage 1 — freeze_feature_extractor=True  (CLIP frozen, mapper+backbone train)
        Stage 2 — freeze_feature_extractor=False (CLIP unfrozen, lower LR)

    Args:
        teacher_model_path: Local path or HF id of the Phase-1 audio checkpoint.
            Must be a MultiModalEmbedderModel (Whisper+Mapper+M2M) trained on the
            target domain.  The teacher is always loaded frozen in eval mode.
        ot_lambda_mapper: Weight for the OT loss at mapper output.
        ot_lambda_encoder: Weight for the OT loss at M2M encoder output.
        sinkhorn_epsilon: Entropy regularisation for Sinkhorn iterations.
        sinkhorn_max_iter: Number of Sinkhorn iterations.
    """

    model_type = "audio_teacher_video_student"

    def __init__(
        self,
        teacher_model_path: Optional[str] = None,
        ot_lambda_mapper: float = 1.0,
        ot_lambda_encoder: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_max_iter: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model_path = teacher_model_path
        self.ot_lambda_mapper = ot_lambda_mapper
        self.ot_lambda_encoder = ot_lambda_encoder
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
