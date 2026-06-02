# Standard Library Imports
from typing import Dict, Sequence

# Third-Party Imports
import torch
import torch.nn.functional as F


def cross_modal_retrieval(
    video_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    **Cross-modal retrieval metrics (video → audio).**

    For each video embedding, ranks all audio embeddings by cosine similarity.
    A retrieval hit at rank k occurs when the paired audio (same index i) appears
    within the top-k most similar audio embeddings for video i.

    **Args:**
    - `video_embs` (torch.Tensor): `[N, D]` — video embeddings.
    - `audio_embs` (torch.Tensor): `[N, D]` — audio embeddings, paired with video (index i ↔ i).
    - `ks` (Sequence[int]): k values for R@k metrics.

    **Returns:**
    - `Dict[str, float]`: Keys ``R@k`` for each k in ks, plus ``MRR`` and ``median_rank``.
    """
    N = video_embs.shape[0]
    video_norm = F.normalize(video_embs, dim=-1)
    audio_norm = F.normalize(audio_embs, dim=-1)

    sim = video_norm @ audio_norm.T  # [N, N]

    sorted_idx = sim.argsort(dim=1, descending=True)  # [N, N]
    gt = torch.arange(N, device=sim.device).unsqueeze(1)  # [N, 1]
    ranks = (sorted_idx == gt).nonzero(as_tuple=False)[:, 1] + 1  # [N], 1-indexed

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = (ranks <= k).float().mean().item()
    metrics["MRR"] = (1.0 / ranks.float()).mean().item()
    metrics["median_rank"] = ranks.float().median().item()
    return metrics
