# Standard Library Imports
import math

# Third-Party Imports
import torch
import torch.nn.functional as F


def sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    x_mask: torch.Tensor = None,
    y_mask: torch.Tensor = None,
    epsilon: float = 0.1,
    max_iter: int = 100,
    normalize: bool = True,
) -> torch.Tensor:
    """
    **Sinkhorn OT loss between two sets of vectors (log-domain formulation).**

    Computes the entropic optimal transport distance between the video and audio
    frame representations of a single sample. The log-domain implementation avoids
    numerical underflow when epsilon is small.

    **Args:**
    - `x` (torch.Tensor): `[T_x, D]` — video frame representations.
    - `y` (torch.Tensor): `[T_y, D]` — audio frame representations.
    - `x_mask` (torch.Tensor): `[T_x]` bool/long, True/1 for valid frames.
    - `y_mask` (torch.Tensor): `[T_y]` bool/long, True/1 for valid frames.
    - `epsilon` (float): Entropy regularization. Smaller → closer to exact OT.
    - `max_iter` (int): Number of Sinkhorn iterations.
    - `normalize` (bool): Project onto unit sphere before computing cost matrix.

    **Returns:**
    - `torch.Tensor`: Scalar Sinkhorn loss, differentiable w.r.t. x and y.
    """
    if normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

    if x_mask is not None:
        x = x[x_mask.bool()]
    if y_mask is not None:
        y = y[y_mask.bool()]

    T_x, T_y = x.shape[0], y.shape[0]

    if T_x == 0 or T_y == 0:
        return (x.sum() + y.sum()) * 0.0

    # Cosine distance cost matrix: C ∈ [0, 2] for unit vectors
    C = 1.0 - (x @ y.T)  # [T_x, T_y]

    log_a = torch.full((T_x,), -math.log(T_x), device=x.device, dtype=x.dtype)
    log_b = torch.full((T_y,), -math.log(T_y), device=x.device, dtype=x.dtype)

    log_K = -C / epsilon
    log_u = torch.zeros(T_x, device=x.device, dtype=x.dtype)
    log_v = torch.zeros(T_y, device=x.device, dtype=x.dtype)

    for _ in range(max_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    transport = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return (transport * C).sum()


def batch_sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    x_mask: torch.Tensor = None,
    y_mask: torch.Tensor = None,
    epsilon: float = 0.1,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    **Sinkhorn OT loss averaged over a batch.**

    **Args:**
    - `x` (torch.Tensor): `[B, T_x, D]` — video representations.
    - `y` (torch.Tensor): `[B, T_y, D]` — audio representations.
    - `x_mask` (torch.Tensor): `[B, T_x]` — video padding mask (1 = valid).
    - `y_mask` (torch.Tensor): `[B, T_y]` — audio padding mask (1 = valid).
    - `epsilon` (float): Entropy regularization strength.
    - `max_iter` (int): Number of Sinkhorn iterations.

    **Returns:**
    - `torch.Tensor`: Scalar mean Sinkhorn loss across the batch.
    """
    B = x.shape[0]
    losses = []
    for i in range(B):
        xm = x_mask[i] if x_mask is not None else None
        ym = y_mask[i] if y_mask is not None else None
        loss_i = sinkhorn_loss(x[i], y[i], xm, ym, epsilon=epsilon, max_iter=max_iter)
        losses.append(loss_i)
    return torch.stack(losses).mean()
