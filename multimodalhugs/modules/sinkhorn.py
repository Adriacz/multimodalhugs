"""
Log-domain Sinkhorn optimal transport loss.

Used to align video encoder representations with audio encoder representations
in the Siamese pretraining phase.  The log-domain formulation avoids numerical
underflow that arises when epsilon is small.
"""
import math
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
    Compute the Sinkhorn (entropic OT) distance between two sets of vectors.

    Args:
        x:        [T_x, D] — video frame representations for one sample.
        y:        [T_y, D] — audio frame representations for one sample.
        x_mask:   [T_x] bool/long, True/1 for valid (non-padding) frames.
        y_mask:   [T_y] bool/long, True/1 for valid (non-padding) frames.
        epsilon:  Entropy regularization strength. Smaller → closer to exact OT.
        max_iter: Number of Sinkhorn iterations.
        normalize: If True, project x and y onto the unit sphere before computing
                   the cosine-distance cost matrix.

    Returns:
        Scalar tensor — the Sinkhorn loss (differentiable w.r.t. x and y).
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

    # Cosine distance ∈ [0, 2] for unit vectors.
    C = 1.0 - (x @ y.T)  # [T_x, T_y]

    log_a = torch.full((T_x,), -math.log(T_x), device=x.device, dtype=x.dtype)
    log_b = torch.full((T_y,), -math.log(T_y), device=x.device, dtype=x.dtype)

    # Stop-gradient trick (envelope theorem): run Sinkhorn iterations without
    # tracking gradients, then backprop only through C in the final product.
    # Gradient of (P * C).sum() w.r.t. C_ij = P_ij, which equals the exact
    # OT gradient at convergence — avoids backpropping through max_iter logsumexp layers.
    with torch.no_grad():
        log_K = -C / epsilon  # [T_x, T_y]
        log_u = torch.zeros(T_x, device=x.device, dtype=x.dtype)
        log_v = torch.zeros(T_y, device=x.device, dtype=x.dtype)
        for _ in range(max_iter):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        transport = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))  # [T_x, T_y]

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
    Compute Sinkhorn loss over a batch, averaging across samples.

    Args:
        x:      [B, T_x, D] — video representations.
        y:      [B, T_y, D] — audio representations.
        x_mask: [B, T_x] — video padding mask (1 = valid).
        y_mask: [B, T_y] — audio padding mask (1 = valid).
        epsilon: Entropy regularization strength.
        max_iter: Number of Sinkhorn iterations.

    Returns:
        Scalar tensor — mean Sinkhorn loss across the batch.
    """
    B = x.shape[0]
    losses = []
    for i in range(B):
        xm = x_mask[i] if x_mask is not None else None
        ym = y_mask[i] if y_mask is not None else None
        loss_i = sinkhorn_loss(x[i], y[i], xm, ym, epsilon=epsilon, max_iter=max_iter)
        losses.append(loss_i)
    return torch.stack(losses).mean()
