"""Loss functions: Huber(delta=1.0) + 0.25 * pairwise margin ranking loss."""

import torch
import torch.nn.functional as F


def pairwise_margin_ranking_loss(scores: torch.Tensor, targets: torch.Tensor,
                                 margin: float = 0.1, n_pair_mult: int = 4,
                                 generator=None) -> torch.Tensor:
    """Mean over sampled ordered pairs of max(0, -sign(y_i - y_j)*(s_i - s_j) + margin).

    Pairs with tied targets are excluded (sign = 0 carries no ranking info).
    """
    B = scores.numel()
    if B < 2:
        return scores.new_zeros(())
    n_pairs = min(n_pair_mult * B, B * (B - 1))
    i = torch.randint(0, B, (n_pairs,), generator=generator, device=scores.device)
    j = torch.randint(0, B, (n_pairs,), generator=generator, device=scores.device)
    keep = i != j
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return scores.new_zeros(())
    dy = targets[i] - targets[j]
    ds = scores[i] - scores[j]
    tied = dy == 0
    if bool(tied.all()):
        return scores.new_zeros(())
    dy, ds = dy[~tied], ds[~tied]
    return torch.clamp(-torch.sign(dy) * ds + margin, min=0.0).mean()


def combined_loss(scores: torch.Tensor, targets: torch.Tensor,
                  huber_delta: float = 1.0, rank_weight: float = 0.25,
                  margin: float = 0.1, generator=None):
    """Huber(delta=1.0) + 0.25 * pairwise margin ranking loss.

    Returns (total_loss, {'huber': ..., 'rank': ...}).
    """
    huber = F.huber_loss(scores, targets, reduction='mean', delta=huber_delta)
    rank = pairwise_margin_ranking_loss(scores, targets, margin=margin,
                                        generator=generator)
    total = huber + rank_weight * rank
    return total, {'huber': huber.detach(), 'rank': rank.detach()}
