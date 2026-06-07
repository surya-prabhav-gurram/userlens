"""
src/models/losses.py
Shared loss functions referenced across training stages.
"""
import torch
import torch.nn.functional as F


def bpr_loss(score_pos: torch.Tensor, score_neg: torch.Tensor,
             ips_weights: torch.Tensor = None) -> torch.Tensor:
    """Bayesian Personalized Ranking loss with optional IPS weighting."""
    loss = -F.logsigmoid(score_pos - score_neg)
    if ips_weights is not None:
        loss = ips_weights * loss
    return loss.mean()


def infonce_loss(user_emb: torch.Tensor, item_emb: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss for in-batch negatives (two-tower)."""
    sim = torch.matmul(user_emb, item_emb.T) / temperature
    B = user_emb.size(0)
    labels = torch.arange(B, device=user_emb.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only at masked positions (BERT4Rec pre-training)."""
    from src.models.bert4rec import masked_cross_entropy_loss
    return masked_cross_entropy_loss(logits, labels)
