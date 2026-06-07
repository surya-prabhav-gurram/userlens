"""
src/models/ranker.py
Sequence-aware cross-attention ranking model.
Architecture: user sequence → Transformer → cross-attention with item → MLP → scalar score
Trained with IPS-weighted BPR pairwise loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bert4rec import BERT4Rec


class CrossAttentionRanker(nn.Module):
    """
    Ranks candidate items against a user's full interaction history
    using cross-attention: item embedding queries the user sequence states.

    This is analogous to Meta's production ranking models where a
    feature interaction layer considers the full user context.
    """

    def __init__(
        self,
        backbone: BERT4Rec,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        d_model = backbone.d_model
        self.backbone = backbone
        self.d_model  = d_model

        # Cross-attention: item queries user sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Item embedding projection (reuse backbone item_emb)
        self.item_emb = backbone.item_emb

        # Scoring MLP: [user_pool; item_emb; user_pool * item_emb] → scalar
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_user_seq(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns full sequence of hidden states (B, L, D)."""
        return self.backbone.encode(input_ids, attn_mask)

    def score(
        self,
        user_hidden: torch.Tensor,   # (B, L, D) — pre-computed user hidden states
        attn_mask: torch.Tensor,     # (B, L)
        item_ids: torch.Tensor,      # (B,) or (B, K) for batch scoring K candidates
    ) -> torch.Tensor:
        """
        Score item_ids against user_hidden.
        Supports two modes:
          - (B,)    → score one item per user → returns (B,)
          - (B, K)  → score K items per user  → returns (B, K)
        """
        single = item_ids.dim() == 1
        if single:
            item_ids = item_ids.unsqueeze(1)  # (B, 1)

        B, K = item_ids.shape
        L    = user_hidden.size(1)

        item_e = self.item_emb(item_ids)  # (B, K, D)

        # Cross-attention: each item queries the user sequence
        # Flatten to (B*K, 1, D) queries, (B*K, L, D) keys/values
        item_q = item_e.view(B * K, 1, self.d_model)
        user_k = user_hidden.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, L, self.d_model)

        # Padding mask: (B*K, L) — True = ignore
        pad_mask = (~attn_mask).unsqueeze(1).expand(-1, K, -1).reshape(B * K, L)

        ctx, _ = self.cross_attn(item_q, user_k, user_k, key_padding_mask=pad_mask)
        ctx = self.cross_norm(ctx.squeeze(1))  # (B*K, D)

        # Mean-pool user hidden for global context
        mask_f = attn_mask.float().unsqueeze(-1)  # (B, L, 1)
        user_pool = (user_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
        user_pool = user_pool.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.d_model)

        item_flat = item_e.view(B * K, self.d_model)

        # Feature interaction: concat [user_pool, item_emb, elementwise_product]
        feat = torch.cat([user_pool, item_flat, user_pool * item_flat], dim=-1)
        scores = self.scorer(feat).squeeze(-1)  # (B*K,)
        scores = scores.view(B, K)

        if single:
            scores = scores.squeeze(1)  # (B,)
        return scores

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        ips_weights: torch.Tensor,
    ) -> dict:
        """
        BPR pairwise forward pass with IPS debiasing.
        loss = mean(ips_weight * -log(sigmoid(score_pos - score_neg)))
        """
        user_hidden = self.encode_user_seq(input_ids, attn_mask)
        score_pos = self.score(user_hidden, attn_mask, pos_item_ids)  # (B,)
        score_neg = self.score(user_hidden, attn_mask, neg_item_ids)  # (B,)

        # BPR loss with IPS weighting
        bpr = -F.logsigmoid(score_pos - score_neg)          # (B,)
        loss = (ips_weights * bpr).mean()

        return {
            "loss":      loss,
            "score_pos": score_pos.detach(),
            "score_neg": score_neg.detach(),
        }
