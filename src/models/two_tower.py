"""
src/models/two_tower.py
Two-tower retrieval model.
User tower:  sequence → BERT4Rec backbone → mean pool → MLP → 128-dim embedding
Item tower:  item_id embedding + text embedding → MLP → 128-dim embedding
Similarity:  dot product (equivalent to cosine after L2-norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bert4rec import BERT4Rec


class UserTower(nn.Module):
    """
    Encodes a user's interaction history into a d_out-dimensional vector.
    Backbone: pre-trained BERT4Rec Transformer.
    """

    def __init__(self, backbone: BERT4Rec, d_out: int = 128):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(backbone.d_model, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids : (B, L)
            attn_mask : (B, L) bool
        Returns:
            user_emb  : (B, d_out) — L2-normalised
        """
        seq_emb = self.backbone.get_sequence_embedding(input_ids, attn_mask, pool="mean")
        proj    = self.proj(seq_emb)
        return F.normalize(proj, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes an item into a d_out-dimensional vector.
    Combines:
      - Learnable item_id embedding (from pre-trained BERT4Rec)
      - Optional frozen text embedding (sentence-transformer output)
    """

    def __init__(
        self,
        backbone: BERT4Rec,
        d_text: int = 384,    # sentence-transformer output dim
        d_out: int  = 128,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        # Reuse BERT4Rec item embedding layer
        self.item_emb = backbone.item_emb
        d_in = backbone.d_model + (d_text if use_text else 0)
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        text_embs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            item_ids  : (B,) or (N,)
            text_embs : (B, d_text) optional — frozen sentence-transformer embeddings
        Returns:
            item_emb  : (B, d_out) — L2-normalised
        """
        id_emb = self.item_emb(item_ids)  # (B, d_model)
        if self.use_text and text_embs is not None:
            combined = torch.cat([id_emb, text_embs], dim=-1)
        else:
            combined = id_emb
        proj = self.proj(combined)
        return F.normalize(proj, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model combining user and item towers.
    Inference: item embeddings are pre-computed; only the user tower runs at query time.
    """

    def __init__(
        self,
        backbone: BERT4Rec,
        d_out: int = 128,
        use_text: bool = False,  # set True when sentence-transformer available
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(backbone, d_out)
        self.item_tower = ItemTower(backbone, d_out=d_out, use_text=use_text)
        self.temperature = temperature

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_item_ids: torch.Tensor,
        text_embs: torch.Tensor = None,
    ) -> dict:
        """
        In-batch contrastive forward pass.
        Returns dict with 'loss' and 'user_emb', 'item_emb'.
        """
        user_emb = self.user_tower(input_ids, attn_mask)          # (B, D)
        item_emb = self.item_tower(pos_item_ids, text_embs)        # (B, D)

        # Similarity matrix: (B, B) dot product
        sim = torch.matmul(user_emb, item_emb.T) / self.temperature  # (B, B)

        # InfoNCE loss: diagonal = positive pairs
        B = user_emb.size(0)
        labels = torch.arange(B, device=user_emb.device)
        loss = (
            F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)
        ) / 2

        return {"loss": loss, "user_emb": user_emb, "item_emb": item_emb}

    def encode_user(self, input_ids, attn_mask) -> torch.Tensor:
        return self.user_tower(input_ids, attn_mask)

    def encode_item(self, item_ids, text_embs=None) -> torch.Tensor:
        return self.item_tower(item_ids, text_embs)
