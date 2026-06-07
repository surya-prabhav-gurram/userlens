"""
src/models/bert4rec.py
BERT4Rec: Transformer-based masked item prediction model.
Architecture: N Transformer encoder layers, learnable item + positional embeddings.
Used for self-supervised pre-training on interaction sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    """
    BERT4Rec model for masked item prediction.
    Reference: Sun et al. (2019) "BERT4Rec: Sequential Recommendation
    with Bidirectional Encoder Representations from Transformer"

    Vocab layout: 0=PAD, 1=MASK, 2=CLS, 3..n_items+2=real items
    """

    def __init__(
        self,
        n_items: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.2,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.n_items  = n_items
        self.d_model  = d_model
        self.vocab_size = n_items + 3  # PAD=0, MASK=1, CLS=2, items=3..

        # Embeddings
        self.item_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Output projection head: hidden → vocab logits
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        # Weight tying: share item embedding weights with output projection
        self.head.weight = self.item_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)

    def encode(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of item sequences.
        Args:
            input_ids : (B, L) — item ids (may contain MASK_ID)
            attn_mask : (B, L) bool — True = real token, False = padding
        Returns:
            hidden    : (B, L, D)
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_norm(x)
        x = self.emb_drop(x)

        # TransformerEncoder expects src_key_padding_mask:
        #   True  → IGNORE this position (padding)
        #   False → attend to this position
        padding_mask = ~attn_mask  # invert: False=real, True=pad

        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        return hidden

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked item prediction.
        Returns:
            logits : (B, L, vocab_size)
        """
        hidden = self.encode(input_ids, attn_mask)
        logits = self.head(hidden)
        return logits

    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Produce a single vector per sequence (for retrieval / ranking).
        pool: 'mean' (average over real tokens) or 'last' (last real token)
        """
        hidden = self.encode(input_ids, attn_mask)  # (B, L, D)
        if pool == "mean":
            mask_f = attn_mask.float().unsqueeze(-1)  # (B, L, 1)
            emb = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:  # last real token
            lengths = attn_mask.long().sum(dim=1) - 1  # (B,)
            emb = hidden[torch.arange(hidden.size(0)), lengths]
        return emb  # (B, D)

    @property
    def item_embeddings(self) -> torch.Tensor:
        """Return the item embedding matrix (vocab_size, D)."""
        return self.item_emb.weight


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss only at masked positions (label != PAD_ID=0).
    Args:
        logits : (B, L, vocab_size)
        labels : (B, L) — 0 means 'not masked, ignore'
    """
    B, L, V = logits.shape
    mask = labels != 0  # True where masked
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    logits_flat = logits.view(B * L, V)
    labels_flat = labels.view(B * L)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0, reduction="mean")
    return loss
