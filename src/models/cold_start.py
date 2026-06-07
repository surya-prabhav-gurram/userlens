"""
src/models/cold_start.py
Cold-start handling via content-based retrieval.
Uses sentence-transformer text embeddings of item metadata
to serve new users/items with no interaction history.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

COLD_USER_THRESHOLD = 3   # users with fewer than this many interactions
COLD_ITEM_THRESHOLD = 20  # items with fewer than this many interactions


class ContentBasedRetriever:
    """
    Text-embedding-based retrieval for cold-start scenarios.
    Items are represented by sentence-transformer embeddings of their metadata.
    No neural training required — purely embedding similarity search.
    """

    def __init__(self, item_text_embeddings: np.ndarray, item_ids: List[int]):
        """
        Args:
            item_text_embeddings : (N, D) array of text embeddings
            item_ids             : list of model item IDs (length N)
        """
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(item_text_embeddings, axis=1, keepdims=True)
        self.embeddings = item_text_embeddings / np.maximum(norms, 1e-8)
        self.item_ids   = np.array(item_ids)

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: int = 50,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k items by cosine similarity to query_emb.
        Returns list of (item_id, score).
        """
        query_emb = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
        scores = self.embeddings @ query_emb  # (N,)

        if exclude_ids:
            exclude_set = set(exclude_ids)
            for i, iid in enumerate(self.item_ids):
                if iid in exclude_set:
                    scores[i] = -1.0

        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top_idx]


class ColdStartRouter:
    """
    Routes inference requests to the appropriate retrieval pathway:
      - Cold user (< 3 interactions)  → content-based retrieval
      - Warm user (>= 3 interactions) → neural two-tower + ranker
      - Blend mode                    → weighted combination
    """

    def __init__(
        self,
        content_retriever: ContentBasedRetriever,
        cold_user_threshold: int = COLD_USER_THRESHOLD,
        cold_item_threshold: int = COLD_ITEM_THRESHOLD,
    ):
        self.content_retriever    = content_retriever
        self.cold_user_threshold  = cold_user_threshold
        self.cold_item_threshold  = cold_item_threshold

    def is_cold_user(self, n_interactions: int) -> bool:
        return n_interactions < self.cold_user_threshold

    def route(self, n_interactions: int) -> str:
        if n_interactions == 0:
            return "cold_content"
        elif n_interactions < self.cold_user_threshold:
            return "cold_blend"
        else:
            return "warm_neural"


class DomainAdapter(nn.Module):
    """
    Few-shot domain adaptation module.
    Freezes the lower Transformer layers and fine-tunes only:
      - Top N layers of the backbone
      - Item embedding layer
      - A domain-specific projection head
    """

    def __init__(self, backbone, n_finetune_layers: int = 2):
        super().__init__()
        self.backbone = backbone

        # Freeze all parameters first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze top n_finetune_layers Transformer layers
        total_layers = len(backbone.transformer.layers)
        for i in range(total_layers - n_finetune_layers, total_layers):
            for param in backbone.transformer.layers[i].parameters():
                param.requires_grad = True

        # Unfreeze item embedding (domain-specific vocabulary)
        for param in backbone.item_emb.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(f"DomainAdapter: {trainable:,} / {total:,} params trainable")

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)


def build_item_text_matrix(
    item_meta_df,
    item2id: Dict[int, int],
    text_col: str = "title",
    embed_dim: int = 384,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a text embedding matrix for all items with metadata.
    Falls back to random embeddings if sentence-transformers not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        use_model = True
        logger.info("Using sentence-transformers for content embeddings")
    except ImportError:
        use_model = False
        logger.warning("sentence-transformers not available — using random content embeddings")

    items_with_meta = item_meta_df[item_meta_df["item_id"].isin(item2id)]
    texts    = items_with_meta[text_col].fillna("").tolist()
    raw_ids  = items_with_meta["item_id"].tolist()
    model_ids = [item2id[r] for r in raw_ids]

    if use_model and texts:
        embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    else:
        embeddings = np.random.randn(len(texts), embed_dim).astype(np.float32)
        if embeddings.shape[0] > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)

    return embeddings.astype(np.float32), model_ids
