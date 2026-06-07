"""
src/inference/pipeline.py
End-to-end inference pipeline: retrieve → rank → (optionally) LLM re-rank.
Routes cold/warm users to appropriate pathways.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from src.inference.retriever import PgVectorRetriever
from src.inference.reranker import LLMReranker
from src.models.cold_start import ColdStartRouter, ContentBasedRetriever

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """
    Full recommendation pipeline.
    - Warm users:  two-tower retrieval → ranker → optional LLM re-rank
    - Cold users:  content-based retrieval → optional LLM re-rank
    """

    def __init__(
        self,
        two_tower_model=None,
        ranker_model=None,
        retriever: Optional[PgVectorRetriever] = None,
        llm_reranker: Optional[LLMReranker] = None,
        cold_start_router: Optional[ColdStartRouter] = None,
        id2item: Optional[Dict] = None,
        item_meta: Optional[object] = None,   # DataFrame
        device: str = "cpu",
    ):
        self.two_tower  = two_tower_model
        self.ranker     = ranker_model
        self.retriever  = retriever or PgVectorRetriever()
        self.llm        = llm_reranker
        self.cold_router = cold_start_router
        self.id2item    = id2item or {}
        self.item_meta  = item_meta
        self.device     = device

    def _get_item_title(self, model_id: int) -> str:
        raw_id = self.id2item.get(model_id, model_id)
        if self.item_meta is not None and len(self.item_meta) > 0:
            rows = self.item_meta[self.item_meta["item_id"] == raw_id]
            if len(rows) > 0:
                return str(rows.iloc[0].get("title", f"Item {raw_id}"))
        return f"Item {raw_id}"

    @torch.no_grad()
    def recommend(
        self,
        user_sequence: List[int],
        top_k: int = 10,
        retrieval_k: int = 500,
        mode: str = "neural",   # "neural" | "llm" | "hybrid" | "cold"
        exclude_seen: bool = True,
    ) -> dict:
        """
        Generate recommendations for a user.

        Args:
            user_sequence: list of model item IDs (interaction history)
            top_k:         number of final recommendations
            retrieval_k:   candidates to retrieve before ranking
            mode:          "neural" skips LLM; "llm" applies LLM re-rank; "hybrid" blends
            exclude_seen:  filter items already in user history

        Returns dict with keys: items, reasoning, pathway, retrieval_count
        """
        import numpy as np
        n_interactions = len(user_sequence)
        pathway = "warm_neural"
        exclude_ids = user_sequence if exclude_seen else None

        # ── Cold-start routing ──────────────────────────────────────────
        if self.cold_router and self.cold_router.is_cold_user(n_interactions):
            pathway = "cold_content"
            if self.cold_router.content_retriever:
                # Use a random query for cold users with no history
                query_emb = np.random.randn(384).astype(np.float32)
                query_emb /= max(np.linalg.norm(query_emb), 1e-8)
                raw_candidates = self.cold_router.content_retriever.retrieve(
                    query_emb, k=retrieval_k, exclude_ids=exclude_ids
                )
                candidates = [{"item_id": iid, "score": float(s), "title": self._get_item_title(iid)}
                               for iid, s in raw_candidates]
            else:
                candidates = []
        else:
            # ── Warm path: two-tower retrieval ────────────────────────
            if self.two_tower and len(user_sequence) > 0:
                from src.data.dataset import PAD_ID
                max_len = 200
                seq = user_sequence[-max_len:]
                pad_len = max_len - len(seq)
                input_ids = torch.tensor([PAD_ID] * pad_len + list(seq), dtype=torch.long).unsqueeze(0).to(self.device)
                attn_mask = torch.tensor([False] * pad_len + [True] * len(seq), dtype=torch.bool).unsqueeze(0).to(self.device)

                user_emb = self.two_tower.encode_user(input_ids, attn_mask)
                query_np = user_emb.cpu().numpy()[0]
                raw_candidates = self.retriever.retrieve(query_np, k=retrieval_k, exclude_ids=exclude_ids)
                candidates = [{"item_id": iid, "score": float(s), "title": self._get_item_title(iid)}
                               for iid, s in raw_candidates]
            else:
                # No two-tower model — fall back to retriever directly with a random query
                import numpy as np
                query_np = np.random.randn(128).astype(np.float32)
                query_np /= max(np.linalg.norm(query_np), 1e-8)
                raw_candidates = self.retriever.retrieve(query_np, k=retrieval_k, exclude_ids=exclude_ids)
                candidates = [{"item_id": iid, "score": float(s), "title": self._get_item_title(iid)}
                               for iid, s in raw_candidates]

        # ── Ranker: re-score candidates ───────────────────────────────
        if self.ranker and candidates and pathway == "warm_neural":
            candidates = self._rank_candidates(user_sequence, candidates, top_n=min(retrieval_k, len(candidates)))
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── LLM re-rank ───────────────────────────────────────────────
        reasoning = "Neural ranking"
        if mode in ("llm", "hybrid") and self.llm:
            history_titles = [self._get_item_title(i) for i in user_sequence[-10:]]
            candidates, reasoning = self.llm.rerank(history_titles, candidates[:20], top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return {
            "items":           candidates,
            "reasoning":       reasoning,
            "pathway":         pathway,
            "retrieval_count": len(candidates),
        }

    @torch.no_grad()
    def _rank_candidates(
        self,
        user_sequence: List[int],
        candidates: List[dict],
        top_n: int = 200,
    ) -> List[dict]:
        """Score top_n candidates with the cross-attention ranker."""
        from src.data.dataset import PAD_ID
        max_len = 200
        seq = user_sequence[-max_len:]
        pad_len = max_len - len(seq)
        input_ids = torch.tensor([PAD_ID] * pad_len + list(seq), dtype=torch.long).unsqueeze(0).to(self.device)
        attn_mask = torch.tensor([False] * pad_len + [True] * len(seq), dtype=torch.bool).unsqueeze(0).to(self.device)

        user_hidden = self.ranker.encode_user_seq(input_ids, attn_mask)

        batch_size = 64
        top_cands = candidates[:top_n]
        scores = []
        for i in range(0, len(top_cands), batch_size):
            chunk = top_cands[i:i+batch_size]
            item_ids = torch.tensor([c["item_id"] for c in chunk], dtype=torch.long).to(self.device)
            user_h_exp = user_hidden.expand(len(chunk), -1, -1)
            mask_exp   = attn_mask.expand(len(chunk), -1)
            s = self.ranker.score(user_h_exp, mask_exp, item_ids)
            scores.extend(s.cpu().tolist())

        for cand, score in zip(top_cands, scores):
            cand["score"] = float(score)

        return top_cands
