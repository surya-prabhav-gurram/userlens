"""
src/inference/reranker.py
LLM-based final re-ranking using Claude API with chain-of-thought reasoning.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a recommendation system expert performing final re-ranking.

User's recent interaction history (item titles):
{history}

Candidate items to re-rank (in no particular order):
{candidates}

Re-rank these candidates from most to least relevant for this user.
Consider: thematic coherence with history, genre/style preferences shown,
diversity (avoid recommending 10 nearly identical items), and novelty.

Return ONLY valid JSON with no markdown or explanation:
{{"ranked_ids": [list of item_ids in ranked order], "reasoning": "brief explanation of ranking logic"}}"""


class LLMReranker:
    """
    Re-ranks top-N candidates using Claude with chain-of-thought reasoning.
    Falls back to neural ranking order if Claude is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model   = model
        self._client = None
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("LLMReranker: Claude client initialized")
            except ImportError:
                logger.warning("anthropic package not installed — LLM re-ranking disabled")

    def rerank(
        self,
        user_history_titles: List[str],
        candidates: List[Dict],   # [{"item_id": int, "title": str, "score": float}]
        top_k: int = 10,
    ) -> Tuple[List[Dict], str]:
        """
        Re-rank candidates. Returns (reranked_list, reasoning_text).
        Falls back gracefully to original order.
        """
        if not self._client or not candidates:
            return candidates[:top_k], "Neural ranking (LLM unavailable)"

        history_str   = "\n".join(f"- {t}" for t in user_history_titles[-10:])
        candidates_str = "\n".join(
            f"- item_id={c['item_id']}: {c.get('title', 'Unknown')} (neural_score={c['score']:.3f})"
            for c in candidates[:20]
        )
        prompt = RERANK_PROMPT.format(history=history_str, candidates=candidates_str)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            ranked_ids = [int(i) for i in data["ranked_ids"]]
            reasoning  = data.get("reasoning", "")

            id_to_cand = {c["item_id"]: c for c in candidates}
            reranked = []
            for rid in ranked_ids:
                if rid in id_to_cand:
                    reranked.append(id_to_cand[rid])
            # Append any candidates not mentioned by LLM
            seen = set(ranked_ids)
            for c in candidates:
                if c["item_id"] not in seen:
                    reranked.append(c)

            return reranked[:top_k], reasoning

        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {e} — using neural order")
            return candidates[:top_k], f"Neural fallback (LLM error: {str(e)[:80]})"
