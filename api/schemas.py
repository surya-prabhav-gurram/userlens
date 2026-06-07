"""
api/schemas.py
Pydantic request/response models for the UserLens API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    sequence: Optional[List[int]] = None  # direct sequence if no user_id
    k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="neural", pattern="^(neural|llm|hybrid)$")
    exclude_seen: bool = True


class RecommendedItem(BaseModel):
    item_id: int
    title: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    user_id: Optional[int]
    items: List[RecommendedItem]
    reasoning: str
    pathway: str
    retrieval_count: int
    mode: str


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    interaction_type: str = Field(default="click", pattern="^(click|like|purchase|skip)$")
    timestamp: Optional[int] = None


class EvalResult(BaseModel):
    timestamp: str
    n_users_evaluated: int
    ndcg_at_10: float
    hitrate_at_10: float
    mrr_at_10: float
    coverage: float
    pop_concentration: Optional[float] = None


class LLMJudgeResult(BaseModel):
    n_evaluated: int
    relevance: float
    diversity: float
    novelty: float
    serendipity: float
    sample_reasonings: List[str]


class BiasAuditResult(BaseModel):
    pop_conc_top10pct_at10: Optional[float] = None
    pop_conc_top20pct_at10: Optional[float] = None
    avg_pop_at10: Optional[float] = None
    n_genres: Optional[int] = None
    genre_entropy: Optional[float] = None
    genre_dist: Optional[Dict[str, float]] = None


class AugmentRequest(BaseModel):
    category: Optional[str] = None
    min_interactions: int = 20
    max_items: int = 50


class ContinualStatus(BaseModel):
    last_run: Optional[str]
    n_new_interactions: int
    n_total_interactions: int
    last_loss: Optional[float]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    db_status: str
    n_items: int
    n_users: int
