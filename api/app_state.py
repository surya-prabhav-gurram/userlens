"""
api/app_state.py
Global application state — models, data, and pipeline loaded once at startup.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    # Loaded data
    train_seqs:   Dict[int, List[int]] = field(default_factory=dict)
    val_targets:  Dict[int, int]       = field(default_factory=dict)
    test_targets: Dict[int, int]       = field(default_factory=dict)
    item2id:      Dict[int, int]       = field(default_factory=dict)
    id2item:      Dict[int, int]       = field(default_factory=dict)
    n_items:      int                  = 0
    popularity:   Optional[np.ndarray] = None
    ips_weights:  Optional[np.ndarray] = None
    item_meta:    Optional[object]     = None  # DataFrame

    # Models
    two_tower:    Optional[object]     = None
    ranker:       Optional[object]     = None

    # Services
    pipeline:     Optional[object]     = None
    retriever:    Optional[object]     = None
    llm_reranker: Optional[object]     = None
    llm_judge:    Optional[object]     = None

    # Runtime tracking
    new_interactions: List[dict]       = field(default_factory=list)
    last_continual_run: Optional[str]  = None
    last_eval_result:   Optional[dict] = None
    last_bias_audit:    Optional[dict] = None
    model_version:      str            = "demo"

    def get_item_title(self, model_id: int) -> str:
        raw_id = self.id2item.get(model_id, model_id)
        if self.item_meta is not None and len(self.item_meta) > 0:
            try:
                rows = self.item_meta[self.item_meta["item_id"] == raw_id]
                if len(rows) > 0:
                    return str(rows.iloc[0].get("title", f"Item {raw_id}"))
            except Exception:
                pass
        return f"Movie {raw_id}"

    def get_user_sequence(self, user_id: int) -> List[int]:
        return self.train_seqs.get(user_id, [])

    def add_interaction(self, user_id: int, item_id: int, interaction_type: str = "click"):
        import time
        self.new_interactions.append({
            "user_id": user_id,
            "item_id": item_id,
            "type": interaction_type,
            "timestamp": int(time.time()),
        })
        # Update in-memory sequence
        if user_id not in self.train_seqs:
            self.train_seqs[user_id] = []
        self.train_seqs[user_id].append(item_id)


# Global singleton
state = AppState()


def load_state(data_dir: str = "data/processed", device: str = "cpu") -> AppState:
    """Load data + models into global state. Called at FastAPI startup."""
    global state
    from src.data.preprocessing import load_processed

    # ── Data ──────────────────────────────────────────────────────────────
    try:
        data = load_processed(data_dir)
        state.train_seqs   = data["train_seqs"]
        state.val_targets  = data["val_targets"]
        state.test_targets = data["test_targets"]
        state.item2id      = data["item2id"]
        state.id2item      = data["id2item"]
        state.n_items      = data["n_items"]
        state.popularity   = data["popularity"]
        state.ips_weights  = data["ips_weights"]
        state.item_meta    = data.get("item_meta")
        logger.info(f"Loaded data: {state.n_items} items, {len(state.train_seqs)} users")
    except Exception as e:
        logger.warning(f"Could not load processed data: {e} — running in demo mode")
        # Create minimal demo state
        _init_demo_state()
        _init_services(device)
        return state

    # ── Models ─────────────────────────────────────────────────────────────
    _load_models(device)

    # ── Services ───────────────────────────────────────────────────────────
    _init_services(device)

    return state


def _init_demo_state():
    """Populate state with tiny synthetic data for demo/testing."""
    import random
    state.n_items = 1000
    state.id2item = {i: i for i in range(3, 1003)}
    state.item2id = {i: i for i in range(3, 1003)}
    state.popularity = np.random.exponential(5, 1003).astype(np.float32)
    state.ips_weights = np.ones(1003, dtype=np.float32)

    for uid in range(1, 201):
        seq_len = random.randint(5, 30)
        state.train_seqs[uid]  = random.sample(range(3, 1000), seq_len)
        state.val_targets[uid] = random.randint(3, 1000)
        state.test_targets[uid]= random.randint(3, 1000)

    logger.info("Demo state initialized with 1000 items, 200 users")


def _load_models(device: str):
    """Try to load trained model checkpoints."""
    pretrain_ckpt  = os.getenv("PRETRAIN_CKPT", "experiments/pretrain/best_bert4rec.pt")
    retrieval_ckpt = os.getenv("RETRIEVAL_CKPT", "experiments/retrieval/best_two_tower.pt")
    ranking_ckpt   = os.getenv("RANKING_CKPT",   "experiments/ranking/best_ranker.pt")

    backbone = None

    if os.path.exists(pretrain_ckpt):
        try:
            from src.training.pretrain import load_pretrained
            backbone = load_pretrained(pretrain_ckpt, device)
            logger.info(f"Loaded BERT4Rec backbone from {pretrain_ckpt}")
        except Exception as e:
            logger.warning(f"Could not load backbone: {e}")

    if backbone and os.path.exists(retrieval_ckpt):
        try:
            from src.training.train_retrieval import load_two_tower
            state.two_tower = load_two_tower(retrieval_ckpt, backbone, device)
            state.model_version = "two_tower"
            logger.info("Loaded two-tower model")
        except Exception as e:
            logger.warning(f"Could not load two-tower: {e}")

    if backbone and os.path.exists(ranking_ckpt):
        try:
            from src.training.train_ranking import load_ranker
            state.ranker = load_ranker(ranking_ckpt, backbone, device)
            state.model_version = "ranker"
            logger.info("Loaded ranker model")
        except Exception as e:
            logger.warning(f"Could not load ranker: {e}")


def _init_services(device: str):
    """Initialize retriever, reranker, and pipeline."""
    from src.inference.retriever import PgVectorRetriever
    from src.inference.reranker import LLMReranker
    from src.inference.pipeline import RecommendationPipeline
    from src.evaluation.llm_judge import LLMJudge

    from src.models.cold_start import ColdStartRouter, ContentBasedRetriever
    state.retriever    = PgVectorRetriever()
    state.llm_reranker = LLMReranker()
    state.llm_judge    = LLMJudge()

    cold_start_router = ColdStartRouter(content_retriever=ContentBasedRetriever(
        np.random.randn(max(state.n_items, 1), 384).astype(np.float32),
        list(state.id2item.keys())[:max(state.n_items, 1)],
    ))

    state.pipeline = RecommendationPipeline(
        two_tower_model=state.two_tower,
        ranker_model=state.ranker,
        retriever=state.retriever,
        llm_reranker=state.llm_reranker,
        cold_start_router=cold_start_router,
        id2item=state.id2item,
        item_meta=state.item_meta,
        device=device,
    )

    # Generate item embeddings from two-tower or fall back to random
    if state.two_tower is not None:
        import torch
        logger.info("Generating item embeddings from two-tower model...")
        all_ids = list(state.id2item.keys())
        batch_size = 512
        all_embs = []
        state.two_tower.eval()
        with torch.no_grad():
            for i in range(0, len(all_ids), batch_size):
                batch = torch.tensor(all_ids[i:i+batch_size], dtype=torch.long)
                emb = state.two_tower.item_tower(batch, None)
                all_embs.append(emb.cpu().numpy())
        emb_matrix = np.vstack(all_embs).astype(np.float32)
        state.retriever.load_embeddings_to_memory(emb_matrix, np.array(all_ids))
        logger.info(f"Loaded {len(all_ids)} item embeddings from two-tower")
    else:
        logger.info("No two-tower model — populating retriever with random embeddings for demo")
        n = state.n_items + 3
        emb_dim = 128
        embs = np.random.randn(n, emb_dim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= np.maximum(norms, 1e-8)
        item_ids = list(range(3, n))
        state.retriever.load_embeddings_to_memory(embs[3:], np.array(item_ids))

    logger.info("Services initialized")
