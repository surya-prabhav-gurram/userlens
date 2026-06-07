"""
src/evaluation/metrics.py
Offline evaluation metrics: NDCG@K, HitRate@K, MRR, Coverage, Diversity.
"""

import math
from typing import Dict, List, Optional

import numpy as np


def ndcg_at_k(recommended: List[int], relevant: int, k: int) -> float:
    """NDCG@K for a single user (one relevant item — leave-one-out protocol)."""
    recommended = recommended[:k]
    if relevant in recommended:
        rank = recommended.index(relevant) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0


def hit_rate_at_k(recommended: List[int], relevant: int, k: int) -> float:
    return 1.0 if relevant in recommended[:k] else 0.0


def mrr(recommended: List[int], relevant: int) -> float:
    if relevant in recommended:
        rank = recommended.index(relevant) + 1
        return 1.0 / rank
    return 0.0


def catalog_coverage(all_recommended: List[List[int]], n_items: int) -> float:
    """Fraction of the item catalog that appears in at least one recommendation list."""
    seen = set()
    for rec_list in all_recommended:
        seen.update(rec_list)
    return len(seen) / max(n_items, 1)


def intra_list_diversity(
    rec_list: List[int],
    item_embeddings: np.ndarray,
) -> float:
    """Average pairwise cosine distance within a recommendation list."""
    if len(rec_list) < 2:
        return 0.0
    embs = item_embeddings[rec_list]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)
    sim_matrix = embs @ embs.T
    n = len(rec_list)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += 1.0 - sim_matrix[i, j]
            count += 1
    return total_dist / max(count, 1)


def popularity_concentration(
    all_recommended: List[List[int]],
    popularity: np.ndarray,
    top_pct: float = 0.1,
) -> float:
    """Fraction of recommended items that are in the top_pct% most popular items."""
    n_items = len(popularity)
    n_top = max(1, int(n_items * top_pct))
    top_items = set(np.argsort(popularity)[-n_top:].tolist())
    all_recs = [item for rec in all_recommended for item in rec]
    if not all_recs:
        return 0.0
    return sum(1 for i in all_recs if i in top_items) / len(all_recs)


def evaluate_pipeline(
    pipeline,
    test_targets: Dict[int, int],
    train_seqs: Dict[int, List[int]],
    ks: List[int] = (10, 20, 50),
    n_users: Optional[int] = None,
    popularity: Optional[np.ndarray] = None,
) -> dict:
    """
    Run full offline evaluation.
    Returns dict with NDCG@K, HitRate@K, MRR, Coverage, PopConc.
    """
    results_per_k: Dict[int, Dict[str, List[float]]] = {
        k: {"ndcg": [], "hit": [], "mrr": []} for k in ks
    }
    all_recommended: Dict[int, List[int]] = {}

    users = list(test_targets.keys())
    if n_users:
        users = users[:n_users]

    for uid in users:
        target = test_targets[uid]
        seq    = train_seqs.get(uid, [])
        result = pipeline.recommend(seq, top_k=max(ks), retrieval_k=500, mode="neural")
        rec_ids = [item["item_id"] for item in result["items"]]
        all_recommended[uid] = rec_ids

        for k in ks:
            results_per_k[k]["ndcg"].append(ndcg_at_k(rec_ids, target, k))
            results_per_k[k]["hit"].append(hit_rate_at_k(rec_ids, target, k))
            results_per_k[k]["mrr"].append(mrr(rec_ids, target))

    n_items = max(max(r) for r in all_recommended.values() if r) if all_recommended else 1
    coverage = catalog_coverage(list(all_recommended.values()), n_items)

    output = {"coverage": coverage, "n_users_evaluated": len(users)}
    for k in ks:
        output[f"ndcg@{k}"]     = float(np.mean(results_per_k[k]["ndcg"]))
        output[f"hitrate@{k}"]  = float(np.mean(results_per_k[k]["hit"]))
        output[f"mrr@{k}"]      = float(np.mean(results_per_k[k]["mrr"]))

    if popularity is not None:
        output["pop_concentration"] = popularity_concentration(
            list(all_recommended.values()), popularity
        )

    return output
