"""
src/evaluation/bias_audit.py
Popularity bias, fairness metrics, and diversity audit.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def popularity_bias_audit(
    all_recommended: Dict[int, List[int]],
    popularity: np.ndarray,
    ks: List[int] = (10, 20),
) -> dict:
    """Measure how concentrated recommendations are on popular items."""
    results = {}
    n_items = len(popularity)

    for k in ks:
        recs = [r[:k] for r in all_recommended.values()]
        flat = [item for rec in recs for item in rec]
        if not flat:
            results[f"pop_conc@{k}"] = 0.0
            continue

        pop_ranks = np.argsort(popularity)[::-1]  # most popular first
        top10pct = set(pop_ranks[:max(1, n_items // 10)].tolist())
        top20pct = set(pop_ranks[:max(1, n_items // 5)].tolist())

        results[f"pop_conc_top10pct@{k}"] = sum(1 for i in flat if i in top10pct) / len(flat)
        results[f"pop_conc_top20pct@{k}"] = sum(1 for i in flat if i in top20pct) / len(flat)

        rec_popularities = [float(popularity[i]) for i in flat if i < len(popularity)]
        results[f"avg_pop@{k}"] = float(np.mean(rec_popularities)) if rec_popularities else 0.0

    return results


def genre_coverage_audit(
    all_recommended: Dict[int, List[int]],
    item_genres: Dict[int, str],
    k: int = 10,
) -> dict:
    """Measure genre diversity in recommendation lists."""
    genre_counts: Dict[str, int] = {}
    total = 0

    for uid, recs in all_recommended.items():
        for item_id in recs[:k]:
            genre = item_genres.get(item_id, "Unknown")
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            total += 1

    if total == 0:
        return {"n_genres": 0, "genre_entropy": 0.0}

    probs = np.array(list(genre_counts.values())) / total
    entropy = -np.sum(probs * np.log2(np.maximum(probs, 1e-10)))

    return {
        "n_genres":      len(genre_counts),
        "genre_entropy": float(entropy),
        "genre_dist":    {g: c / total for g, c in sorted(genre_counts.items(), key=lambda x: -x[1])[:10]},
    }


def compute_full_bias_audit(
    pipeline,
    test_targets: Dict[int, int],
    train_seqs:   Dict[int, List[int]],
    popularity:   np.ndarray,
    item_genres:  Optional[Dict[int, str]] = None,
    n_users:      int = 200,
) -> dict:
    """Run complete bias audit. Returns combined metrics dict."""
    import random
    users = random.sample(list(test_targets.keys()), min(n_users, len(test_targets)))
    all_recommended: Dict[int, List[int]] = {}

    for uid in users:
        seq    = train_seqs.get(uid, [])
        result = pipeline.recommend(seq, top_k=20, mode="neural")
        all_recommended[uid] = [item["item_id"] for item in result["items"]]

    audit = popularity_bias_audit(all_recommended, popularity)
    if item_genres:
        audit.update(genre_coverage_audit(all_recommended, item_genres))

    return audit
