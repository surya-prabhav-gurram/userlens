"""
src/data/preprocessing.py
Sequence construction, vocab building, train/val/test splitting.
Standard leave-one-out protocol used in RecSys research.
"""

import os
import json
import pickle
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Special token IDs
PAD_ID   = 0
MASK_ID  = 1
CLS_ID   = 2
FIRST_ITEM_ID = 3  # real items start at 3


def load_movielens(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load MovieLens-25M ratings.csv and return a clean DataFrame."""
    path = os.path.join(data_dir, "ml-25m", "ratings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MovieLens-25M not found at {path}. "
            "Download from https://grouplens.org/datasets/movielens/25m/ "
            "and extract to data/raw/ml-25m/"
        )
    logger.info("Loading MovieLens-25M ratings...")
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    logger.info(f"Loaded {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    return df


def load_movielens_movies(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load movies.csv for item metadata."""
    path = os.path.join(data_dir, "ml-25m", "movies.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["item_id", "title", "genres"])
    df = pd.read_csv(path, dtype={"movieId": int})
    df.columns = ["item_id", "title", "genres"]
    return df


def filter_interactions(df: pd.DataFrame,
                        min_user_interactions: int = 5,
                        min_item_interactions: int = 5) -> pd.DataFrame:
    """k-core filtering: keep only users and items with >= k interactions."""
    logger.info(f"Filtering: min_user={min_user_interactions}, min_item={min_item_interactions}")
    while True:
        before = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]
        after = len(df)
        if before == after:
            break
    logger.info(f"After filtering: {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    return df.reset_index(drop=True)


def build_vocab(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Map raw item IDs → consecutive integer IDs starting at FIRST_ITEM_ID.
    Returns:
        item2id: raw_id → model_id
        id2item: model_id → raw_id
    """
    unique_items = sorted(df["item_id"].unique())
    item2id = {raw: idx + FIRST_ITEM_ID for idx, raw in enumerate(unique_items)}
    id2item = {v: k for k, v in item2id.items()}
    logger.info(f"Vocabulary size: {len(item2id)} items (IDs {FIRST_ITEM_ID}..{max(item2id.values())})")
    return item2id, id2item


def build_sequences(df: pd.DataFrame,
                    item2id: Dict[int, int],
                    max_seq_len: int = 200) -> Dict[int, List[int]]:
    """
    Build per-user interaction sequences sorted by timestamp.
    Truncates to last max_seq_len items.
    """
    df = df.sort_values(["user_id", "timestamp"])
    sequences = {}
    for uid, group in df.groupby("user_id"):
        items = [item2id[i] for i in group["item_id"].tolist() if i in item2id]
        if items:
            sequences[int(uid)] = items[-max_seq_len:]
    logger.info(f"Built sequences for {len(sequences):,} users")
    return sequences


def leave_one_out_split(sequences: Dict[int, List[int]]) -> Tuple[
    Dict[int, List[int]],  # train
    Dict[int, int],         # val  (user → val item)
    Dict[int, int],         # test (user → test item)
]:
    """
    Standard RecSys evaluation protocol:
      - test  = last interaction in sequence
      - val   = second-to-last
      - train = everything before val
    Users with fewer than 3 interactions are dropped.
    """
    train, val, test = {}, {}, {}
    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue
        train[uid] = seq[:-2]
        val[uid]   = seq[-2]
        test[uid]  = seq[-1]
    logger.info(f"Split: {len(train):,} train users, {len(val):,} val, {len(test):,} test")
    return train, val, test


def compute_item_popularity(train_seqs: Dict[int, List[int]], n_items: int) -> np.ndarray:
    """Return array of interaction counts per item (index = model item id)."""
    counts = np.zeros(n_items + FIRST_ITEM_ID, dtype=np.float32)
    for seq in train_seqs.values():
        for item in seq:
            counts[item] += 1
    return counts


def compute_ips_weights(popularity: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """
    Inverse Propensity Scoring weights.
    propensity(i) = (count(i) / max_count) ^ alpha
    weight(i)     = 1 / propensity(i)   (clipped at 10 to avoid extremes)
    """
    max_count = popularity.max()
    propensity = np.where(popularity > 0, (popularity / max_count) ** alpha, 1.0)
    weights = np.where(propensity > 0, 1.0 / propensity, 1.0)
    weights = np.clip(weights, 1.0, 10.0)
    return weights.astype(np.float32)


def save_processed(
    train_seqs: Dict[int, List[int]],
    val_targets: Dict[int, int],
    test_targets: Dict[int, int],
    item2id: Dict[int, int],
    id2item: Dict[int, int],
    popularity: np.ndarray,
    ips_weights: np.ndarray,
    output_dir: str = "data/processed",
    movies_df: Optional[pd.DataFrame] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Save sequences as parquet
    rows = [{"user_id": uid, "items": items} for uid, items in train_seqs.items()]
    pa_table = pa.Table.from_pylist(rows)
    pq.write_table(pa_table, os.path.join(output_dir, "train_sequences.parquet"))

    rows_val  = [{"user_id": uid, "target": tgt} for uid, tgt in val_targets.items()]
    rows_test = [{"user_id": uid, "target": tgt} for uid, tgt in test_targets.items()]
    pq.write_table(pa.Table.from_pylist(rows_val),  os.path.join(output_dir, "val_targets.parquet"))
    pq.write_table(pa.Table.from_pylist(rows_test), os.path.join(output_dir, "test_targets.parquet"))

    # Save vocab
    with open(os.path.join(output_dir, "item2id.json"), "w") as f:
        json.dump({str(k): v for k, v in item2id.items()}, f)
    with open(os.path.join(output_dir, "id2item.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in id2item.items()}, f)

    # Save arrays
    np.save(os.path.join(output_dir, "popularity.npy"), popularity)
    np.save(os.path.join(output_dir, "ips_weights.npy"), ips_weights)

    # Save item metadata if available
    if movies_df is not None and len(movies_df) > 0:
        movies_df["model_id"] = movies_df["item_id"].map(item2id)
        movies_df = movies_df.dropna(subset=["model_id"])
        movies_df["model_id"] = movies_df["model_id"].astype(int)
        movies_df.to_parquet(os.path.join(output_dir, "item_metadata.parquet"), index=False)

    logger.info(f"Saved processed data to {output_dir}/")


def load_processed(data_dir: str = "data/processed") -> dict:
    """Load all processed data artifacts."""
    train_df = pd.read_parquet(os.path.join(data_dir, "train_sequences.parquet"))
    val_df   = pd.read_parquet(os.path.join(data_dir, "val_targets.parquet"))
    test_df  = pd.read_parquet(os.path.join(data_dir, "test_targets.parquet"))

    train_seqs   = {row.user_id: row.items for row in train_df.itertuples()}
    val_targets  = {row.user_id: row.target for row in val_df.itertuples()}
    test_targets = {row.user_id: row.target for row in test_df.itertuples()}

    with open(os.path.join(data_dir, "item2id.json")) as f:
        item2id = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(data_dir, "id2item.json")) as f:
        id2item = {int(k): v for k, v in json.load(f).items()}

    popularity  = np.load(os.path.join(data_dir, "popularity.npy"))
    ips_weights = np.load(os.path.join(data_dir, "ips_weights.npy"))

    meta_path = os.path.join(data_dir, "item_metadata.parquet")
    item_meta = pd.read_parquet(meta_path) if os.path.exists(meta_path) else pd.DataFrame()

    n_items = len(item2id)

    return dict(
        train_seqs=train_seqs, val_targets=val_targets, test_targets=test_targets,
        item2id=item2id, id2item=id2item, n_items=n_items,
        popularity=popularity, ips_weights=ips_weights, item_meta=item_meta,
    )


def run_pipeline(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    min_user: int = 5,
    min_item: int = 5,
    max_seq_len: int = 200,
):
    """End-to-end preprocessing pipeline."""
    logging.basicConfig(level=logging.INFO)
    df       = load_movielens(raw_dir)
    movies   = load_movielens_movies(raw_dir)
    df       = filter_interactions(df, min_user, min_item)
    item2id, id2item = build_vocab(df)
    seqs     = build_sequences(df, item2id, max_seq_len)
    train, val, test = leave_one_out_split(seqs)
    n_items  = len(item2id)
    pop      = compute_item_popularity(train, n_items)
    ips      = compute_ips_weights(pop)
    save_processed(train, val, test, item2id, id2item, pop, ips, out_dir, movies)
    print(f"✓ Preprocessing complete. n_items={n_items}, n_users={len(train)}")


if __name__ == "__main__":
    run_pipeline()
