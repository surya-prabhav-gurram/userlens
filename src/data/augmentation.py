"""
src/data/augmentation.py
LLM-based generative data augmentation for long-tail items.
Uses Claude API to generate synthetic interaction sequences.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AUGMENT_PROMPT = """You are generating synthetic user interaction data for a recommendation system.

Target item: {item_title} (genre: {genre})

Here are real interaction sequences from users who liked similar items:
{few_shot_examples}

Generate {n_sequences} plausible interaction sequences for different user archetypes
who would interact with the target item. Each sequence should be a coherent list of
item IDs from the vocabulary that reflects a realistic user taste profile.

Available item IDs range from 3 to {max_item_id}.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
[{{"user_profile": "brief description", "sequence": [list of 5-15 item ids], "reasoning": "why this profile"}}]"""


def get_tail_items(
    popularity: np.ndarray,
    item2id: Dict,
    threshold: int = 20,
) -> List[int]:
    """Return model item IDs for items with fewer than threshold interactions."""
    tail = [mid for mid in item2id.values() if mid < len(popularity) and popularity[mid] < threshold]
    return tail


def build_augmentation_prompt(
    item_id: int,
    item_meta_df,
    id2item: Dict,
    train_seqs: Dict,
    popularity: np.ndarray,
    n_sequences: int = 5,
    n_similar: int = 3,
) -> str:
    raw_id = id2item.get(item_id, item_id)
    meta_row = None
    if item_meta_df is not None and len(item_meta_df) > 0:
        rows = item_meta_df[item_meta_df["item_id"] == raw_id]
        if len(rows) > 0:
            meta_row = rows.iloc[0]

    title = meta_row["title"] if meta_row is not None else f"Item {raw_id}"
    genre = meta_row.get("genres", "Unknown") if meta_row is not None else "Unknown"

    # Find popular items as few-shot examples
    popular_items = np.argsort(popularity)[-20:][::-1]
    example_seqs = []
    for pop_item in popular_items[:n_similar]:
        for uid, seq in list(train_seqs.items())[:100]:
            if pop_item in seq:
                snippet = seq[-8:]
                example_seqs.append(str(snippet))
                break
    few_shot = "\n".join(example_seqs[:n_similar]) if example_seqs else "[[3, 4, 5, 6, 7]]"
    max_item_id = max(id2item.keys()) if id2item else 50000

    return AUGMENT_PROMPT.format(
        item_title=title,
        genre=genre,
        few_shot_examples=few_shot,
        n_sequences=n_sequences,
        max_item_id=max_item_id,
    )


def parse_augmentation_response(text: str, valid_ids: set, item_id: int) -> List[List[int]]:
    """Parse Claude's JSON response and validate item IDs."""
    try:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        sequences = []
        for entry in data:
            seq = entry.get("sequence", [])
            # Ensure target item is in the sequence
            seq = [int(i) for i in seq if int(i) in valid_ids]
            if item_id not in seq:
                seq = [item_id] + seq
            if len(seq) >= 3:
                sequences.append(seq)
        return sequences
    except Exception as e:
        logger.warning(f"Failed to parse augmentation response: {e}")
        return [[item_id, item_id + 1, item_id + 2]]  # minimal fallback


async def augment_tail_items_async(
    tail_item_ids: List[int],
    item_meta_df,
    id2item: Dict,
    train_seqs: Dict,
    popularity: np.ndarray,
    anthropic_api_key: str,
    n_sequences_per_item: int = 5,
    max_items: int = 100,
) -> Dict[int, List[List[int]]]:
    """
    Generate synthetic sequences for tail items using Claude API.
    Returns dict: item_id → list of synthetic sequences.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — skipping augmentation")
        return {}

    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    valid_ids = set(id2item.keys())
    results: Dict[int, List[List[int]]] = {}

    items_to_process = tail_item_ids[:max_items]
    logger.info(f"Augmenting {len(items_to_process)} tail items...")

    for i, item_id in enumerate(items_to_process):
        prompt = build_augmentation_prompt(
            item_id, item_meta_df, id2item, train_seqs, popularity, n_sequences_per_item
        )
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            sequences = parse_augmentation_response(text, valid_ids, item_id)
            results[item_id] = sequences
            if (i + 1) % 10 == 0:
                logger.info(f"  Augmented {i+1}/{len(items_to_process)} items")
        except Exception as e:
            logger.warning(f"Failed to augment item {item_id}: {e}")
            results[item_id] = [[item_id]]

    total_seqs = sum(len(v) for v in results.values())
    logger.info(f"Generated {total_seqs} synthetic sequences for {len(results)} items")
    return results


def save_synthetic_sequences(
    augmented: Dict[int, List[List[int]]],
    output_path: str = "data/synthetic/augmented_sequences.json",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flat = []
    for item_id, seqs in augmented.items():
        for seq in seqs:
            flat.append({"item_id": item_id, "sequence": seq, "is_synthetic": True})
    with open(output_path, "w") as f:
        json.dump(flat, f)
    logger.info(f"Saved {len(flat)} synthetic sequences to {output_path}")


def load_synthetic_sequences(path: str = "data/synthetic/augmented_sequences.json") -> List[List[int]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [entry["sequence"] for entry in data if len(entry["sequence"]) >= 3]
