"""
src/training/continual.py
Stage 6: Continual / incremental fine-tuning with experience replay.
Prevents catastrophic forgetting by mixing new + historical data.
"""

import os
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data.dataset import ContinualDataset
from src.models.bert4rec import BERT4Rec, masked_cross_entropy_loss

logger = logging.getLogger(__name__)


def run_continual_update(
    model: BERT4Rec,
    new_sequences: Dict[int, List[int]],
    historical_sequences: Optional[Dict[int, List[int]]] = None,
    replay_ratio: float = 0.2,
    n_steps: int = 500,
    lr: float = 1e-5,
    batch_size: int = 64,
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> dict:
    """
    Incrementally fine-tune model on new interactions with experience replay.
    Returns dict of training metrics.
    """
    dataset = ContinualDataset(
        new_sequences=new_sequences,
        historical_sequences=historical_sequences,
        replay_ratio=replay_ratio,
    )
    if len(dataset) == 0:
        logger.warning("No samples for continual update")
        return {"steps": 0, "loss": 0.0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        if steps >= n_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        pos_item  = batch["pos_item"].to(device)

        # For continual learning, treat positive item as masked target
        labels = torch.zeros_like(input_ids)
        # Mask the last real token position
        lengths = attn_mask.long().sum(dim=1) - 1
        for b in range(input_ids.size(0)):
            labels[b, lengths[b]] = pos_item[b]
            input_ids[b, lengths[b]] = 1  # MASK_ID

        logits = model(input_ids, attn_mask)
        loss = masked_cross_entropy_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    logger.info(f"Continual update: {steps} steps, avg_loss={avg_loss:.4f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "continual_loss": avg_loss}, output_path)

    return {"steps": steps, "loss": avg_loss}
