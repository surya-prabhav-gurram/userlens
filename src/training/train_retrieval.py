"""
src/training/train_retrieval.py
Stage 2: Two-tower retrieval model fine-tuning with in-batch negatives.
Initializes from pre-trained BERT4Rec backbone.
"""

import os
import logging
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.data.preprocessing import load_processed
from src.data.dataset import TwoTowerDataset
from src.models.bert4rec import BERT4Rec
from src.models.two_tower import TwoTowerModel
from src.training.pretrain import load_pretrained

logger = logging.getLogger(__name__)


def recall_at_k(user_embs: torch.Tensor, item_embs: torch.Tensor, k: int = 10) -> float:
    """In-batch Recall@K: fraction of users where true item is in top-K."""
    sim = torch.matmul(user_embs, item_embs.T)  # (B, B)
    B = sim.size(0)
    labels = torch.arange(B, device=sim.device)
    topk_indices = sim.topk(k, dim=1).indices  # (B, k)
    hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def train_retrieval(
    data_dir:       str   = "data/processed",
    pretrain_ckpt:  Optional[str] = None,
    output_dir:     str   = "experiments/retrieval",
    d_out:          int   = 128,
    temperature:    float = 0.07,
    batch_size:     int   = 256,
    n_epochs:       int   = 30,
    lr:             float = 3e-4,
    weight_decay:   float = 0.01,
    patience:       int   = 5,
    device:         str   = "auto",
    mlflow_uri:     Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data = load_processed(data_dir)

    train_ds = TwoTowerDataset(data["train_seqs"], data["val_targets"], split="train")
    val_ds   = TwoTowerDataset(data["train_seqs"], data["val_targets"], split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Load backbone
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        logger.info(f"Loading pre-trained backbone from {pretrain_ckpt}")
        backbone = load_pretrained(pretrain_ckpt, device)
    else:
        logger.info("No pretrain checkpoint — training two-tower from scratch")
        backbone = BERT4Rec(n_items=data["n_items"]).to(device)

    model = TwoTowerModel(backbone, d_out=d_out, temperature=temperature).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("userlens_retrieval")
        mlflow.start_run(run_name="two_tower_train")

    best_recall = 0.0
    no_improve  = 0
    best_ckpt   = os.path.join(output_dir, "best_two_tower.pt")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        train_recall = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            pos_item  = batch["pos_item"].to(device)

            out = model(input_ids, attn_mask, pos_item)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_recall += recall_at_k(out["user_emb"], out["item_emb"], k=10)

        scheduler.step()
        avg_train_loss   = train_loss / len(train_loader)
        avg_train_recall = train_recall / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_recall = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attn_mask"].to(device)
                pos_item  = batch["pos_item"].to(device)
                out = model(input_ids, attn_mask, pos_item)
                val_loss   += out["loss"].item()
                val_recall += recall_at_k(out["user_emb"], out["item_emb"], k=10)

        avg_val_loss   = val_loss / max(len(val_loader), 1)
        avg_val_recall = val_recall / max(len(val_loader), 1)

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} train_R@10={avg_train_recall:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_R@10={avg_val_recall:.4f}"
        )

        if MLFLOW_AVAILABLE and mlflow_uri:
            mlflow.log_metrics({
                "train_loss": avg_train_loss, "train_recall10": avg_train_recall,
                "val_loss": avg_val_loss, "val_recall10": avg_val_recall,
            }, step=epoch)

        if avg_val_recall > best_recall:
            best_recall = avg_val_recall
            no_improve  = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_recall10": best_recall,
                "n_items": data["n_items"],
                "d_out": d_out,
                "temperature": temperature,
            }, best_ckpt)
            logger.info(f"  ✓ Best model saved (val_R@10={best_recall:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.end_run()

    logger.info(f"Two-tower training complete. Best: {best_ckpt}")
    return best_ckpt


def load_two_tower(ckpt_path: str, backbone: BERT4Rec, device: str = "cpu") -> TwoTowerModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TwoTowerModel(
        backbone,
        d_out=ckpt.get("d_out", 128),
        temperature=ckpt.get("temperature", 0.07),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="data/processed")
    parser.add_argument("--pretrain_ckpt", default="experiments/pretrain/best_bert4rec.pt")
    parser.add_argument("--output_dir",    default="experiments/retrieval")
    parser.add_argument("--n_epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--device",       default="mps")
    parser.add_argument("--mlflow_uri",    default=None)
    args = parser.parse_args()
    train_retrieval(**vars(args))
