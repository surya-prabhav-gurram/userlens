"""
src/training/train_ranking.py
Stage 3: Cross-attention ranker training with IPS-weighted BPR loss.
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
from src.data.dataset import RankingDataset
from src.models.bert4rec import BERT4Rec
from src.models.ranker import CrossAttentionRanker
from src.training.pretrain import load_pretrained

logger = logging.getLogger(__name__)


def train_ranking(
    data_dir:       str   = "data/processed",
    pretrain_ckpt:  Optional[str] = None,
    output_dir:     str   = "experiments/ranking",
    batch_size:     int   = 128,
    n_epochs:       int   = 20,
    lr:             float = 1e-4,
    weight_decay:   float = 0.01,
    patience:       int   = 5,
    device:         str   = "auto",
    use_ips:        bool  = True,
    mlflow_uri:     Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data = load_processed(data_dir)
    popularity  = data["popularity"]
    ips_weights = data["ips_weights"] if use_ips else None

    train_ds = RankingDataset(
        data["train_seqs"], data["n_items"],
        ips_weights=ips_weights,
        popularity=popularity,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    logger.info(f"Ranking dataset: {len(train_ds)} samples")

    # Backbone
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        logger.info(f"Loading backbone from {pretrain_ckpt}")
        backbone = load_pretrained(pretrain_ckpt, device)
    else:
        backbone = BERT4Rec(n_items=data["n_items"]).to(device)

    model = CrossAttentionRanker(backbone).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Ranker parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("userlens_ranking")
        mlflow.start_run(run_name="ranker_train")
        mlflow.log_params({"use_ips": use_ips, "batch_size": batch_size, "lr": lr})

    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(output_dir, "best_ranker.pt")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_pos_higher = 0
        n_total = 0

        for batch in train_loader:
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attn_mask"].to(device)
            pos_item    = batch["pos_item"].to(device)
            neg_item    = batch["neg_item"].to(device)
            ips_weight  = batch["ips_weight"].to(device)

            out = model(input_ids, attn_mask, pos_item, neg_item, ips_weight)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_pos_higher += (out["score_pos"] > out["score_neg"]).sum().item()
            n_total += pos_item.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        pairwise_acc = n_pos_higher / max(n_total, 1)

        logger.info(f"Epoch {epoch:3d}/{n_epochs} | loss={avg_loss:.4f} | pairwise_acc={pairwise_acc:.4f}")

        if MLFLOW_AVAILABLE and mlflow_uri:
            mlflow.log_metrics({"train_loss": avg_loss, "pairwise_acc": pairwise_acc}, step=epoch)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "train_loss": avg_loss, "n_items": data["n_items"],
            }, best_ckpt)
            logger.info(f"  ✓ Best ranker saved (loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.end_run()

    logger.info(f"Ranking training complete. Best: {best_ckpt}")
    return best_ckpt


def load_ranker(ckpt_path: str, backbone: BERT4Rec, device: str = "cpu") -> CrossAttentionRanker:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CrossAttentionRanker(backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="data/processed")
    parser.add_argument("--pretrain_ckpt", default="experiments/pretrain/best_bert4rec.pt")
    parser.add_argument("--output_dir",    default="experiments/ranking")
    parser.add_argument("--n_epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--device",       default="mps")
    parser.add_argument("--no_ips",        action="store_true")
    args = parser.parse_args()
    train_ranking(use_ips=not args.no_ips, **{k: v for k, v in vars(args).items() if k != "no_ips"})
