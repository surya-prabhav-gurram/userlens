"""
src/training/pretrain.py
Stage 1: Self-supervised pre-training with masked item prediction.
Trains BERT4Rec on full interaction sequences — no explicit labels needed.
"""

import os
import logging
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.data.preprocessing import load_processed
from src.data.dataset import BERT4RecDataset
from src.models.bert4rec import BERT4Rec, masked_cross_entropy_loss

logger = logging.getLogger(__name__)


def pretrain(
    max_users:      int   = 20000,
    data_dir:       str   = "data/processed",
    output_dir:     str   = "experiments/pretrain",
    d_model:        int   = 256,
    n_heads:        int   = 4,
    n_layers:       int   = 2,
    d_ff:           int   = 1024,
    dropout:        float = 0.2,
    max_seq_len:    int   = 200,
    mask_prob:      float = 0.15,
    batch_size:     int   = 256,
    n_epochs:       int   = 50,
    lr:             float = 1e-4,
    weight_decay:   float = 0.01,
    patience:       int   = 5,
    device:         str   = "auto",
    mlflow_uri:     Optional[str] = None,
    run_name:       str   = "bert4rec_pretrain",
):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    data = load_processed(data_dir)
    train_seqs = data["train_seqs"]
    if max_users and max_users < len(train_seqs):
        import random
        uids = random.sample(list(train_seqs.keys()), max_users)
        train_seqs = {u: train_seqs[u] for u in uids}
    train_ds = BERT4RecDataset(train_seqs, data["n_items"], max_seq_len, mask_prob)
    val_ds   = BERT4RecDataset(
{uid: data["train_seqs"][uid] + [data["val_targets"][uid]]
         for uid in list(data["val_targets"].keys())[:20000] if uid in data["train_seqs"]},
        data["n_items"], max_seq_len, mask_prob
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = BERT4Rec(
        n_items=data["n_items"],
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,      # 10% warmup
        anneal_strategy="cos",
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("userlens_pretrain")
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "batch_size": batch_size, "lr": lr, "n_epochs": n_epochs,
            "mask_prob": mask_prob, "n_items": data["n_items"],
        })

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0
    best_ckpt     = os.path.join(output_dir, "best_bert4rec.pt")

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            logits = model(input_ids, attn_mask)
            loss   = masked_cross_entropy_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            # Accuracy at masked positions
            mask_pos = labels != 0
            if mask_pos.sum() > 0:
                preds = logits.argmax(dim=-1)
                train_correct += (preds[mask_pos] == labels[mask_pos]).sum().item()
                train_total   += mask_pos.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                attn_mask = batch["attn_mask"].to(device)
                logits    = model(input_ids, attn_mask)
                loss      = masked_cross_entropy_loss(logits, labels)
                val_loss += loss.item()
                mask_pos  = labels != 0
                if mask_pos.sum() > 0:
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds[mask_pos] == labels[mask_pos]).sum().item()
                    val_total   += mask_pos.sum().item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc      = val_correct / max(val_total, 1)
        current_lr   = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )

        if MLFLOW_AVAILABLE and mlflow_uri:
            mlflow.log_metrics({
                "train_loss": avg_train_loss, "train_acc": train_acc,
                "val_loss":   avg_val_loss,   "val_acc":   val_acc,
                "lr":         current_lr,
            }, step=epoch)

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"bert4rec_epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": avg_val_loss}, ckpt_path)

        # Best model & early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": avg_val_loss, "val_acc": val_acc,
                "n_items": data["n_items"], "d_model": d_model,
                "n_heads": n_heads, "n_layers": n_layers,
                "d_ff": d_ff, "max_seq_len": max_seq_len,
            }, best_ckpt)
            logger.info(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if MLFLOW_AVAILABLE and mlflow_uri:
        mlflow.log_artifact(best_ckpt)
        mlflow.end_run()

    logger.info(f"Pre-training complete. Best model: {best_ckpt}")
    return best_ckpt


def load_pretrained(ckpt_path: str, device: str = "cpu") -> BERT4Rec:
    """Load a pre-trained BERT4Rec checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BERT4Rec(
        n_items   = ckpt["n_items"],
        d_model   = ckpt.get("d_model",  256),
        n_heads   = ckpt.get("n_heads",  4),
        n_layers  = ckpt.get("n_layers", 2),
        d_ff      = ckpt.get("d_ff",     1024),
        max_seq_len = ckpt.get("max_seq_len", 200),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--output_dir",  default="experiments/pretrain")
    parser.add_argument("--n_epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--mlflow_uri",  default=None)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--d_model",      type=int, default=64)
    parser.add_argument("--max_users",    type=int, default=20000)
    args = parser.parse_args()
    pretrain(**vars(args))
