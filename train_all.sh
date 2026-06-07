#!/usr/bin/env bash
# train_all.sh — Run the full UserLens training pipeline end-to-end
# Usage: bash train_all.sh [--device cpu|cuda] [--data_dir data/processed]
set -euo pipefail

DEVICE="cpu"
DATA_DIR="data/processed"

while [[ $# -gt 0 ]]; do
  case $1 in
    --device)   DEVICE="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "============================================================"
echo " UserLens Training Pipeline"
echo " device=$DEVICE  data_dir=$DATA_DIR"
echo "============================================================"

# Stage 0: Preprocess data
echo ""
echo "─── Stage 0: Preprocessing ───"
python -m src.data.preprocessing

# Stage 1: BERT4Rec pre-training
echo ""
echo "─── Stage 1: BERT4Rec Pre-Training ───"
python -m src.training.pretrain \
  --data_dir "$DATA_DIR" \
  --output_dir experiments/pretrain \
  --n_epochs 50 \
  --batch_size 256 \
  --device "$DEVICE"

# Stage 2: Two-tower retrieval
echo ""
echo "─── Stage 2: Two-Tower Retrieval ───"
python -m src.training.train_retrieval \
  --data_dir "$DATA_DIR" \
  --pretrain_ckpt experiments/pretrain/best_bert4rec.pt \
  --output_dir experiments/retrieval \
  --n_epochs 30 \
  --device "$DEVICE"

# Stage 3: Ranker
echo ""
echo "─── Stage 3: Cross-Attention Ranker ───"
python -m src.training.train_ranking \
  --data_dir "$DATA_DIR" \
  --pretrain_ckpt experiments/pretrain/best_bert4rec.pt \
  --output_dir experiments/ranking \
  --n_epochs 20 \
  --device "$DEVICE"

echo ""
echo "============================================================"
echo " ✓ Training complete!"
echo " Run the API: uvicorn api.main:app --reload"
echo "============================================================"
