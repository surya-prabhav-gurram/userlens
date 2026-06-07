# UserLens — Adaptive Recommendation System

**BERT4Rec · Two-Tower Retrieval · Cross-Attention Ranking · LLM Re-ranking · Continual Learning**

Built by [Surya Prabhav Gurram](mailto:suryaprabhavg@gmail.com) | [GitHub](https://github.com/surya-prabhav-gurram)

---

## Quick Start

### 1. Clone & configure
```bash
git clone git@github.com:surya-prabhav-gurram/userlens.git
cd userlens
cp env.example .env
# Edit .env — set ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Run with Docker Compose (recommended)
```bash
docker compose up --build
# API:     http://localhost:8000/docs
# UI:      http://localhost:3000
# MLflow:  http://localhost:5000
```

### 3. Run locally (no Docker)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Frontend (new terminal)
cd frontend && npm install && npm start
```

### 4. Download data & preprocess
```bash
# Download MovieLens-25M from https://grouplens.org/datasets/movielens/25m/
# Extract to data/raw/ml-25m/
python3 -m src.data.preprocessing
```

### 5. Train all stages
```bash
caffeinate -i bash -c "
python3 -m src.training.pretrain --n_epochs 50 --batch_size 64 --device mps --d_model 256 --max_users 162541 && \
python3 -m src.training.train_retrieval --pretrain_ckpt experiments/pretrain/best_bert4rec.pt --device mps && \
python3 -m src.training.train_ranking --pretrain_ckpt experiments/pretrain/best_bert4rec.pt --device mps && \
echo 'ALL TRAINING COMPLETE'
"
```

---

## Project Structure

```
userlens/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── app_state.py
│   ├── schemas.py
│   └── routes/
│       ├── recommend.py    # POST /recommend
│       ├── feedback.py     # POST /feedback
│       └── eval.py         # GET /eval/*, POST /continual/*
├── src/
│   ├── data/               # Preprocessing, datasets, augmentation
│   ├── models/             # BERT4Rec, TwoTower, Ranker, ColdStart
│   ├── training/           # Pretrain, retrieval, ranking, continual
│   ├── inference/          # Pipeline, retriever, LLM reranker
│   └── evaluation/         # Metrics, LLM judge, bias audit
├── frontend/               # React UI (4 views)
├── experiments/            # Trained model checkpoints
│   ├── pretrain/best_bert4rec.pt
│   ├── retrieval/best_two_tower.pt
│   └── ranking/best_ranker.pt
├── data/processed/         # Preprocessed MovieLens-25M
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── train_all.sh
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /recommend | Full pipeline (neural / llm / hybrid) |
| POST | /feedback | Log interaction → continual learning queue |
| GET  | /eval/latest | Latest offline metrics |
| POST | /eval/run | Trigger eval run |
| GET  | /eval/bias | Bias audit results |
| POST | /eval/bias/run | Trigger bias audit |
| POST | /continual/trigger | Trigger incremental fine-tuning |
| GET  | /health | Service health |
| GET  | /users | List users |
| GET  | /users/{id}/history | User watch history |
| GET  | /items/{id} | Item details |

---

## Achieved Results

Trained on MovieLens-25M (162,541 users, 32,720 items) on Apple MPS.

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall@10 (two-tower retrieval) | > 0.50 | **0.51** ✅ |
| Pairwise accuracy (ranker) | > 0.75 | **0.97** ✅ |
| NDCG@10 (end-to-end) | > 0.15 | 0.035 |
| HitRate@10 | > 0.25 | 0.076 |

NDCG gap vs target is due to d_model=256 with early stopping at epoch 19. Retrieval and ranking stages exceed targets. Recommendation quality is qualitatively strong — users with arthouse taste receive Tarkovsky, Haneke, and Kieslowski recommendations.

---

## Training Details

| Stage | Model | Params | Metric |
|-------|-------|--------|--------|
| Stage 1 — BERT4Rec pretrain | Transformer (d=256, L=2, H=4) | 10M | val_loss 9.10 |
| Stage 2 — Two-tower retrieval | Dual encoder + InfoNCE | 10M | Recall@10 0.51 |
| Stage 3 — Cross-attn ranker | Cross-attention + BPR | 10.5M | Pairwise acc 0.97 |

---

## Recommendation Modes

- **Neural** — Two-tower retrieval → cross-attention ranking
- **LLM Re-rank** — Neural candidates → Claude reranks with natural language reasoning
- **Hybrid** — Blends neural scores with LLM reranking
- **Cold Start** — Content-based fallback for users with < 3 interactions, switches to neural at ≥ 3
