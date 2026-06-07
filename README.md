# UserLens — Adaptive Recommendation System

**BERT4Rec · Two-Tower Retrieval · Cross-Attention Ranking · LLM Re-ranking · Continual Learning**

Built for [Surya Prabhav Gurram](mailto:suryaprabhavg@gmail.com) | [GitHub](https://github.com/surya-prabhav-gurram)

---

## Quick Start

### 1. Clone & configure
```bash
git clone <your-repo>
cd userlens
cp env.example .env
# Edit .env — set ANTHROPIC_API_KEY
```

### 2. Run with Docker Compose (recommended)
```bash
docker compose up --build
# API:     http://localhost:8000
# UI:      http://localhost:3000
# MLflow:  http://localhost:5000
```

### 3. Download data & preprocess
```bash
# Download MovieLens-25M from https://grouplens.org/datasets/movielens/25m/
# Extract to data/raw/ml-25m/
python -m src.data.preprocessing
```

### 4. Train (local)
```bash
pip install -r requirements.txt
bash train_all.sh --device cpu
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

---

## Results (expected after full training)

| Metric | Target |
|--------|--------|
| NDCG@10 | > 0.15 |
| HitRate@10 | > 0.25 |
| Recall@100 (retrieval) | > 0.50 |
| Pairwise accuracy (ranker) | > 0.75 |
