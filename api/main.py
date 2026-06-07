from dotenv import load_dotenv
load_dotenv()
"""
api/main.py
FastAPI application entry point.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import recommend, feedback, eval as eval_routes
from api.app_state import load_state, state
from api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UserLens",
    description="Adaptive Recommendation System — BERT4Rec + Two-Tower + LLM Re-ranking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(eval_routes.router)


@app.on_event("startup")
async def startup():
    data_dir = os.getenv("DATA_DIR", "data/processed")
    device   = os.getenv("DEVICE", "cpu")
    logger.info(f"Starting UserLens API — data_dir={data_dir}, device={device}")
    load_state(data_dir=data_dir, device=device)
    logger.info("UserLens API ready")


@app.get("/health", response_model=HealthResponse)
async def health():
    db_status = "ok" if (state.retriever and state.retriever._conn) else "in-memory"
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        db_status=db_status,
        n_items=state.n_items,
        n_users=len(state.train_seqs),
    )


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    title = state.get_item_title(item_id)
    pop = float(state.popularity[item_id]) if (state.popularity is not None and item_id < len(state.popularity)) else 0.0
    return {"item_id": item_id, "title": title, "popularity": pop}


@app.get("/users")
async def list_users(limit: int = 20):
    users = list(state.train_seqs.keys())[:limit]
    return {
        "users": [
            {
                "user_id": int(uid),
                "n_interactions": len(state.train_seqs[uid]),
                "last_5_items": [int(i) for i in state.train_seqs[uid][-5:]],
            }
            for uid in users
        ]
    }


@app.get("/users/{user_id}/history")
async def user_history(user_id: int, limit: int = 20):
    seq = state.get_user_sequence(user_id)
    items = [
        {"item_id": int(iid), "title": state.get_item_title(iid)}
        for iid in seq[-limit:]
    ]
    return {"user_id": int(user_id), "n_interactions": len(seq), "history": items}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
