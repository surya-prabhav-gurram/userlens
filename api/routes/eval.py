"""
api/routes/eval.py
GET /eval/latest, GET /eval/bias, POST /eval/run, GET /continual/status
"""

import asyncio
import datetime
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.app_state import state
from api.schemas import BiasAuditResult, ContinualStatus

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/eval/latest")
async def eval_latest():
    if state.last_eval_result:
        return state.last_eval_result
    return {
        "message": "No evaluation run yet. POST /eval/run to trigger.",
        "ndcg@10": 0.0, "hitrate@10": 0.0, "mrr@10": 0.0, "coverage": 0.0,
    }


@router.post("/eval/run")
async def eval_run(background_tasks: BackgroundTasks, n_users: int = 100):
    """Trigger an offline evaluation run in the background."""
    background_tasks.add_task(_run_eval, n_users)
    return {"status": "evaluation started", "n_users": n_users}


async def _run_eval(n_users: int):
    try:
        from src.evaluation.metrics import evaluate_pipeline
        if state.pipeline is None:
            return
        result = evaluate_pipeline(
            state.pipeline,
            state.test_targets,
            state.train_seqs,
            ks=[10, 20],
            n_users=n_users,
            popularity=state.popularity,
        )
        result["timestamp"] = datetime.datetime.utcnow().isoformat()
        state.last_eval_result = result
        logger.info(f"Eval complete: NDCG@10={result.get('ndcg@10', 0):.4f}")
    except Exception as e:
        logger.error(f"Eval failed: {e}")


@router.get("/eval/bias")
async def eval_bias():
    if state.last_bias_audit:
        return state.last_bias_audit
    return {"message": "No bias audit run yet. POST /eval/bias/run to trigger."}


@router.post("/eval/bias/run")
async def bias_run(background_tasks: BackgroundTasks, n_users: int = 100):
    background_tasks.add_task(_run_bias_audit, n_users)
    return {"status": "bias audit started"}


async def _run_bias_audit(n_users: int):
    try:
        from src.evaluation.bias_audit import compute_full_bias_audit
        if state.pipeline is None or state.popularity is None:
            return
        result = compute_full_bias_audit(
            state.pipeline,
            state.test_targets,
            state.train_seqs,
            state.popularity,
            n_users=n_users,
        )
        result["timestamp"] = datetime.datetime.utcnow().isoformat()
        state.last_bias_audit = result
        logger.info("Bias audit complete")
    except Exception as e:
        logger.error(f"Bias audit failed: {e}")


@router.get("/continual/status", response_model=ContinualStatus)
async def continual_status():
    return ContinualStatus(
        last_run=state.last_continual_run,
        n_new_interactions=len(state.new_interactions),
        n_total_interactions=sum(len(v) for v in state.train_seqs.values()),
        last_loss=None,
    )


@router.post("/continual/trigger")
async def continual_trigger(background_tasks: BackgroundTasks):
    """Manually trigger a continual learning update."""
    background_tasks.add_task(_run_continual)
    return {"status": "continual learning update triggered"}


async def _run_continual():
    import datetime
    try:
        from src.training.continual import run_continual_update
        if not state.new_interactions:
            logger.info("No new interactions for continual update")
            return

        # Build new sequences from queued interactions
        from collections import defaultdict
        new_seqs = defaultdict(list)
        for interaction in state.new_interactions:
            new_seqs[interaction["user_id"]].append(interaction["item_id"])

        backbone = getattr(state.two_tower, "user_tower", None)
        backbone = getattr(backbone, "backbone", None) if backbone else None
        if backbone is None:
            logger.warning("No backbone available for continual update")
            return

        result = run_continual_update(
            model=backbone,
            new_sequences=dict(new_seqs),
            historical_sequences=state.train_seqs,
        )

        state.new_interactions.clear()
        state.last_continual_run = datetime.datetime.utcnow().isoformat()
        logger.info(f"Continual update complete: {result}")
    except Exception as e:
        logger.error(f"Continual update failed: {e}")
