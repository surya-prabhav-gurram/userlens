"""
api/routes/feedback.py
POST /feedback — log a new user-item interaction.
"""

import time
from fastapi import APIRouter
from api.schemas import FeedbackRequest
from api.app_state import state

router = APIRouter()


@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    state.add_interaction(req.user_id, req.item_id, req.interaction_type)
    return {"status": "ok", "queued_interactions": len(state.new_interactions)}
