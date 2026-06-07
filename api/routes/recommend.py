"""
api/routes/recommend.py
POST /recommend — full recommendation pipeline.
"""

from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendedItem, RecommendResponse
from api.app_state import state

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    # Resolve sequence
    if req.sequence is not None:
        sequence = req.sequence
        user_id = req.user_id
    elif req.user_id is not None:
        sequence = state.get_user_sequence(req.user_id)
        user_id = req.user_id
    else:
        raise HTTPException(status_code=400, detail="Provide user_id or sequence")

    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = state.pipeline.recommend(
        user_sequence=sequence,
        top_k=req.k,
        mode=req.mode,
        exclude_seen=req.exclude_seen,
    )

    items = []
    for rank, item in enumerate(result["items"], start=1):
        items.append(RecommendedItem(
            item_id=item["item_id"],
            title=item.get("title") or state.get_item_title(item["item_id"]),
            score=item["score"],
            rank=rank,
        ))

    return RecommendResponse(
        user_id=user_id,
        items=items,
        reasoning=result.get("reasoning", ""),
        pathway=result.get("pathway", "unknown"),
        retrieval_count=result.get("retrieval_count", len(items)),
        mode=req.mode,
    )
