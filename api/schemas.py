from pydantic import BaseModel, Field
from typing import List


# ==========================================
# RECOMMENDATION ITEM
# ==========================================

class RecommendationItem(BaseModel):
    movieId: int
    title: str
    score: float


# ==========================================
# RECOMMENDATION RESPONSE
# ==========================================

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]