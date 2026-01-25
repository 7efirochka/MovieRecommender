from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .recommender import recommend_by_hybride

app = FastAPI(
    title = "Movie recommender API",
    description = "Movie recommendations system based on your favorites films",
    version="1.0.0"
)

class RecommendationRequest(BaseModel):
    titles: str
    n: Optional[int] = 5

@app.get("/")
async def root():
    return {"message": "Welcome to the movie recommendation API"}

@app.post("/recommend")
async def reccomend(request: RecommendationRequest):
    try:
        recommendations = recommend_by_hybride(request.titles, request.n)
        return {"recommendations": recommendations}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")