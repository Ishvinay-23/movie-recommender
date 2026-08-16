import pickle

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from surprise import dump

from src.hybrid import recommend_movies_hybrid


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid movie recommendation system",
    version="1.0.0"
)


# ==========================================
# LOAD TRAINED ARTIFACTS ONCE
# ==========================================

print("Loading trained artifacts...")

_, svd_default = dump.load(
    "models/svd_model.pkl"
)

genre_matrix = np.load(
    "models/genre_matrix.npy"
)

with open("models/train.pkl", "rb") as f:
    train = pickle.load(f)

with open("models/movies.pkl", "rb") as f:
    movies = pickle.load(f)

with open("models/train_movie_stats.pkl", "rb") as f:
    train_movie_stats = pickle.load(f)

with open("models/train_popularity.pkl", "rb") as f:
    train_popularity = pickle.load(f)


print("All artifacts loaded successfully!")
print("Genre matrix:", genre_matrix.shape)
print("Train:", train.shape)
print("Movies:", movies.shape)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API is running"
    }


# ==========================================
# RECOMMENDATION ENDPOINT
# ==========================================

@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    k: int = Query(
        default=10,
        ge=1,
        le=100
    )
):
    if user_id not in train["userId"].values:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )

    recommendations = recommend_movies_hybrid(
        user_id=user_id,
        svd_model=svd_default,
        train=train,
        movies=movies,
        genre_matrix=genre_matrix,
        train_movie_stats=train_movie_stats,
        K=k,
        svd_weight=0.8
    )

    if recommendations.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations available for user {user_id}"
        )

    return {
        "user_id": user_id,
        "recommendations": recommendations.to_dict(
            orient="records"
        )
    }
