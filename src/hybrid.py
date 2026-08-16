import pandas as pd

from src.collaborative import get_all_predictions
from src.content import recommend_movies_content


def recommend_movies_hybrid(user_id, svd_model, train, movies, genre_matrix, train_movie_stats, K=10, svd_weight=0.8):
    svd_preds = get_all_predictions(svd_model, user_id, train, movies)
    content_preds = recommend_movies_content(
        user_id, train, movies, genre_matrix, train_movie_stats, K=len(movies)
    )[["movieId", "content_score"]]

    hybrid = svd_preds.merge(content_preds, on="movieId", how="inner")

    if hybrid.empty:
        return pd.DataFrame(columns=["movieId", "title", "hybrid_score"])

    svd_min, svd_max = hybrid["predicted_rating"].min(), hybrid["predicted_rating"].max()
    content_min, content_max = hybrid["content_score"].min(), hybrid["content_score"].max()

    hybrid["svd_score"] = (hybrid["predicted_rating"] - svd_min) / (svd_max - svd_min)
    hybrid["content_normalized"] = (hybrid["content_score"] - content_min) / (content_max - content_min)

    hybrid["hybrid_score"] = (
        svd_weight * hybrid["svd_score"] + (1 - svd_weight) * hybrid["content_normalized"]
    )

    hybrid = hybrid.merge(movies[["movieId", "title"]], on="movieId")
    recommendations = hybrid.sort_values("hybrid_score", ascending=False)

    return recommendations[["movieId", "title", "hybrid_score"]].head(K)
