def build_popularity_model(train, movies, M=50):
    train_movie_stats = train.groupby("movieId")["rating"].agg(["mean", "count"])
    train_movie_stats = train_movie_stats.rename(columns={"mean": "avg_rating", "count": "rating_count"})

    C_train = train["rating"].mean()

    train_movie_stats["weighted_score"] = (
        (train_movie_stats["rating_count"] / (train_movie_stats["rating_count"] + M)) * train_movie_stats["avg_rating"]
        + (M / (train_movie_stats["rating_count"] + M)) * C_train
    )

    train_popularity = train_movie_stats.sort_values("weighted_score", ascending=False).reset_index()
    train_popularity = train_popularity.merge(movies[["movieId", "title"]], on="movieId")

    return train_movie_stats, train_popularity, C_train


def recommend_movies(user_id, train, train_popularity, K=10):
    user_rated_movies = train[train["userId"] == user_id]["movieId"]
    recommendations = train_popularity[~train_popularity["movieId"].isin(user_rated_movies)]
    return recommendations.head(K)
