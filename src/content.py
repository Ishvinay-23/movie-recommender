import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


MIN_RATINGS = 20


def build_genre_features(movies):
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].str.split("|")

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genre_list"])
    return movies, mlb, genre_matrix


def get_user_taste_vector(user_id, train, movies, genre_matrix):
    user_ratings = train[(train["userId"] == user_id) & (train["rating"] >= 4.0)]

    if user_ratings.empty:
        return None

    rated_movie_indices = movies[movies["movieId"].isin(user_ratings["movieId"])].index
    user_genre_matrix = genre_matrix[rated_movie_indices]

    weights = user_ratings["rating"].values
    user_profile = (user_genre_matrix * weights[:, None]).sum(axis=0) / weights.sum()

    return user_profile


def recommend_movies_content(user_id, train, movies, genre_matrix, train_movie_stats, K=10):
    user_profile = get_user_taste_vector(user_id, train, movies, genre_matrix)

    if user_profile is None:
        return pd.DataFrame(columns=["movieId", "title", "content_score"])

    user_rated_movies = set(train[train["userId"] == user_id]["movieId"])
    unrated_mask = ~movies["movieId"].isin(user_rated_movies)
    unrated_movies = movies[unrated_mask].copy()

    unrated_movies = unrated_movies.merge(
        train_movie_stats[["rating_count"]], left_on="movieId", right_index=True, how="left"
    )
    unrated_movies["rating_count"] = unrated_movies["rating_count"].fillna(0)
    unrated_movies = unrated_movies[unrated_movies["rating_count"] >= MIN_RATINGS]

    unrated_indices = unrated_movies.index
    unrated_genre_matrix = genre_matrix[unrated_indices]

    similarity_scores = cosine_similarity(
        unrated_genre_matrix, user_profile.reshape(1, -1)
    ).flatten()

    unrated_movies["content_score"] = similarity_scores
    recommendations = unrated_movies.sort_values("content_score", ascending=False)

    return recommendations[["movieId", "title", "content_score"]].head(K)
