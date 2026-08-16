import pandas as pd
from surprise import Dataset, Reader, SVD


def train_svd_models(train):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd_default = SVD(n_factors=100, n_epochs=20, reg_all=0.02, random_state=42)
    svd_default.fit(trainset)

    svd_reg = SVD(n_factors=100, n_epochs=20, reg_all=0.10, random_state=42)
    svd_reg.fit(trainset)

    svd_20 = SVD(n_factors=20, n_epochs=20, reg_all=0.02, random_state=42)
    svd_20.fit(trainset)

    return svd_default, svd_reg, svd_20


def train_svd_default(train):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd_default = SVD(n_factors=100, n_epochs=20, reg_all=0.02, random_state=42)
    svd_default.fit(trainset)
    return svd_default


def get_all_predictions(model, user_id, train, movies):
    user_rated_movies = set(train[train["userId"] == user_id]["movieId"])
    unrated_movies = movies[~movies["movieId"].isin(user_rated_movies)]

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in unrated_movies["movieId"]
    ]
    return pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])


def recommend_movies_svd(model, user_id, train, movies, K=10):
    predictions_df = get_all_predictions(model, user_id, train, movies)
    recommendations = predictions_df.sort_values("predicted_rating", ascending=False)
    recommendations = recommendations.merge(movies[["movieId", "title"]], on="movieId")
    return recommendations.head(K)
