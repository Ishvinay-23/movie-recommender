from surprise import Dataset, Reader, accuracy


def split_train_test_by_timestamp(ratings, test_ratio=0.20):
    ratings = ratings.sort_values(["userId", "timestamp"])

    user_counts = ratings.groupby("userId").size()
    test_counts = (user_counts * test_ratio).astype(int).clip(lower=1)

    test_indices = []
    for user_id, count in test_counts.items():
        user_ratings = ratings[ratings["userId"] == user_id]
        last_ratings = user_ratings.tail(count)
        test_indices.extend(last_ratings.index)

    test = ratings.loc[test_indices]
    train = ratings.drop(test.index)
    return train, test


def precision_at_k_for_user(user_id, recommender, test, K=10):
    user_test = test[(test["userId"] == user_id) & (test["rating"] >= 4.0)]
    relevant_movies = set(user_test["movieId"])
    recommendations = recommender(user_id, K)
    recommended_movies = set(recommendations["movieId"])
    hits = len(recommended_movies & relevant_movies)
    return hits / K


def evaluate_precision_at_k(recommender, test, K=10):
    users = test.loc[test["rating"] >= 4.0, "userId"].unique()
    scores = [precision_at_k_for_user(u, recommender, test, K) for u in users]
    return sum(scores) / len(scores)


def recall_at_k_for_user(user_id, recommender, test, K=10):
    user_test = test[(test["userId"] == user_id) & (test["rating"] >= 4.0)]
    relevant_movies = set(user_test["movieId"])
    if len(relevant_movies) == 0:
        return 0
    recommendations = recommender(user_id, K)
    recommended_movies = set(recommendations["movieId"])
    hits = len(recommended_movies & relevant_movies)
    return hits / len(relevant_movies)


def evaluate_recall_at_k(recommender, test, K=10):
    users = test.loc[test["rating"] >= 4.0, "userId"].unique()
    scores = [recall_at_k_for_user(u, recommender, test, K) for u in users]
    return sum(scores) / len(scores)


def evaluate_svd_rmse(model, train, test):
    reader = Reader(rating_scale=(0.5, 5.0))
    _ = Dataset.load_from_df(train[["userId", "movieId", "rating"]], reader)
    test_data = Dataset.load_from_df(test[["userId", "movieId", "rating"]], reader)
    testset = test_data.build_full_trainset().build_testset()
    svd_predictions = model.test(testset)
    return accuracy.rmse(svd_predictions)
