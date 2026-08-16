import pandas as pd

from src.evaluate import (
    evaluate_precision_at_k,
    evaluate_recall_at_k,
    precision_at_k_for_user,
    recall_at_k_for_user,
    split_train_test_by_timestamp,
)


def test_split_train_test_by_timestamp_has_all_rows():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [10, 11, 12, 20, 21],
            "rating": [4.0, 3.5, 5.0, 4.5, 2.0],
            "timestamp": [1, 2, 3, 1, 2],
        }
    )

    train, test = split_train_test_by_timestamp(ratings, test_ratio=0.20)

    assert len(train) + len(test) == len(ratings)
    assert set(train.index).isdisjoint(set(test.index))
    assert test.groupby("userId").size().min() >= 1


def test_precision_and_recall_at_k_simple_case():
    test = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [100, 101, 102],
            "rating": [5.0, 4.0, 3.0],
        }
    )

    def recommender(user_id, K=10):
        return pd.DataFrame({"movieId": [100, 999][:K]})

    p = precision_at_k_for_user(1, recommender, test, K=2)
    r = recall_at_k_for_user(1, recommender, test, K=2)

    assert p == 0.5
    assert r == 0.5


def test_evaluate_precision_and_recall_across_users():
    test = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 20, 21],
            "rating": [4.5, 3.0, 4.0, 2.5],
        }
    )

    recs = {
        1: [10, 99],
        2: [20, 98],
    }

    def recommender(user_id, K=10):
        return pd.DataFrame({"movieId": recs[user_id][:K]})

    precision = evaluate_precision_at_k(recommender, test, K=2)
    recall = evaluate_recall_at_k(recommender, test, K=2)

    assert precision == 0.5
    assert recall == 1.0
