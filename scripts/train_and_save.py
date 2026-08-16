import os
import pickle
import sys
from pathlib import Path

import numpy as np
from surprise import dump


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_popularity_model
from src.collaborative import train_svd_default
from src.content import build_genre_features
from src.data_loader import load_data
from src.evaluate import split_train_test_by_timestamp


# ==========================================
# 1. CREATE MODELS DIRECTORY
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

ratings, movies = load_data("data/raw/ml-latest-small")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)


# ==========================================
# 3. TRAIN / TEST SPLIT
# Same split used in main.py
# ==========================================

train, test = split_train_test_by_timestamp(ratings, test_ratio=0.20)

print("Train size:", len(train))
print("Test size:", len(test))


# ==========================================
# 4. POPULARITY STATISTICS
# ==========================================

train_movie_stats, train_popularity, _ = build_popularity_model(train, movies, M=50)

print("Popularity statistics created.")


# ==========================================
# 5. TRAIN DEFAULT SVD
# ==========================================

print("Starting SVD training...")
svd_default = train_svd_default(train)
print("SVD training completed!")


# ==========================================
# 6. CREATE GENRE MATRIX
# ==========================================

movies_with_genres, mlb, genre_matrix = build_genre_features(movies)

print("Genre matrix shape:", genre_matrix.shape)
print("Number of genres:", len(mlb.classes_))


# ==========================================
# 7. SAVE SVD MODEL
# ==========================================

dump.dump(
    "models/svd_model.pkl",
    algo=svd_default
)

print("Saved: models/svd_model.pkl")


# ==========================================
# 8. SAVE GENRE MATRIX
# ==========================================

np.save(
    "models/genre_matrix.npy",
    genre_matrix
)

print("Saved: models/genre_matrix.npy")


# ==========================================
# 9. SAVE DATAFRAMES
# ==========================================

with open("models/train.pkl", "wb") as f:
    pickle.dump(train, f)

with open("models/movies.pkl", "wb") as f:
    pickle.dump(movies_with_genres, f)

with open("models/train_movie_stats.pkl", "wb") as f:
    pickle.dump(train_movie_stats, f)

with open("models/train_popularity.pkl", "wb") as f:
    pickle.dump(train_popularity, f)

print("Saved: models/train.pkl")
print("Saved: models/movies.pkl")
print("Saved: models/train_movie_stats.pkl")
print("Saved: models/train_popularity.pkl")


# ==========================================
# 10. FINAL SANITY CHECK
# ==========================================

print("\n===== ARTIFACT SUMMARY =====")

print("SVD model: trained")
print("Genre matrix:", genre_matrix.shape)
print("Train dataframe:", train.shape)
print("Movies dataframe:", movies_with_genres.shape)
print("Movie statistics:", train_movie_stats.shape)
print("Popularity dataframe:", train_popularity.shape)

print("\nAll training artifacts saved successfully!")
