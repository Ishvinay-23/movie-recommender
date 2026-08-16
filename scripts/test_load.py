import pickle
import numpy as np

from surprise import dump


# ==========================================
# 1. LOAD SAVED ARTIFACTS
# ==========================================

print("Loading saved artifacts...")

_, svd_model = dump.load(
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


# ==========================================
# 2. SANITY CHECK
# ==========================================

print("\n===== LOADED ARTIFACTS =====")

print("Genre matrix shape:", genre_matrix.shape)
print("Train shape:", train.shape)
print("Movies shape:", movies.shape)
print("Movie stats shape:", train_movie_stats.shape)
print("Popularity shape:", train_popularity.shape)


# ==========================================
# 3. TEST SVD MODEL
# ==========================================

user_id = 1
movie_id = 318

prediction = svd_model.predict(
    user_id,
    movie_id
)

print("\n===== SVD TEST =====")
print("User:", user_id)
print("Movie:", movie_id)
print("Predicted rating:", prediction.est)


# ==========================================
# 4. TEST MOVIE LOOKUP
# ==========================================

movie = movies[
    movies["movieId"] == movie_id
]

print("\n===== MOVIE LOOKUP =====")
print(movie[["movieId", "title", "genres"]])