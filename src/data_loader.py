import pandas as pd


DEFAULT_DATA_DIR = "data/raw/ml-latest-small"


def load_data(data_dir=DEFAULT_DATA_DIR):
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    return ratings, movies
