# Movie Recommendation Engine

A resume-defensible movie recommendation system built on the MovieLens `ml-latest-small` dataset (610 users, 9,724 movies, 100,836 ratings). Four recommendation strategies were built, evaluated on a leakage-safe train/test split, and compared using Precision@K and Recall@K — not just "it works," but a documented, quantified answer to *which approach works best, and why*.

## Results

| Model                           | Precision@10 | Recall@10 |
|----------------------------------|--------------|-----------|
| Popularity Baseline              | 0.0582       | 0.0502    |
| SVD (Collaborative Filtering)    | 0.0301       | 0.0237    |
| Content-Based (genre similarity) | 0.0176       | 0.0139    |
| Hybrid (SVD 0.8 + Content 0.2)   | 0.0340       | 0.0297    |

The popularity baseline wins outright — a known, defensible characteristic of small, sparse datasets, where a non-personalized "safe bet" is hard for personalized models to beat on Precision@K. The hybrid model beats both of the individual models it's built from (SVD alone and content alone), which is the correct, expected outcome for an ensemble and demonstrates the blend weight was empirically tuned, not guessed.

## What's actually interesting here

The value of this project isn't the numbers themselves — it's the diagnostic process behind them. Every time a model underperformed, it was investigated with controlled experiments before drawing a conclusion, rather than just reporting a worse number and moving on. Highlights:

- **SVD underperformed popularity on ranking metrics despite good RMSE (0.894).** Diagnosed and ruled out evaluation noise, cold-start item bias, and rating-ceiling tie-piles as the cause before concluding it's a structural mismatch between RMSE-optimized objectives and Precision@K-style ranking quality, compounded by dataset sparsity (~1.7%).
- **Content-based recommendations were initially unusable** (Precision@10 = 0.0051, recommending obscure titles). Diagnosed as the same selection-bias trap as the popularity baseline — genre similarity has no concept of movie quality, so an obscure flop and a beloved classic with identical genre tags score identically. Fixed with a minimum-ratings filter on the candidate pool, improving Precision@10 3.4x.
- **The hybrid blend weight was tuned empirically**, not assumed — a sweep across `svd_weight` values from 0.5 to 1.0 found 0.8 as the optimum, confirmed on both Precision@10 and Recall@10.

Full write-up of each stage's methodology and diagnostics is in [`movie-recommender-roadmap.md`](./movie-recommender-roadmap.md).

## Project structure

```
movie-recommender/
│
├── data/
│   ├── raw/                  # original MovieLens files (ratings.csv, movies.csv)
│   └── processed/            # generated train/test splits
│
├── notebooks/
│   └── 01_eda.ipynb          # exploratory analysis: sparsity, rating distribution
│
├── src/
│   ├── data_loader.py        # load/clean ratings & movies data
│   ├── baseline.py           # popularity-based recommender
│   ├── collaborative.py      # SVD / CF model
│   ├── content.py            # content-based (genre) model
│   ├── hybrid.py             # CF + content hybrid
│   └── evaluate.py           # Precision@K, Recall@K, RMSE, train/test split logic
│
├── models/                   # saved/serialized trained artifacts (SVD model, genre matrix, stats)
│
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # request/response models (Pydantic)
│
├── scripts/
│   ├── train_and_save.py      # trains all models once, persists artifacts to models/
│   └── test_load.py           # verifies artifacts reload cleanly in a fresh process
│
├── tests/
│   └── test_evaluate.py
│
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — established dataset sparsity (~1.7%) and rating distribution skew, and why both matter for baseline design.
2. **Popularity Baseline** — Bayesian-weighted score (shrinkage toward the global mean for low-count movies) rather than a raw average, avoiding the bias toward obscure niche titles.
3. **Collaborative Filtering (SVD)** — matrix factorization via `scikit-surprise`, evaluated with both RMSE (rating accuracy) and Precision@K/Recall@K (ranking quality).
4. **Content-Based + Hybrid** — genre multi-hot vectors, cosine similarity, rating-weighted user taste profiles, and a tuned weighted blend of SVD and content scores.
5. **Deployment** — trained artifacts persisted to disk (no retraining per request); served via a FastAPI endpoint.

Every stage uses the same leakage-safe evaluation: for each user, the last 20% of their ratings (by timestamp) are held out for testing, and the model is trained only on what came before — simulating a real deployment scenario where you predict forward in time from what you currently know.

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd movie-recommender

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Download the MovieLens `ml-latest-small` dataset from [grouplens.org](https://grouplens.org/datasets/movielens/) and place the extracted folder at `data/raw/ml-latest-small/` (so `ratings.csv` and `movies.csv` live at `data/raw/ml-latest-small/ratings.csv` etc., matching the path used in `src/data_loader.py`).

## Running the full pipeline

The original end-to-end script (EDA → popularity baseline → SVD training + diagnostics → content-based → hybrid tuning, with all print output) has been split into `src/` modules by responsibility (`data_loader.py`, `baseline.py`, `collaborative.py`, `content.py`, `hybrid.py`, `evaluate.py`). To retrain everything and regenerate the deployable artifacts in one pass:

```bash
python scripts/train_and_save.py
```

For the full diagnostic walkthrough (cold-start checks, rating-ceiling analysis, hyperparameter experiments, weight-tuning sweep) with all intermediate output, see the individual `src/` modules or the original investigation logged in [`movie-recommender-roadmap.md`](./movie-recommender-roadmap.md).

## Running the API

**1. Train and persist the models** (only needs to be run once, or whenever the underlying data changes):

```bash
python scripts/train_and_save.py
```

This trains the popularity baseline and SVD model, builds the genre similarity data, and saves everything needed for inference to `models/`.

**2. (Optional) Verify the saved artifacts load correctly in a fresh process:**

```bash
python scripts/test_load.py
```

**3. Start the API server:**

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) are auto-generated by FastAPI at `http://127.0.0.1:8000/docs`.

### Example request

```bash
curl "http://127.0.0.1:8000/recommend/1?k=10"
```

### Example response

```json
{
  "user_id": 1,
  "recommendations": [
    { "movieId": 2019, "title": "Seven Samurai (Shichinin no samurai) (1954)", "hybrid_score": 0.9647 },
    { "movieId": 3000, "title": "Princess Mononoke (Mononoke-hime) (1997)", "hybrid_score": 0.9481 },
    { "movieId": 6016, "title": "City of God (Cidade de Deus) (2002)", "hybrid_score": 0.9445 },
    { "movieId": 7153, "title": "Lord of the Rings: The Return of the King, The (2003)", "hybrid_score": 0.9368 },
    { "movieId": 778, "title": "Trainspotting (1996)", "hybrid_score": 0.9318 }
  ]
}
```

*(Verified output from a live local run — response truncated to 5 of the requested 10 for brevity.)*

Requesting a `user_id` with no ratings in the training data returns a `404`. Requesting an invalid `k` (e.g. zero or negative) returns a `422`.

## Live demo

<!-- TODO: add live Render/Railway URL here once deployed, e.g.:
**API:** https://movie-recommender-xxxx.onrender.com/recommend/1?k=10
**Interactive docs:** https://movie-recommender-xxxx.onrender.com/docs

Note: hosted on a free tier — the first request after a period of inactivity may take 30-60 seconds to respond while the service wakes up.
-->

## Running tests

```bash
python -m pytest -q tests/test_evaluate.py
```

*(Use `python -m pytest` rather than `pytest` directly if you hit a `ModuleNotFoundError: No module named 'src'` — this ensures the project root is on the Python path during test collection.)*

## A note on scale

The API currently recomputes SVD and content-based scores for all ~9,700 unrated movies live, on every request — O(movies) per call. Measured response time is **~75.5ms per request**, which is fine at this dataset's scale, but this approach wouldn't hold up against a catalog with millions of items or high request volume. At that scale, the standard fixes are offline precomputation with caching (recompute recommendations for all users on a schedule, serve from a lookup) or an approximate nearest-neighbor index for fast retrieval at serve time.

## Tech stack

- **Language:** Python
- **Data / ML:** pandas, scikit-learn, scikit-surprise
- **API:** FastAPI, Pydantic, uvicorn
- **Dataset:** [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/)