# Sleeper FAAB Optimizer

This project is a starter application for building a **Sleeper-powered FAAB bidding assistant**.
It pulls your historical league data, extracts waiver/FAB bids, joins player quality signals (rankings), and trains a model to recommend bid amounts.

## What it does

- Collects Sleeper user + league + transaction history.
- Normalizes waiver transactions into a bid dataset.
- Merges optional player ranking inputs from outside ranking sites.
- Trains a regression model to estimate likely winning bids.
- Generates recommendation values for future waiver decisions.
- Provides a local HTML interface for training and recommendations.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI workflow

### 1) Collect your Sleeper data

```bash
fab-opt collect --username YOUR_SLEEPER_USERNAME --season 2024 --out data/history.json
```

### 2) (Optional) Add rankings data

Create `data/rankings.csv` with at least:

```csv
player_id,rank
1234,1
5678,2
```

You can also provide `rank_score` directly:

```csv
player_id,rank_score
1234,0.95
5678,0.75
```

### 3) Train bid model

```bash
fab-opt train --history data/history.json --rankings data/rankings.csv --model-out models/fab_bid_model.joblib
```

### 4) Get a recommendation

```bash
fab-opt recommend \
  --model-path models/fab_bid_model.joblib \
  --league-id 1234567890 \
  --season 2025 \
  --week 1 \
  --winner-count 3 \
  --rank-score 0.85
```

## Local HTML interface

Run the local web app:

```bash
fab-opt serve --host 127.0.0.1 --port 5000
```

Then open:

- `http://127.0.0.1:5000`

The UI includes:

- **Train Model** form for `history.json`, optional rankings CSV, and output model path.
- **Recommend Bid** form using league context + rank score.

## Notes for your full product roadmap

To match your long-term goal (analyzing **all data** and predicting future bidding behavior), next steps should include:

1. Multi-season ingestion and feature store (manager budget habits, roster depth, injury context).
2. Opponent-level bidding profiles by manager and league settings.
3. Confidence intervals around bid recommendations.
4. Rank aggregation from multiple external sources (expert ranks, ADP, rest-of-season projections).
5. Objective function for expected value (win probability vs. budget preservation).
6. Pipeline for annual carry-forward training to model next-year behavior shifts.
