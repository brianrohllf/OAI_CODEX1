from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from fab_optimizer.data import extract_bid_dataframe, load_json, merge_rankings
from fab_optimizer.model import BidModel


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/train")
    def train() -> str:
        history_path = Path(request.form.get("history_path", "data/history.json"))
        rankings_path_raw = request.form.get("rankings_path", "data/rankings.csv")
        rankings_path = Path(rankings_path_raw)
        model_out = Path(request.form.get("model_path", "models/fab_bid_model.joblib"))

        if not history_path.exists():
            return render_template("index.html", train_error=f"History file not found: {history_path}")

        payload = load_json(history_path)
        bids = extract_bid_dataframe(payload)
        merged = merge_rankings(bids, rankings_path if rankings_path.exists() else None)

        if merged.empty:
            return render_template(
                "index.html",
                train_error="No waiver bids found in history. Cannot train model.",
            )

        model = BidModel.train(merged)
        model.save(model_out)

        return render_template(
            "index.html",
            train_success=f"Model trained and saved to {model_out} ({len(merged)} bid rows).",
        )

    @app.post("/recommend")
    def recommend() -> str:
        model_path = Path(request.form.get("model_path", "models/fab_bid_model.joblib"))

        if not model_path.exists():
            return render_template("index.html", recommend_error=f"Model file not found: {model_path}")

        league_id = request.form.get("league_id", "")
        season = int(request.form.get("season", "2025"))
        week = int(request.form.get("week", "1"))
        winner_count = int(request.form.get("winner_count", "1"))
        rank_score = float(request.form.get("rank_score", "0.0"))

        frame = pd.DataFrame(
            [
                {
                    "league_id": league_id,
                    "season": season,
                    "week": week,
                    "winner_count": winner_count,
                    "rank_score": rank_score,
                }
            ]
        )

        model = BidModel.load(model_path)
        bid = float(model.predict(frame).iloc[0])

        return render_template("index.html", recommendation=f"Recommended bid: ${bid:.2f}")

    return app


app = create_app()
