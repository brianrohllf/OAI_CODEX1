from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["league_id", "season", "week", "winner_count", "rank_score"]
TARGET = "bid"


@dataclass
class BidModel:
    pipeline: Pipeline

    @classmethod
    def train(cls, frame: pd.DataFrame) -> "BidModel":
        if frame.empty:
            raise ValueError("No bid records available for model training")

        x = frame[FEATURES]
        y = frame[TARGET]

        transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["league_id"]),
                ("num", "passthrough", ["season", "week", "winner_count", "rank_score"]),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("features", transformer),
                ("model", RandomForestRegressor(n_estimators=300, random_state=7)),
            ]
        )
        pipeline.fit(x, y)
        return cls(pipeline=pipeline)

    def predict(self, payload: pd.DataFrame) -> pd.Series:
        return pd.Series(self.pipeline.predict(payload[FEATURES]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "BidModel":
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)
