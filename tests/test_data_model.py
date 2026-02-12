from pathlib import Path

import pandas as pd

from fab_optimizer.data import extract_bid_dataframe, merge_rankings
from fab_optimizer.model import BidModel


def test_extract_bid_dataframe_handles_waiver_bids() -> None:
    payload = {
        "league_snapshots": [
            {
                "league": {"league_id": "123", "season": "2024", "scoring_settings": {"rec": 1}},
                "transactions": [
                    {
                        "type": "waiver",
                        "settings": {"waiver_bid": 12},
                        "week": 2,
                        "consenter_ids": ["1", "2"],
                        "adds": {"player_1": "3"},
                        "roster_id": 3,
                    }
                ],
            }
        ]
    }

    frame = extract_bid_dataframe(payload)

    assert len(frame) == 1
    assert frame.iloc[0]["bid"] == 12.0
    assert frame.iloc[0]["winner_count"] == 2


def test_model_train_predict_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"league_id": "1", "season": 2024, "week": 1, "winner_count": 2, "rank_score": 0.9, "bid": 20.0},
            {"league_id": "1", "season": 2024, "week": 5, "winner_count": 3, "rank_score": 0.4, "bid": 11.0},
            {"league_id": "2", "season": 2023, "week": 3, "winner_count": 1, "rank_score": 0.8, "bid": 19.0},
            {"league_id": "2", "season": 2023, "week": 7, "winner_count": 2, "rank_score": 0.1, "bid": 7.0},
        ]
    )

    model = BidModel.train(frame)
    pred = model.predict(frame)
    assert len(pred) == len(frame)

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = BidModel.load(path)
    pred2 = loaded.predict(frame)
    assert len(pred2) == len(frame)


def test_merge_rankings_adds_rank_score(tmp_path: Path) -> None:
    bids = pd.DataFrame([{"player_id": "p1", "league_id": "1", "season": 2024, "week": 1, "winner_count": 1, "bid": 10.0}])
    rankings = tmp_path / "rankings.csv"
    rankings.write_text("player_id,rank\np1,2\n", encoding="utf-8")

    merged = merge_rankings(bids, rankings)
    assert "rank_score" in merged.columns
    assert merged.iloc[0]["rank_score"] > 0
