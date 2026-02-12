from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from fab_optimizer.client import SleeperClient
from fab_optimizer.data import (
    SleeperCollector,
    extract_bid_dataframe,
    load_json,
    merge_rankings,
    save_json,
)
from fab_optimizer.model import BidModel
from fab_optimizer.web import create_app

app = typer.Typer(help="Sleeper FAAB optimizer")


@app.command()
def collect(
    username: str = typer.Option(..., help="Sleeper username"),
    season: int = typer.Option(..., help="League season year"),
    sport: str = typer.Option("nfl", help="Sleeper sport"),
    weeks: int = typer.Option(18, help="How many transaction weeks to query"),
    out: Path = typer.Option(Path("data/history.json"), help="Output JSON path"),
) -> None:
    collector = SleeperCollector(client=SleeperClient())
    payload = collector.collect_user_history(username=username, season=season, sport=sport, regular_season_weeks=weeks)
    save_json(out, payload)
    typer.echo(f"Saved history to {out}")


@app.command()
def train(
    history: Path = typer.Option(Path("data/history.json"), help="Collected history JSON"),
    rankings: Path = typer.Option(Path("data/rankings.csv"), help="Optional rankings CSV"),
    model_out: Path = typer.Option(Path("models/fab_bid_model.joblib"), help="Model output path"),
    bids_out: Path = typer.Option(Path("data/bids.csv"), help="Normalized bids dataset"),
) -> None:
    payload = load_json(history)
    bids = extract_bid_dataframe(payload)
    merged = merge_rankings(bids, rankings if rankings.exists() else None)
    merged.to_csv(bids_out, index=False)

    model = BidModel.train(merged)
    model.save(model_out)

    typer.echo(f"Saved normalized bids to {bids_out}")
    typer.echo(f"Saved trained model to {model_out}")


@app.command()
def recommend(
    model_path: Path = typer.Option(Path("models/fab_bid_model.joblib"), help="Trained model path"),
    league_id: str = typer.Option(...),
    season: int = typer.Option(...),
    week: int = typer.Option(...),
    winner_count: int = typer.Option(1, help="Expected number of competing bidders"),
    rank_score: float = typer.Option(0.0, help="Player quality score from rankings"),
) -> None:
    model = BidModel.load(model_path)
    row = pd.DataFrame(
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
    bid = float(model.predict(row).iloc[0])
    typer.echo(f"Recommended bid: ${bid:.2f}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host for local web UI"),
    port: int = typer.Option(5000, help="Port for local web UI"),
) -> None:
    app = create_app()
    typer.echo(f"Starting web UI on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    app()
