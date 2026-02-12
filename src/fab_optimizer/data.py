from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from fab_optimizer.client import SleeperClient


@dataclass
class LeagueSnapshot:
    league: dict[str, Any]
    users: list[dict[str, Any]]
    rosters: list[dict[str, Any]]
    transactions: list[dict[str, Any]]


@dataclass
class SleeperCollector:
    client: SleeperClient

    def collect_user_history(
        self,
        username: str,
        season: int,
        sport: str = "nfl",
        regular_season_weeks: int = 18,
    ) -> dict[str, Any]:
        user = self.client.get_user(username)
        leagues = self.client.get_user_leagues(user["user_id"], sport=sport, season=season)
        snapshots: list[dict[str, Any]] = []

        for league in leagues:
            league_id = str(league["league_id"])
            users = self.client.get_league_users(league_id)
            rosters = self.client.get_league_rosters(league_id)

            transactions: list[dict[str, Any]] = []
            for week in range(1, regular_season_weeks + 1):
                transactions.extend(self.client.get_league_transactions(league_id, week=week))

            snapshots.append(
                {
                    "league": league,
                    "users": users,
                    "rosters": rosters,
                    "transactions": transactions,
                }
            )

        return {
            "user": user,
            "season": season,
            "sport": sport,
            "league_snapshots": snapshots,
        }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_bid_dataframe(history: dict[str, Any]) -> pd.DataFrame:
    """
    Normalize historical waiver/FAB transactions into a model-ready dataframe.
    """
    rows: list[dict[str, Any]] = []

    for snapshot in history.get("league_snapshots", []):
        league = snapshot["league"]
        league_id = str(league["league_id"])

        for tx in snapshot.get("transactions", []):
            if tx.get("type") != "waiver":
                continue

            settings = tx.get("settings", {})
            status_updated = tx.get("status_updated")
            week = tx.get("leg") or tx.get("week")
            bids = tx.get("consenter_ids") or []

            # Sleeper payload formats vary by season/league settings.
            bid_value = settings.get("waiver_bid")
            if bid_value is None:
                bid_value = settings.get("bid")
            if bid_value is None:
                bid_value = settings.get("amount")

            if bid_value is None:
                continue

            add_map = tx.get("adds") or {}
            player_id = next(iter(add_map.keys()), None)

            rows.append(
                {
                    "league_id": league_id,
                    "season": league.get("season"),
                    "scoring_type": league.get("scoring_settings", {}).get("rec"),
                    "draft_slot": tx.get("draft_slot"),
                    "week": week,
                    "bid": float(bid_value),
                    "status_updated": status_updated,
                    "winner_count": len(bids),
                    "player_id": player_id,
                    "roster_id": tx.get("roster_id"),
                }
            )

    return pd.DataFrame(rows)


def merge_rankings(bids: pd.DataFrame, rankings_path: Path | None = None) -> pd.DataFrame:
    if rankings_path is None or not rankings_path.exists() or bids.empty:
        bids["rank_score"] = 0.0
        return bids

    rankings = pd.read_csv(rankings_path)
    if "player_id" not in rankings.columns:
        raise ValueError("rankings CSV must include player_id")

    if "rank_score" not in rankings.columns:
        # Convert rank -> score where low rank is better.
        if "rank" not in rankings.columns:
            raise ValueError("rankings CSV must include either rank_score or rank")
        rankings = rankings.assign(rank_score=1 / rankings["rank"].clip(lower=1))

    merged = bids.merge(rankings[["player_id", "rank_score"]], on="player_id", how="left")
    merged["rank_score"] = merged["rank_score"].fillna(0.0)
    return merged
