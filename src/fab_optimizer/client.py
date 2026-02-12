from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://api.sleeper.app/v1"


@dataclass
class SleeperClient:
    """Thin client for the Sleeper public API."""

    timeout: int = 30

    def _get(self, path: str) -> Any:
        url = f"{BASE_URL}{path}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_user(self, username: str) -> dict[str, Any]:
        return self._get(f"/user/{username}")

    def get_user_leagues(self, user_id: str, sport: str, season: int) -> list[dict[str, Any]]:
        return self._get(f"/user/{user_id}/leagues/{sport}/{season}")

    def get_league_users(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f"/league/{league_id}/users")

    def get_league_rosters(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f"/league/{league_id}/rosters")

    def get_league_transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self._get(f"/league/{league_id}/transactions/{week}")

    def get_nfl_players(self) -> dict[str, Any]:
        return self._get("/players/nfl")
