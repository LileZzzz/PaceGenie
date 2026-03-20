from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MOCK_DATA_PATH = Path(__file__).parent.parent / "data" / "mock_garmin.json"

INTENSITY_MAP = {
    "easy": "easy",
    "recovery": "easy",
    "long": "moderate",
    "tempo": "moderate",
    "interval": "hard",
}


def _load_mock_data() -> dict:
    """Load the local mock Garmin JSON as a fallback data source."""
    with open(MOCK_DATA_PATH, "r") as f:
        return json.load(f)


def _filter_runs_by_days(runs: list[dict], days: int) -> list[dict]:
    """Return runs within the last *days*; fall back to the tail entries if dates are stale."""
    cutoff = datetime.now() - timedelta(days=days)
    recent = [
        r for r in runs
        if datetime.strptime(r["date"], "%Y-%m-%d") >= cutoff
    ]
    if not recent:
        recent = runs[-days:]
    return recent


# ---------------------------------------------------------------------------
# Tool 1 - get_recent_runs
# ---------------------------------------------------------------------------

class GetRecentRunsInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    days: int = Field(default=7, description="Number of recent days to query, default 7")


@tool(args_schema=GetRecentRunsInput)
def get_recent_runs(user_id: str, days: int = 7) -> str:
    """Fetch the user's running records for the last N days, including distance, pace, and heart rate."""
    try:
        data = _load_mock_data()
        recent = _filter_runs_by_days(data.get("recent_runs", []), days)
        return json.dumps(
            {"user_id": user_id, "runs": recent, "days": days},
            ensure_ascii=False,
        )
    except Exception:
        data = _load_mock_data()
        return json.dumps(
            {"user_id": user_id, "runs": data.get("recent_runs", [])},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool 2 - get_training_load
# ---------------------------------------------------------------------------

class GetTrainingLoadInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    days: int = Field(default=14, description="Number of recent days to analyze, default 14")


@tool(args_schema=GetTrainingLoadInput)
def get_training_load(user_id: str, days: int = 14) -> str:
    """Analyze training load for the last N days: total mileage, week-over-week change,
    intensity distribution (easy/moderate/hard), and injury risk assessment."""
    try:
        data = _load_mock_data()
        runs = _filter_runs_by_days(data.get("recent_runs", []), days)

        total_km = sum(r["distance_km"] for r in runs)

        week_cutoff = datetime.now() - timedelta(days=7)
        this_week = [
            r for r in runs
            if datetime.strptime(r["date"], "%Y-%m-%d") >= week_cutoff
        ]
        last_week = [
            r for r in runs
            if datetime.strptime(r["date"], "%Y-%m-%d") < week_cutoff
        ]

        # Fall back to weekly_summary when date-based split yields empty halves
        if not this_week or not last_week:
            summary = data.get("weekly_summary", {})
            this_week_km = summary.get("this_week_km", 0)
            last_week_km = summary.get("last_week_km", 0)
        else:
            this_week_km = sum(r["distance_km"] for r in this_week)
            last_week_km = sum(r["distance_km"] for r in last_week)

        change_pct = (
            (this_week_km - last_week_km) / last_week_km * 100
            if last_week_km > 0
            else 0.0
        )

        intensity_km: dict[str, float] = {}
        for r in runs:
            bucket = INTENSITY_MAP.get(r.get("type", "easy"), "easy")
            intensity_km[bucket] = intensity_km.get(bucket, 0) + r["distance_km"]

        distribution = (
            {k: round(v / total_km * 100, 1) for k, v in intensity_km.items()}
            if total_km > 0
            else {}
        )

        injury_risk = (
            "HIGH - weekly mileage increase exceeds 10%"
            if change_pct > 10
            else "LOW"
        )

        result = {
            "user_id": user_id,
            "period_days": days,
            "total_km": round(total_km, 1),
            "this_week_km": round(this_week_km, 1),
            "last_week_km": round(last_week_km, 1),
            "week_over_week_change_pct": round(change_pct, 1),
            "intensity_distribution_pct": distribution,
            "injury_risk": injury_risk,
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        data = _load_mock_data()
        summary = data.get("weekly_summary", {})
        return json.dumps(
            {
                "user_id": user_id,
                "this_week_km": summary.get("this_week_km"),
                "last_week_km": summary.get("last_week_km"),
                "4_week_avg_km": summary.get("4_week_avg_km"),
                "note": "Returned cached weekly summary due to data retrieval error",
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool 3 - get_race_history
# ---------------------------------------------------------------------------

class GetRaceHistoryInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")


@tool(args_schema=GetRaceHistoryInput)
def get_race_history(user_id: str) -> str:
    """Retrieve the user's personal bests and recent race records."""
    try:
        data = _load_mock_data()
        result = {
            "user_id": user_id,
            "personal_bests": data.get("personal_bests", {}),
            "injury_history": data.get("injury_history", []),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return json.dumps(
            {
                "user_id": user_id,
                "personal_bests": {},
                "note": "Race history unavailable due to data retrieval error",
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool 4 - search_knowledge (placeholder until Day 5-6 hybrid RAG)
# ---------------------------------------------------------------------------

class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="Search query for the running knowledge base")


@tool(args_schema=SearchKnowledgeInput)
def search_knowledge(query: str) -> str:
    """Search the running knowledge base for training principles, pace zones, injury prevention, and race preparation."""
    return "Knowledge base search is under development. Please check back later."


ALL_TOOLS = [get_recent_runs, get_training_load, get_race_history, search_knowledge]
