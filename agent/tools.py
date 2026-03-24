from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rag.retriever import get_retriever

MOCK_DATA_PATH = Path(__file__).parent.parent / "data" / "mock_garmin.json"

INTENSITY_MAP = {
    "easy": "easy",
    "recovery": "easy",
    "long": "moderate",
    "tempo": "moderate",
    "interval": "hard",
}


class RunRecord(TypedDict):
    date: str
    distance_km: float
    duration_minutes: int
    avg_pace_per_km: str
    avg_hr: int
    max_hr: int
    elevation_gain_m: int
    type: str


class WeeklySummary(TypedDict):
    this_week_km: float
    last_week_km: float
    four_week_avg_km: float


class MockGarminData(TypedDict):
    user_id: str
    recent_runs: list[RunRecord]
    weekly_summary: WeeklySummary
    personal_bests: dict[str, float]
    injury_history: list[str]


class RecentRunsPayload(TypedDict):
    user_id: str
    runs: list[RunRecord]
    days: int


class TrainingLoadPayload(TypedDict):
    user_id: str
    period_days: int
    total_km: float
    this_week_km: float
    last_week_km: float
    week_over_week_change_pct: float
    intensity_distribution_pct: dict[str, float]
    injury_risk: str


class RaceHistoryPayload(TypedDict):
    user_id: str
    personal_bests: dict[str, float]
    injury_history: list[str]


def _load_mock_data() -> MockGarminData:
    """Read mock Garmin data so tools can return deterministic fallback results."""
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    return raw_data


def _parse_iso_date(date_text: str) -> datetime:
    """Parse an ISO date string so date filtering remains explicit and testable."""
    return datetime.strptime(date_text, "%Y-%m-%d")


def _filter_runs_by_days(runs: list[RunRecord], days: int) -> list[RunRecord]:
    """Prefer date-window filtering but degrade to recent tail entries for stale datasets."""
    safe_days = max(days, 1)
    cutoff = datetime.now() - timedelta(days=safe_days)
    recent = [run for run in runs if _parse_iso_date(run["date"]) >= cutoff]
    if recent:
        return recent
    # NOTE: Mock data may not match today's date, so keep behavior stable with tail slicing.
    return runs[-safe_days:]


def _split_runs_by_week(
    runs: list[RunRecord],
) -> tuple[list[RunRecord], list[RunRecord]]:
    """Split runs into this-week (last 7 days) and last-week (7-14 days ago) buckets.

    Capping last_week at the 7-14 day window prevents older runs from inflating the
    'last week' total when the caller requests a 21- or 30-day window.
    """
    now = datetime.now()
    week_start = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)
    this_week_runs = [run for run in runs if _parse_iso_date(run["date"]) >= week_start]
    last_week_runs = [
        run for run in runs
        if two_weeks_ago <= _parse_iso_date(run["date"]) < week_start
    ]
    return this_week_runs, last_week_runs


def _calculate_change_pct(this_week_km: float, last_week_km: float) -> float:
    """Compute week-over-week change while avoiding division errors at low volume."""
    if last_week_km <= 0:
        return 0.0
    return (this_week_km - last_week_km) / last_week_km * 100


def _calculate_intensity_distribution(runs: list[RunRecord]) -> dict[str, float]:
    """Group mileage by intensity to keep coaching feedback grounded in workload mix."""
    total_km = sum(run["distance_km"] for run in runs)
    if total_km <= 0:
        return {}

    intensity_km: dict[str, float] = {}
    for run in runs:
        bucket = INTENSITY_MAP.get(run.get("type", "easy"), "easy")
        intensity_km[bucket] = intensity_km.get(bucket, 0.0) + run["distance_km"]
    return {
        key: round(value / total_km * 100, 1) for key, value in intensity_km.items()
    }


# ---------------------------------------------------------------------------
# Tool 1 - get_recent_runs
# ---------------------------------------------------------------------------


class GetRecentRunsInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    days: int = Field(
        default=7, description="Number of recent days to query, default 7"
    )


@tool(args_schema=GetRecentRunsInput)
def get_recent_runs(user_id: str, days: int = 7) -> str:
    """Return raw per-session run records (distance, pace, heart rate) for the last N days.

    Call this when the user asks to list or review individual runs, e.g. 'show my last 5 runs'
    or 'what did I run on Monday' -- not for trend analysis or injury risk assessment.
    """
    try:
        data = _load_mock_data()
        recent = _filter_runs_by_days(data.get("recent_runs", []), days)
        payload: RecentRunsPayload = {
            "user_id": user_id,
            "runs": recent,
            "days": max(days, 1),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        print(f"[tools] get_recent_runs error: {e}")
        return json.dumps({"user_id": user_id, "runs": [], "days": days})


# ---------------------------------------------------------------------------
# Tool 2 - get_training_load
# ---------------------------------------------------------------------------


class GetTrainingLoadInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    days: int = Field(
        default=14, description="Number of recent days to analyze, default 14"
    )


@tool(args_schema=GetTrainingLoadInput)
def get_training_load(user_id: str, days: int = 14) -> str:
    """Compute aggregated training load: total km, week-over-week mileage change, intensity mix, and injury risk score.

    Call this when the user asks about training volume trends, whether mileage is increasing too fast,
    overall training load, or injury risk -- not for listing individual run sessions.
    """
    try:
        data = _load_mock_data()
        runs = _filter_runs_by_days(data.get("recent_runs", []), days)
        total_km = sum(r["distance_km"] for r in runs)
        this_week_runs, last_week_runs = _split_runs_by_week(runs)

        # Fall back to weekly_summary when date-based split yields empty halves
        if not this_week_runs or not last_week_runs:
            summary = data.get("weekly_summary", {})
            this_week_km = float(summary.get("this_week_km", 0.0))
            last_week_km = float(summary.get("last_week_km", 0.0))
        else:
            this_week_km = sum(run["distance_km"] for run in this_week_runs)
            last_week_km = sum(run["distance_km"] for run in last_week_runs)

        change_pct = _calculate_change_pct(this_week_km, last_week_km)
        distribution = _calculate_intensity_distribution(runs)
        injury_risk = (
            "HIGH - weekly mileage increase exceeds 10%" if change_pct > 10 else "LOW"
        )

        result: TrainingLoadPayload = {
            "user_id": user_id,
            "period_days": max(days, 1),
            "total_km": round(total_km, 1),
            "this_week_km": round(this_week_km, 1),
            "last_week_km": round(last_week_km, 1),
            "week_over_week_change_pct": round(change_pct, 1),
            "intensity_distribution_pct": distribution,
            "injury_risk": injury_risk,
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        print(f"[tools] get_training_load error: {e}")
        return json.dumps({"user_id": user_id, "total_km": 0.0, "period_days": days})


# ---------------------------------------------------------------------------
# Tool 3 - get_race_history
# ---------------------------------------------------------------------------


class GetRaceHistoryInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")


@tool(args_schema=GetRaceHistoryInput)
def get_race_history(user_id: str) -> str:
    """Return personal bests and injury history so the agent can calibrate goals and constraints.

    Call this when the user asks about race goals, race preparation, or how past performance/injuries affect planning.
    """
    try:
        data = _load_mock_data()
        result: RaceHistoryPayload = {
            "user_id": user_id,
            "personal_bests": data.get("personal_bests", {}),
            "injury_history": data.get("injury_history", []),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        print(f"[tools] get_race_history error: {e}")
        return json.dumps({"user_id": user_id, "personal_bests": {}, "injury_history": []})


# ---------------------------------------------------------------------------
# Tool 4 - search_knowledge (hybrid vector + BM25 via pgvector)
# ---------------------------------------------------------------------------


class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="Search query for the running knowledge base")
    top_k: int = Field(default=3, description="Number of results to return, default 3")


@tool(args_schema=SearchKnowledgeInput)
def search_knowledge(query: str, top_k: int = 3) -> str:
    """Search the running knowledge base using hybrid vector + BM25 retrieval and return ranked chunks with source.

    Call this when the user asks for coaching advice grounded in running science:
    training principles, pace zones, injury prevention, or race preparation.
    """
    try:
        results = get_retriever().search(query, top_k=top_k)
        if not results:
            return json.dumps([])
        payload = [
            {"source": r["source"], "content": r["content"], "score": r["score"]}
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        print(f"[tools] search_knowledge error: {e}")
        return json.dumps([])


ALL_TOOLS = [get_recent_runs, get_training_load, get_race_history, search_knowledge]
