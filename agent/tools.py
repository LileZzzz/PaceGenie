from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)

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


_mock_data_cache: MockGarminData | None = None


def _load_mock_data() -> MockGarminData:
    """Return cached mock Garmin data, loading from disk only on first call."""
    global _mock_data_cache
    if _mock_data_cache is None:
        with open(MOCK_DATA_PATH, "r", encoding="utf-8") as file:
            _mock_data_cache = json.load(file)
    return _mock_data_cache


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
        logger.exception("[tools] get_recent_runs error: %s", e)
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
        logger.exception("[tools] get_training_load error: %s", e)
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
        logger.exception("[tools] get_race_history error: %s", e)
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
        logger.exception("[tools] search_knowledge error: %s", e)
        return json.dumps([])


# ---------------------------------------------------------------------------
# Tool 5 - get_weekly_trend
# ---------------------------------------------------------------------------


class GetWeeklyTrendInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    weeks: int = Field(default=8, ge=1, description="Number of past weeks to analyze, default 8")


class WeeklyTrendEntry(TypedDict):
    week_label: str
    start_date: str
    total_km: float
    runs: int
    avg_pace_per_km: str
    avg_hr: int


class WeeklyTrendPayload(TypedDict):
    user_id: str
    weeks_analyzed: int
    trend: list[WeeklyTrendEntry]
    overall_direction: str


def _pace_str_to_seconds(pace: str) -> float:
    """Convert 'M:SS' pace string to total seconds for averaging."""
    try:
        parts = pace.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 360.0  # default to 6:00/km if unparseable


def _seconds_to_pace_str(seconds: float) -> str:
    """Convert total seconds back to 'M:SS' pace string."""
    total = round(seconds)
    m = total // 60
    s = total % 60
    return f"{m}:{s:02d}"


@tool(args_schema=GetWeeklyTrendInput)
def get_weekly_trend(user_id: str, weeks: int = 8) -> str:
    """Return week-by-week training volume trend over the last N weeks.

    Call this when the user asks about long-term trends: is mileage increasing or
    decreasing over time, how training has evolved over the past months, or whether
    they are building fitness progressively.
    """
    try:
        data = _load_mock_data()
        all_runs: list[RunRecord] = data.get("recent_runs", [])
        now = datetime.now()
        # Pre-parse dates once to avoid O(weeks × runs) strptime calls in the loop.
        parsed_runs = [(r, _parse_iso_date(r["date"])) for r in all_runs]

        trend: list[WeeklyTrendEntry] = []
        for i in range(weeks, 0, -1):
            week_end = now - timedelta(days=(i - 1) * 7)
            week_start = week_end - timedelta(days=7)
            week_runs = [
                r for r, d in parsed_runs
                if week_start <= d < week_end
            ]
            if week_runs:
                total_km = round(sum(r["distance_km"] for r in week_runs), 1)
                avg_pace_secs = sum(
                    _pace_str_to_seconds(r["avg_pace_per_km"]) for r in week_runs
                ) / len(week_runs)
                avg_hr = round(sum(r["avg_hr"] for r in week_runs) / len(week_runs))
            else:
                total_km = 0.0
                avg_pace_secs = 0.0
                avg_hr = 0

            trend.append(WeeklyTrendEntry(
                week_label=f"W-{i}" if i > 1 else "This week",
                start_date=week_start.strftime("%Y-%m-%d"),
                total_km=total_km,
                runs=len(week_runs),
                avg_pace_per_km=_seconds_to_pace_str(avg_pace_secs) if avg_pace_secs > 0 else "-",
                avg_hr=avg_hr,
            ))

        # Determine overall direction from first to last non-zero week
        non_zero = [w for w in trend if w["total_km"] > 0]
        if len(non_zero) >= 2:
            first_km = non_zero[0]["total_km"]
            last_km = non_zero[-1]["total_km"]
            change_pct = (last_km - first_km) / first_km * 100
            if change_pct > 5:
                direction = f"INCREASING (+{round(change_pct, 1)}% over {len(non_zero)} weeks)"
            elif change_pct < -5:
                direction = f"DECREASING ({round(change_pct, 1)}% over {len(non_zero)} weeks)"
            else:
                direction = "STABLE (less than 5% change)"
        else:
            direction = "INSUFFICIENT DATA"

        payload = WeeklyTrendPayload(
            user_id=user_id,
            weeks_analyzed=weeks,
            trend=trend,
            overall_direction=direction,
        )
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        logger.exception("[tools] get_weekly_trend error: %s", e)
        return json.dumps({"user_id": user_id, "trend": [], "weeks_analyzed": weeks})


# ---------------------------------------------------------------------------
# Tool 6 - get_pace_prediction
# ---------------------------------------------------------------------------


class GetPacePredictionInput(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    target_distance_km: float = Field(
        gt=0,
        description="Target race distance in km (e.g. 5.0, 10.0, 21.1, 42.2)",
    )


class PacePredictionPayload(TypedDict):
    user_id: str
    target_distance_km: float
    predicted_finish_time: str
    predicted_pace_per_km: str
    based_on: str
    confidence: str


def _riegel_predict(known_time_min: float, known_dist_km: float, target_dist_km: float) -> float:
    """Riegel's formula: T2 = T1 * (D2/D1)^1.06 — the standard race time predictor."""
    return known_time_min * ((target_dist_km / known_dist_km) ** 1.06)


def _compute_confidence(ref_dist: float, target_dist: float) -> str:
    """High confidence when reference and target distances are within 150% of each other."""
    if abs(ref_dist - target_dist) / target_dist < 1.5:
        return "HIGH"
    return "MEDIUM (reference distance differs significantly)"


def _minutes_to_time_str(minutes: float) -> str:
    """Convert decimal minutes to H:MM:SS or M:SS string."""
    total_seconds = int(round(minutes * 60))
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


@tool(args_schema=GetPacePredictionInput)
def get_pace_prediction(user_id: str, target_distance_km: float) -> str:
    """Predict finish time and pace for a target race distance using the user's actual training data.

    Uses Riegel's formula calibrated against the user's best recent performance.
    Call this when the user asks 'can I run a sub-X time?', 'what pace should I target?',
    or 'what is my predicted time for a half marathon / 10K / 5K?'.
    """
    try:
        data = _load_mock_data()
        pbs = data.get("personal_bests", {})

        reference_points: list[tuple[float, float]] = []
        pb_map = {
            5.0: pbs.get("5k_minutes"),
            10.0: pbs.get("10k_minutes"),
            21.1: pbs.get("half_marathon_minutes"),
        }
        for dist, time_min in pb_map.items():
            if time_min:
                reference_points.append((dist, float(time_min)))

        # Also derive a reference from recent tempo/interval runs if no PB available
        if not reference_points:
            all_runs: list[RunRecord] = data.get("recent_runs", [])
            quality_runs = [r for r in all_runs if r.get("type") in ("tempo", "interval")]
            if quality_runs:
                best_pace_str = min(
                    quality_runs, key=lambda r: _pace_str_to_seconds(r["avg_pace_per_km"])
                )["avg_pace_per_km"]
                est_5k_time = _pace_str_to_seconds(best_pace_str) / 60 * 5
                reference_points.append((5.0, est_5k_time))

        if not reference_points:
            return json.dumps({
                "user_id": user_id,
                "error": "No race history or quality runs available to base prediction on.",
            })

        # Use the reference point closest to target distance for most accurate prediction
        ref_dist, ref_time = min(
            reference_points, key=lambda p: abs(p[0] - target_distance_km)
        )

        predicted_min = _riegel_predict(ref_time, ref_dist, target_distance_km)
        predicted_pace_secs = (predicted_min / target_distance_km) * 60

        payload = PacePredictionPayload(
            user_id=user_id,
            target_distance_km=target_distance_km,
            predicted_finish_time=_minutes_to_time_str(predicted_min),
            predicted_pace_per_km=_seconds_to_pace_str(predicted_pace_secs),
            based_on=f"Personal best: {ref_dist}km in {_minutes_to_time_str(ref_time)}",
            confidence=_compute_confidence(ref_dist, target_distance_km),
        )
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        logger.exception("[tools] get_pace_prediction error: %s", e)
        return json.dumps({"user_id": user_id, "error": str(e)})


ALL_TOOLS = [
    get_recent_runs,
    get_training_load,
    get_race_history,
    search_knowledge,
    get_weekly_trend,
    get_pace_prediction,
]
