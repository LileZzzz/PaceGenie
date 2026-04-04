"""Unit tests covering the code-review fixes applied to PaceGenie.

Tests are grouped by the file they exercise:
  - api/main.py  : ChatRequest validation, CORS origins, p50 median
  - agent/tools.py : Field constraints (gt=0, ge=1), _seconds_to_pace_str rounding
  - agent/nodes.py : MessageUpdate / ReflectionUpdate type annotations,
                     get_llm_with_tools singleton
  - agent/graph.py : get_graph thread-safety (double-checked lock)

Run:  uv run pytest tests/test_fixes.py -v
"""
from __future__ import annotations

import statistics
import threading
from typing import get_type_hints

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# api/main.py — ChatRequest validation
# ---------------------------------------------------------------------------


class TestChatRequestValidation:
    """Pydantic Field constraints on ChatRequest."""

    @pytest.fixture(autouse=True)
    def _import(self):
        # Import inside fixture so load_dotenv() in conftest runs first
        from api.main import ChatRequest

        self.ChatRequest = ChatRequest

    def test_valid_request_passes(self):
        req = self.ChatRequest(message="hello", user_id="user_1", session_id="abc-123")
        assert req.message == "hello"
        assert req.user_id == "user_1"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError, match="string_too_short"):
            self.ChatRequest(message="")

    def test_message_too_long_rejected(self):
        with pytest.raises(ValidationError):
            self.ChatRequest(message="x" * 4097)

    def test_user_id_special_chars_rejected(self):
        """SQL/prompt-injection characters must be rejected."""
        for bad in ["'; DROP TABLE--", "user<script>", "user id", "user@domain"]:
            with pytest.raises(ValidationError, match="pattern"):
                self.ChatRequest(message="hi", user_id=bad)

    def test_user_id_too_long_rejected(self):
        with pytest.raises(ValidationError):
            self.ChatRequest(message="hi", user_id="a" * 65)

    def test_session_id_special_chars_rejected(self):
        with pytest.raises(ValidationError, match="pattern"):
            self.ChatRequest(message="hi", session_id="../../etc/passwd")

    def test_default_user_id_is_valid(self):
        req = self.ChatRequest(message="hello")
        assert req.user_id == "demo_user"
        assert req.session_id == "default"

    def test_alphanumeric_with_dash_underscore_allowed(self):
        req = self.ChatRequest(message="hi", user_id="user-123_ABC")
        assert req.user_id == "user-123_ABC"


# ---------------------------------------------------------------------------
# api/main.py — CORS origins read from env
# ---------------------------------------------------------------------------


class TestCorsOrigins:
    """_allowed_origins is a list (not wildcard '*')."""

    def test_allowed_origins_is_list(self):
        import api.main as main_module

        assert isinstance(main_module._allowed_origins, list)
        assert "*" not in main_module._allowed_origins

    def test_allowed_origins_non_empty(self):
        import api.main as main_module

        assert len(main_module._allowed_origins) >= 1


# ---------------------------------------------------------------------------
# api/main.py — p50 uses statistics.median (correct for even n)
# ---------------------------------------------------------------------------


class TestP50Median:
    """Verify the p50 formula uses true median, not off-by-one index."""

    def test_even_length_median_is_average_of_two_middle(self):
        # For [1, 2, 3, 4]: median = 2.5, old formula returned sorted[2] = 3
        values = [1.0, 2.0, 3.0, 4.0]
        assert statistics.median(values) == 2.5

    def test_odd_length_median_is_middle_element(self):
        values = [1.0, 2.0, 3.0]
        assert statistics.median(values) == 2.0

    def test_get_timing_stats_uses_median(self, monkeypatch):
        """Inject known values and confirm p50 equals the true median."""
        from api.main import _response_times, get_timing_stats

        _response_times.clear()
        for v in [10.0, 20.0, 30.0, 40.0]:
            _response_times.append(v)

        stats = get_timing_stats()
        # True median of [10, 20, 30, 40] = 25.0; old formula returned 30
        assert stats["p50_ms"] == round(statistics.median([10.0, 20.0, 30.0, 40.0]))

        _response_times.clear()


# ---------------------------------------------------------------------------
# agent/tools.py — Field constraint: target_distance_km > 0
# ---------------------------------------------------------------------------


class TestGetPacePredictionInputValidation:
    """GetPacePredictionInput.target_distance_km must be > 0."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.tools import GetPacePredictionInput

        self.Model = GetPacePredictionInput

    def test_zero_distance_rejected(self):
        with pytest.raises(ValidationError):
            self.Model(user_id="u1", target_distance_km=0.0)

    def test_negative_distance_rejected(self):
        with pytest.raises(ValidationError):
            self.Model(user_id="u1", target_distance_km=-5.0)

    def test_valid_distance_passes(self):
        m = self.Model(user_id="u1", target_distance_km=42.2)
        assert m.target_distance_km == 42.2


# ---------------------------------------------------------------------------
# agent/tools.py — Field constraint: weeks >= 1
# ---------------------------------------------------------------------------


class TestGetWeeklyTrendInputValidation:
    """GetWeeklyTrendInput.weeks must be >= 1."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.tools import GetWeeklyTrendInput

        self.Model = GetWeeklyTrendInput

    def test_zero_weeks_rejected(self):
        with pytest.raises(ValidationError):
            self.Model(user_id="u1", weeks=0)

    def test_negative_weeks_rejected(self):
        with pytest.raises(ValidationError):
            self.Model(user_id="u1", weeks=-3)

    def test_valid_weeks_passes(self):
        m = self.Model(user_id="u1", weeks=8)
        assert m.weeks == 8


# ---------------------------------------------------------------------------
# agent/tools.py — _seconds_to_pace_str rounds correctly
# ---------------------------------------------------------------------------


class TestSecondsToPaceStr:
    """_seconds_to_pace_str should round, not truncate."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.tools import _seconds_to_pace_str

        self.fn = _seconds_to_pace_str

    def test_rounds_up_near_minute_boundary(self):
        # 359.9 s should round to 6:00, not truncate to 5:59
        assert self.fn(359.9) == "6:00"

    def test_rounds_down(self):
        # 360.4 s → still 6:00
        assert self.fn(360.4) == "6:00"

    def test_exact_value(self):
        assert self.fn(360.0) == "6:00"

    def test_sub_ten_seconds_zero_padded(self):
        assert self.fn(305.0) == "5:05"

    def test_zero_seconds(self):
        assert self.fn(0.0) == "0:00"


# ---------------------------------------------------------------------------
# agent/nodes.py — get_llm_with_tools returns same object on repeated calls
# ---------------------------------------------------------------------------


class TestGetLlmWithToolsSingleton:
    """get_llm_with_tools() must return the identical object every time."""

    def test_singleton_identity(self):
        from agent.nodes import get_llm_with_tools

        a = get_llm_with_tools()
        b = get_llm_with_tools()
        assert a is b, "get_llm_with_tools() must return the same singleton"

    def test_thread_safe_singleton(self):
        """Concurrent callers must all receive the same object."""
        from agent.nodes import get_llm_with_tools

        results: list = []

        def call():
            results.append(get_llm_with_tools())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is results[0] for r in results), "All threads must get the same singleton"


# ---------------------------------------------------------------------------
# agent/nodes.py — TypedDict annotations use list[BaseMessage]
# ---------------------------------------------------------------------------


class TestTypedDictAnnotations:
    """MessageUpdate and ReflectionUpdate messages fields should be list[BaseMessage]."""

    def test_message_update_messages_type(self):
        from langchain_core.messages import BaseMessage

        from agent.nodes import MessageUpdate

        hints = get_type_hints(MessageUpdate)
        assert hints["messages"] == list[BaseMessage]

    def test_reflection_update_messages_type(self):
        from langchain_core.messages import BaseMessage

        from agent.nodes import ReflectionUpdate

        hints = get_type_hints(ReflectionUpdate)
        assert hints["messages"] == list[BaseMessage]


# ---------------------------------------------------------------------------
# agent/graph.py — get_graph thread-safety
# ---------------------------------------------------------------------------


class TestGetGraphThreadSafety:
    """Concurrent calls to get_graph() must return the identical compiled graph."""

    def test_thread_safe_singleton(self):
        from agent.graph import get_graph

        results: list = []

        def call():
            results.append(get_graph())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is results[0] for r in results), "All threads must get the same graph"
