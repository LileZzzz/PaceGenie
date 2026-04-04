"""Shared pytest configuration and fixtures for PaceGenie tests.

load_dotenv() is called here (not in individual test files) so environment
variables are available before any agent imports happen.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import pytest

from agent.graph import build_graph


@pytest.fixture(scope="session")
def graph():
    """Singleton compiled graph shared across the test session."""
    return build_graph()
