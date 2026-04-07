"""Ablation config constants for PaceGenie evaluation.

Re-exports the canonical AgentConfig presets from agent.config so evaluation
scripts have a clean, co-located import path without reaching into agent/.
"""
from agent.config import (  # noqa: F401
    AgentConfig,
    BASELINE,
    NO_RAG,
    NO_REFLECTION,
    BETTER_PROMPT,
    CONFIG_MAP,
)

__all__ = ["AgentConfig", "BASELINE", "NO_RAG", "NO_REFLECTION", "BETTER_PROMPT", "CONFIG_MAP"]
