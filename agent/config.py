"""Agent configuration for ablation studies and evaluation.

Defines AgentConfig — a frozen dataclass that toggles feature flags
(RAG, reflection, system prompt variant) used by build_graph() in graph.py.

The production API always uses get_graph() (singleton, all features ON).
Evaluation scripts use build_graph(config) with these presets to produce
distinct LangSmith experiment rows for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent graph variant.

    All fields default to the production settings so AgentConfig() == BASELINE.
    """
    rag_enabled: bool = True
    reflection_enabled: bool = True
    system_prompt_variant: str = "default"      # "default" only (grounding_v2 removed)
    knowledge_tools_enabled: bool = True        # False → search_knowledge removed from ToolNode
    semantic_reflection: bool = False           # True → reflection trigger is LLM-as-judge


# ---------------------------------------------------------------------------
# Named presets — used by evaluation scripts and run_ablation.py
# ---------------------------------------------------------------------------
BASELINE = AgentConfig()

NO_RAG = AgentConfig(
    rag_enabled=False,
    knowledge_tools_enabled=False,  # removes search_knowledge tool — truly no knowledge
)

NO_REFLECTION = AgentConfig(
    reflection_enabled=False,
)

SEMANTIC_REFLECT = AgentConfig(
    semantic_reflection=True,       # LLM-as-judge replaces rule-based trigger
)

# ---------------------------------------------------------------------------
# Config map — maps CLI flag values to (AgentConfig, version_tag) pairs
# ---------------------------------------------------------------------------
CONFIG_MAP: dict[str, tuple[AgentConfig, str]] = {
    "baseline":         (BASELINE,         "v1-baseline"),
    "no-rag":           (NO_RAG,           "v2-no-rag"),
    "no-reflection":    (NO_REFLECTION,    "v3-no-reflection"),
    "semantic-reflect": (SEMANTIC_REFLECT, "v4-semantic-reflect"),
}
