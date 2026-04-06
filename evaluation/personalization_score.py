"""Personalization Score — LLM-as-Judge evaluation.

Runs 20 questions through the PaceGenie agent and scores each response 1-5
based on how well it cites the user's actual Garmin data (distances, paces,
heart rates, injury history, personal bests).

Usage:
    uv run python evaluation/personalization_score.py

Requirements:
    - docker-compose up -d  (local postgres + backend)
    - .env with LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

Output:
    Per-question score + overall average
    Resume bullet suggestion at the end
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.graph import get_graph

# ---------------------------------------------------------------------------
# 20 evaluation questions — mix of personalization, RAG, and tool-use prompts
# ---------------------------------------------------------------------------
QUESTIONS: list[tuple[str, str]] = [
    ("ps-01", "How has my training volume been over the last 30 days? Am I at risk of injury?"),
    ("ps-02", "I have a history of knee injuries. What should I watch out for in my current training?"),
    ("ps-03", "I want to finish a half marathon in under 110 minutes. Based on my current fitness, is that realistic?"),
    ("ps-04", "Is my training intensity distribution reasonable? Am I doing enough easy running?"),
    ("ps-05", "How did my training go last week, and how should I structure this week?"),
    ("ps-06", "What is my current weekly mileage and how does it compare to last week?"),
    ("ps-07", "Based on my recent runs, what pace should I target for an easy run tomorrow?"),
    ("ps-08", "Am I overtraining? What does my heart rate data suggest?"),
    ("ps-09", "What is my predicted finish time for a 10K based on my recent race history?"),
    ("ps-10", "How many kilometres have I run in the last 7 days?"),
    ("ps-11", "Should I increase my mileage next week, or is my body showing signs of fatigue?"),
    ("ps-12", "What was my fastest recent run, and what pace did I hit?"),
    ("ps-13", "How does my current training compare to my peak weeks before my last race?"),
    ("ps-14", "Based on my injury history, which muscle groups should I focus on strengthening?"),
    ("ps-15", "What is my typical heart rate during easy runs versus hard runs?"),
    ("ps-16", "How long until I could realistically attempt a full marathon based on my current base?"),
    ("ps-17", "What percentage of my runs are in zone 2 heart rate?"),
    ("ps-18", "I want to run a 5K PR next month. What does my training suggest about my current fitness?"),
    ("ps-19", "Can you summarize my training from the past two weeks in numbers?"),
    ("ps-20", "What is my average cadence and how does it compare to the recommended 170-180 spm?"),
]

# ---------------------------------------------------------------------------
# LLM-as-Judge prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of an AI running coach's response.

Your task: Score the response from 1 to 5 based on PERSONALIZATION — how well it cites
the specific user's actual data (distances, paces, heart rate values, injury history,
personal bests, weekly mileage, etc.) from their Garmin training data.

Scoring rubric:
  1 — Generic advice with NO specific numbers or personal data cited
  2 — Mostly generic with 1 vague personal reference
  3 — Some personalization, cites 1-2 specific numbers from user's data
  4 — Good personalization, cites 3+ specific numbers, clearly tailored to this user
  5 — Excellent personalization, richly cites data, directly addresses user's history and goals

IMPORTANT: Only consider whether the response uses the USER'S SPECIFIC DATA.
Generic running advice (e.g., "you should run 80% easy") does NOT count as personalization
unless it references the user's actual percentages.

Respond with ONLY a single integer: 1, 2, 3, 4, or 5. No explanation."""


def _ask_agent(question: str, session_id: str) -> str:
    """Run one question through the agent and return the final reply."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(
        {
            "messages": [("user", question)],
            "user_id": "demo_user",
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config=config,
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return "(no response)"


def _judge_response(question: str, answer: str, llm: ChatOpenAI) -> int:
    """Ask the LLM judge to score the answer 1-5. Returns the integer score."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nAgent response:\n{answer}"),
    ]
    raw = llm.invoke(messages).content.strip()
    # Parse first digit found
    for ch in raw:
        if ch.isdigit() and ch in "12345":
            return int(ch)
    return 3  # fallback if judge returns unexpected text


def main() -> None:
    print("PaceGenie — Personalization Score (LLM-as-Judge)")
    print("=" * 60)
    print(f"Running {len(QUESTIONS)} questions through the agent ...\n")

    judge_llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "kimi-k2.5"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )

    scores: list[int] = []

    for qid, question in QUESTIONS:
        print(f"[{qid}] {question[:70]}")

        # Get agent answer
        answer = _ask_agent(question, session_id=qid)

        # Get judge score
        score = _judge_response(question, answer, judge_llm)
        scores.append(score)

        # Show brief answer preview
        preview = answer[:120].replace("\n", " ")
        print(f"       Answer: {preview}...")
        print(f"       Score : {'★' * score}{'☆' * (5 - score)} ({score}/5)\n")

    # Summary
    avg = statistics.mean(scores)
    median = statistics.median(scores)

    print("=" * 60)
    print(f"Questions scored : {len(scores)}")
    print(f"Average score    : {avg:.2f} / 5.0")
    print(f"Median score     : {median:.1f} / 5.0")
    print(f"Score breakdown  : {scores}")
    print(f"Distribution     : 1s={scores.count(1)} 2s={scores.count(2)} 3s={scores.count(3)} 4s={scores.count(4)} 5s={scores.count(5)}")
    print("=" * 60)
    print(f"\nResume bullet → \"Personalization Score {avg:.1f}/5.0 (LLM-as-Judge, {len(scores)} Garmin-grounded questions)\"")


if __name__ == "__main__":
    main()
