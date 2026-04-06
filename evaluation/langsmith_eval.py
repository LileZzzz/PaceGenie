"""LangSmith structured evaluation for PaceGenie.

Runs 20 personalization questions through the agent and scores each response
with 3 evaluators published to the LangSmith dashboard:

  1. hallucination    — did the agent invent facts? (0=hallucinated, 1=grounded)
  2. answer_relevance — did it answer the question? (0.0–1.0)
  3. personalization  — does it cite the user's Garmin data? (1–5 → 0.0–1.0)

Usage:
    docker-compose up -d
    uv run python evaluation/langsmith_eval.py

View results at: smith.langchain.com → PaceGenie project → Experiments
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT, ANSWER_RELEVANCE_PROMPT

from agent.graph import get_graph

# ---------------------------------------------------------------------------
# Dataset — 20 Garmin-grounded personalization questions
# ---------------------------------------------------------------------------
DATASET_NAME = "PaceGenie Personalization Eval"

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
# LLM-as-Judge prompt for personalization (custom — prebuilts don't cover this)
# ---------------------------------------------------------------------------
PERSONALIZATION_PROMPT = """\
You are an expert evaluator assessing an AI running coach's response.

Score the response from 1 to 5 based on PERSONALIZATION — how well it cites
the user's actual Garmin training data (distances, paces, heart rate values,
injury history, personal bests, weekly mileage, etc.).

Scoring rubric:
  1 — Generic advice with NO specific numbers or personal data cited
  2 — Mostly generic with 1 vague personal reference
  3 — Some personalization, cites 1-2 specific numbers from user's data
  4 — Good personalization, cites 3+ specific numbers, clearly tailored
  5 — Excellent, richly cites data, directly addresses user's history and goals

IMPORTANT: Generic running advice (e.g., "run 80% easy") does NOT count as
personalization unless it references the user's actual percentages.

Respond with ONLY a single integer: 1, 2, 3, 4, or 5. No explanation.

<question>{question}</question>
<response>{answer}</response>
"""

# ---------------------------------------------------------------------------
# Target function — wraps the agent for evaluate()
# ---------------------------------------------------------------------------
def run_agent(inputs: dict) -> dict:
    """evaluate() calls this with example.inputs. Returns {"reply": str}."""
    graph = get_graph()
    result = graph.invoke(
        {
            "messages": [("user", inputs["question"])],
            "user_id": "demo_user",
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config={"configurable": {"thread_id": inputs["session_id"]}},
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"reply": str(msg.content)}
    return {"reply": ""}


# ---------------------------------------------------------------------------
# Evaluator 3 — Custom personalization judge (DIY — no openevals dep needed)
# ---------------------------------------------------------------------------
def personalization_judge(inputs: dict, outputs: dict) -> dict:
    """Score 1-5 how well the reply cites the user's personal Garmin data."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "kimi-k2.5"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )
    prompt = PERSONALIZATION_PROMPT.format(
        question=inputs.get("question", ""),
        answer=outputs.get("reply", ""),
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    score = 3  # fallback
    for ch in raw:
        if ch in "12345":
            score = int(ch)
            break
    return {
        "key": "personalization_score",
        "score": score / 5.0,  # normalize to 0–1 for LangSmith
        "comment": f"Raw score: {score}/5",
    }


# ---------------------------------------------------------------------------
# Dataset setup — idempotent (skips creation if dataset already exists)
# ---------------------------------------------------------------------------
def ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
        return

    print(f"Creating dataset '{DATASET_NAME}' with {len(QUESTIONS)} examples...")
    client.create_dataset(
        DATASET_NAME,
        description="20 Garmin-grounded personalization questions for PaceGenie agent",
    )
    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=[
            {"inputs": {"question": q, "session_id": qid}}
            for qid, q in QUESTIONS
        ],
    )
    print("Dataset created.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("PaceGenie — LangSmith Evaluation")
    print("=" * 50)

    # Verify LangSmith env vars
    if not os.getenv("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set in .env")
        sys.exit(1)

    client = Client()
    ensure_dataset(client)

    # Build the two prebuilt openevals evaluators
    # They use the same LLM config as the agent
    model_str = f"openai/{os.getenv('LLM_MODEL', 'kimi-k2.5')}"

    hallucination_evaluator = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        model=model_str,
        feedback_key="hallucination",
    )

    relevance_evaluator = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        model=model_str,
        feedback_key="answer_relevance",
    )

    print(f"Running {len(QUESTIONS)} questions through the agent...")
    print("(This will take ~10 minutes — each question calls the LLM)\n")

    results = evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[
            hallucination_evaluator,
            relevance_evaluator,
            personalization_judge,
        ],
        experiment_prefix="pacegenie-baseline",
        metadata={
            "model": os.getenv("LLM_MODEL"),
            "description": "Baseline evaluation — hallucination, relevance, personalization",
        },
        max_concurrency=1,  # sequential to avoid rate limiting
    )

    print(f"\nResults URL: {results.url}")
    print("Open the URL above to see the full dashboard.\n")

    # Print summary to terminal
    try:
        df = results.to_pandas()
        print("=" * 50)
        for col in ["feedback.hallucination", "feedback.answer_relevance", "feedback.personalization_score"]:
            if col in df.columns:
                avg = df[col].mean()
                label = col.replace("feedback.", "")
                print(f"  {label:<25} avg = {avg:.2f}")

        # Personalization resume bullet (convert back to 1-5)
        if "feedback.personalization_score" in df.columns:
            p_avg = df["feedback.personalization_score"].mean() * 5
            print(f"\nResume bullet →")
            print(f'  "Personalization Score {p_avg:.1f}/5.0 (LangSmith Eval, {len(QUESTIONS)} Garmin-grounded questions)"')
        print("=" * 50)
    except Exception:
        print("(Could not summarize results — check the LangSmith URL above)")


if __name__ == "__main__":
    main()