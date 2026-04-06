"""Load test — sends 100 sequential POST /api/chat requests to the local backend.

Usage:
    uv run python scripts/load_test.py               # default: 100 requests
    uv run python scripts/load_test.py --n 50        # custom count
    uv run python scripts/load_test.py --url https://pacegenie-backend.onrender.com

After this runs, fetch latency percentiles:
    curl http://localhost:8000/metrics/timing
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import json

QUESTIONS = [
    "How has my training volume been this week?",
    "Am I at risk of overtraining?",
    "What was my average pace last week?",
    "How is my heart rate trending?",
    "Can I run a sub-2 hour half marathon?",
    "What is my weekly mileage trend?",
    "How many easy runs did I do this month?",
    "What is my current training load?",
    "Should I increase my mileage next week?",
    "What pace zones am I training in?",
]


def send_request(base_url: str, question: str, session_id: str) -> tuple[bool, float]:
    """Send one chat request. Returns (success, duration_ms)."""
    payload = json.dumps({
        "message": question,
        "user_id": "demo_user",
        "session_id": session_id,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        duration_ms = (time.perf_counter() - t0) * 1000
        return True, duration_ms
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        print(f"  ✗ Request failed: {exc}", file=sys.stderr)
        return False, duration_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the PaceGenie /api/chat endpoint")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the backend")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    total = args.n

    print(f"Sending {total} requests to {base_url}/api/chat ...")
    print("(each request calls the LLM — this will take a few minutes)\n")

    durations: list[float] = []
    failures = 0

    for i in range(total):
        question = QUESTIONS[i % len(QUESTIONS)]
        session_id = f"loadtest_{i}"
        ok, ms = send_request(base_url, question, session_id)

        if ok:
            durations.append(ms)
            status = f"{ms:6.0f}ms"
        else:
            failures += 1
            status = "FAILED"

        print(f"  [{i+1:3d}/{total}] {status}  — {question[:50]}")

    # Summary
    if durations:
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        p50 = durations_sorted[int(n * 0.50)]
        p95 = durations_sorted[max(0, int(n * 0.95) - 1)]
        p99 = durations_sorted[max(0, int(n * 0.99) - 1)]
        mean = sum(durations) / n

        print(f"\n{'─'*40}")
        print(f"Requests sent  : {total}")
        print(f"Successful     : {n}")
        print(f"Failed         : {failures}")
        print(f"Mean latency   : {mean:,.0f}ms")
        print(f"P50 latency    : {p50:,.0f}ms")
        print(f"P95 latency    : {p95:,.0f}ms")
        print(f"P99 latency    : {p99:,.0f}ms")
        print(f"{'─'*40}")
        print(f"\nResume bullet → \"P95 response latency {p95:,.0f}ms under {total}-request load\"")
        print(f"\nFor server-side stats: curl {base_url}/metrics/timing")
    else:
        print("\nAll requests failed — is the backend running?")
        print(f"  Start it with: docker-compose up -d")
        print(f"  Then: curl {base_url}/health")


if __name__ == "__main__":
    main()
