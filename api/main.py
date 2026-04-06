"""FastAPI backend for the PaceGenie running-coach agent.

Exposes:
  POST /api/chat        — one conversational turn through the LangGraph agent
  GET  /health          — liveness probe
  GET  /metrics/timing  — P50/P95/P99 latency stats (see docs/METRICS.md §4)
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from agent.graph import get_graph

# 10 requests per minute per IP — enough for a demo, blocks abuse.
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# In-memory timing store — capped at 10k samples to prevent unbounded growth.
# (dev/demo; swap for Redis/DB in production)
# ---------------------------------------------------------------------------
_response_times: deque[float] = deque(maxlen=10_000)


# ---------------------------------------------------------------------------
# Lifespan: warm the singleton graph before the first request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-build the LangGraph compiled graph so the first chat request is fast."""
    get_graph()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PaceGenie API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Read allowed origins from env; default to localhost dev servers.
# Set ALLOWED_ORIGINS="https://your-app.com,https://api.your-app.com" in production.
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Record per-request latency for P95 tracking (see docs/METRICS.md §4)."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    _response_times.append(duration_ms)
    response.headers["X-Response-Time-Ms"] = str(round(duration_ms))
    return response


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
def get_agent() -> CompiledStateGraph:
    """Inject the singleton compiled graph; avoids rebuilding on every request."""
    return get_graph()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4096)
    user_id: str = Field(
        default="demo_user",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class TimingStats(TypedDict):
    count: int
    mean_ms: int
    p50_ms: int
    p95_ms: int
    p99_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for load-balancers and uptime monitors."""
    return {"status": "ok"}


@app.get("/metrics/timing")
def get_timing_stats() -> TimingStats | dict[str, str]:
    """Return latency percentiles over all requests since the last restart."""
    if not _response_times:
        return {"error": "no data yet — send some requests first"}
    sorted_times = sorted(_response_times)
    n = len(sorted_times)
    return TimingStats(
        count=n,
        mean_ms=round(statistics.mean(sorted_times)),
        p50_ms=round(statistics.median(sorted_times)),
        p95_ms=round(sorted_times[max(0, int(n * 0.95) - 1)]),
        p99_ms=round(sorted_times[max(0, int(n * 0.99) - 1)]),
    )


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,  # required by slowapi rate limiter
    req: ChatRequest,
    agent: CompiledStateGraph = Depends(get_agent),
) -> ChatResponse:
    """Run one conversational turn through the LangGraph ReAct agent.

    session_id maps to LangGraph's thread_id, so MemorySaver retains
    conversation history across requests in the same session.
    """
    config = {"configurable": {"thread_id": req.session_id}}
    result = await agent.ainvoke(
        {
            "messages": [("user", req.message)],
            "user_id": req.user_id,
            "retrieved_context": None,
            "reflection_count": 0,
        },
        config=config,
    )

    reply = "(no response)"
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply = str(msg.content)
            break

    return ChatResponse(reply=reply, session_id=req.session_id)
