# PaceGenie — AI Running Coach

An AI-powered running coach built with LangGraph, Hybrid RAG, and a React dashboard. Ask natural-language questions about your training data and get grounded, personalized answers backed by your actual Garmin metrics.

---

## What It Does

- **Chat with your data** — Ask "Am I at injury risk?" or "Can I break 1:50 in the half?" and get answers pulled directly from your training history.
- **Hybrid RAG** — Knowledge base answers use pgvector (semantic) + BM25 (keyword) retrieval fused with Reciprocal Rank Fusion for higher-quality context.
- **Semantic self-reflection** — LLM-as-judge evaluates each answer and injects a targeted critique if it fails to use retrieved data (Reflexion pattern). Fired on 45% of eval questions vs 0% for rule-based triggers.
- **Race time prediction** — Uses Riegel's formula applied to your personal bests to predict finishing times at any distance.
- **Multi-turn memory** — Conversation history is preserved within a session so follow-up questions have full context.
- **Real-time dashboard** — Split-screen layout with heart rate trends, weekly volume, training zones, personal bests, and injury log.

---

## Architecture

```
User ──► React Frontend (Vite + Tailwind)
              │
              ▼ POST /api/chat
         FastAPI Backend
              │
              ▼
         LangGraph Agent (ReAct + Reflection)
         ┌────────────────────────────────┐
         │  retrieve_context              │  ◄── Hybrid RAG (pgvector + BM25)
         │        ↓                       │
         │  generate_response ──► tools   │  ◄── Garmin data tools (mock)
         │        ↓         ◄────┘        │
         │  reflect_on_answer             │  ◄── Semantic quality gate (LLM-as-judge)
         │        ↓                       │
         │  generate_response (retry)     │
         └────────────────────────────────┘
```

**Key design decisions:**
- `retrieve_context` runs before every LLM call — keeps generation focused on reasoning, not retrieval
- Semantic reflection: LLM judge issues APPROVE or REVISE + targeted critique; max 2 retries
- All Garmin tools fall back to `data/mock_garmin.json` — no live Garmin API required
- LLM is configurable via env vars — works with any OpenAI-compatible endpoint (tested with Kimi K2.5)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) StateGraph |
| LLM | Any OpenAI-compatible API (Kimi K2.5, GPT-4o, etc.) |
| Embeddings | Google `text-embedding-004` (768-dim) |
| Vector store | PostgreSQL + pgvector |
| Keyword search | BM25 (`rank-bm25`) |
| RAG fusion | Reciprocal Rank Fusion (k=60, α=0.6/0.4) |
| Memory | LangGraph `MemorySaver` (in-process, thread-scoped) |
| API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS |
| Charts | Recharts |
| Markdown | react-markdown + remark-gfm |

---

## Quickstart

### Prerequisites

- Python 3.12+, [`uv`](https://docs.astral.sh/uv/) package manager
- Node.js 18+ and npm
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension (for RAG; optional)

### 1. Clone and install

```bash
git clone <repo-url>
cd PaceGenie

# Python dependencies
uv sync

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM — any OpenAI-compatible endpoint
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1   # omit for standard OpenAI

# PostgreSQL + pgvector (optional — skip if not using RAG)
DATABASE_URL=postgresql://user:pass@localhost:5432/pacegenie

# Google embeddings (optional — required for RAG ingestion)
GOOGLE_API_KEY=AIza...

# LangSmith tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=PaceGenie
```

### 3. Run the backend

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok"}`

### 4. Run the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`

---

## Populating the Knowledge Base (optional)

The agent works out of the box using mock Garmin data. To enable RAG over the running knowledge base:

```bash
# 1. Start PostgreSQL with pgvector
docker compose up -d postgres

# 2. Ingest knowledge documents
uv run python rag/ingest.py
```

Knowledge documents live in `data/knowledge/` and cover: training plans, VO2max intervals, heart rate zones, nutrition, and recovery protocols.

---

## Running Tests

```bash
# Integration test — full agent end-to-end (10 questions)
uv run pytest tests/test_graph.py -v

# Mock data quality — validates personalized answer grounding
uv run pytest tests/test_mock_quality.py -v

# All tests
uv run pytest tests/ -v
```

Tests require a running LLM (set `LLM_API_KEY` in `.env`).

---

## Project Structure

```
PaceGenie/
├── agent/
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── nodes.py          # retrieve_context, generate_response, reflect_on_answer
│   ├── tools.py          # Garmin data tools (6 tools)
│   ├── state.py          # AgentState TypedDict
│   └── utils.py          # Shared helpers
├── api/
│   └── main.py           # FastAPI app, /api/chat, /health, /metrics/timing
├── rag/
│   ├── retriever.py      # Hybrid retriever (pgvector + BM25 + RRF)
│   ├── embeddings.py     # Google embedding wrapper
│   └── ingest.py         # Knowledge base ingestion pipeline
├── data/
│   ├── mock_garmin.json  # Mock training data (24 runs)
│   └── knowledge/        # RAG source documents (9 files)
├── evaluation/
│   ├── run_agent.py      # Phase 1: run agent, cache replies to JSONL
│   ├── run_judges.py     # Phase 2: score cache with 3 LLM-as-judge evaluators
│   ├── run_ablation.py   # Orchestrate all 4 configs sequentially
│   ├── langsmith_eval.py # LangSmith dataset + evaluator definitions
│   ├── METRICS_REPORT.md # Full ablation findings with analysis
│   └── cache/            # JSONL reply caches per config
├── tests/
│   ├── conftest.py       # Shared fixtures and load_dotenv
│   ├── test_graph.py     # End-to-end agent integration tests (10 questions)
│   └── test_mock_quality.py  # Data-grounding quality validation
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── chat/ChatInterface.tsx
│   │   │   └── dashboard/   # 16 dashboard components
│   │   ├── lib/config.ts     # API endpoint config
│   │   └── types/index.ts
│   └── data/mock_garmin.json # Frontend copy of mock data
└── pyproject.toml
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `get_recent_runs` | Last N runs with distance, pace, HR, type |
| `get_training_load` | 14-day load, weekly km, injury risk score |
| `get_race_history` | Personal bests across distances |
| `search_knowledge` | Hybrid RAG search over coaching knowledge base |
| `get_weekly_trend` | Week-by-week volume table (4–8 weeks) |
| `get_pace_prediction` | Riegel's formula race time prediction from PBs |

---

## Evaluation Results

4-config ablation study — LLM-as-judge, 20 Garmin-grounded questions ([full report](evaluation/METRICS_REPORT.md)):

| Config | Hallucination (grounded ↑) | Answer Relevance ↑ | Personalization ↑ |
|--------|---------------------------|-------------------|------------------|
| baseline | 85% | 0.96 | 4.7 / 5.0 |
| no-rag | 95% | 0.96 | 4.5 / 5.0 |
| no-reflection | 80% | 0.98 | 4.5 / 5.0 |
| **semantic-reflect** | 85% | 0.97 | **4.8 / 5.0** |

**Key finding:** Semantic reflection (LLM-as-judge trigger) fired on 45% of questions vs 0% rule-based, improving Personalization 4.5→4.8/5.0 (+6.7%) and doubling avg tool calls per question (1.6→3.0).

To run the evaluation pipeline:

```bash
# Phase 1 — run agent, cache replies
uv run python evaluation/run_agent.py --config semantic-reflect --n 20

# Phase 2 — score with LLM judges (+ optional LangSmith push)
uv run python evaluation/run_judges.py --config semantic-reflect --n 20 --langsmith
```

---

## Deployment

**Backend (Railway):**
1. Add a `Dockerfile` (FastAPI + uvicorn)
2. Add PostgreSQL plugin in Railway dashboard
3. Set env vars: `LLM_API_KEY`, `DATABASE_URL`, `GOOGLE_API_KEY`
4. Run `uv run python rag/ingest.py` post-deploy

**Frontend (Vercel):**
1. Connect repo, set root to `frontend/`
2. Set `VITE_API_URL=https://your-app.railway.app`
3. Deploy

---

## License

MIT
