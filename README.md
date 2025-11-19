# Employee Support & Knowledge Base Agent for Google (Enterprise Multi-Agent Version)

This repository contains the **enterprise, multi-agent version** of the Employee Support Agent
you can submit for the **5-Day AI Agents Intensive Capstone**.

It is designed to score highly on:

- Category 1: Pitch (problem, solution, value)
- Category 2: Implementation (architecture, code, AI integration)
- Bonus: Gemini use, deployment, video

## 🌟 Core Idea

An **Employee Support & Knowledge Base Agent** for a large organization like Google:

- Answers HR, IT, and general policy questions
- Uses a **multi-agent system** (classifier, retriever, answer agent, escalation agent)
- Retrieves internal policies via TF-IDF search
- Tracks **sessions & memory** across conversations
- Logs interactions and metrics for **observability**
- Can be **deployed to Cloud Run** via the provided Dockerfile + FastAPI API

---

## 🧱 Architecture Overview

Main implementation in:

- `src/employee_support_multiagent.py`

Key components:

- **ClassificationAgent** – classifies query into HR / IT / GENERAL
- **RetrievalAgent** – uses `DocumentSearchTool` (custom tool) over a KnowledgeBase with TF-IDF
- **AnswerAgent** – LLM-powered agent that generates answers using `call_llm`
- **EscalationAgent** – decides whether to open a ticket and uses `TicketingTool`
- **SessionStore** – manages short-term chat history per session (session + state)
- **MemoryBank** – long-term memory persisted in `memory/long_term_memory.jsonl`
- **Observability** – logs interactions and metrics in `logs/`
- **SimpleEvaluator** – reads logs and computes basic metrics

The high-level orchestration is in:

- `EmployeeSupportOrchestrator`

Flow:

1. User query enters `/query` endpoint (FastAPI, `api_fastapi.py`)
2. Orchestrator:
   - logs user turn in `SessionStore`
   - runs **ClassificationAgent** (sequential)
   - runs **RetrievalAgent** (category-specific or parallel across all categories)
   - calls **AnswerAgent** (LLM) with conversation history + retrieved context
   - calls **EscalationAgent** to decide on ticketing
   - updates logs, metrics, and **MemoryBank**

---

## ✅ Features Mapped to Competition Requirements

**Multi-agent system**
- Implemented via:
  - `ClassificationAgent`
  - `RetrievalAgent`
  - `AnswerAgent`
  - `EscalationAgent`
  - `EmployeeSupportOrchestrator` orchestrates them sequentially.

**Parallel & Sequential agents**
- Sequential chain: classify → retrieve → answer → escalate (see `handle_query`).
- Parallel behavior: `KnowledgeBase.search` uses `ThreadPoolExecutor` to query HR/IT/GENERAL indexes in parallel when category is GENERAL.

**Tools**
- Custom tools:
  - `DocumentSearchTool` – wraps knowledge base retrieval
  - `TicketingTool` – mock ServiceNow/Jira ticket creator
- These are used by **RetrievalAgent** and **EscalationAgent**.

**Sessions & Memory**
- `SessionStore` – in-memory session management (similar to InMemorySessionService).
- `MemoryBank` – long-term memory, storing important Q&A events.
- Context compaction – `SessionStore.get_history(max_turns=10)` simulates context compaction by trimming conversation history.

**Observability**
- `Observability` class:
  - writes `logs/interactions.jsonl`
  - tracks metrics in `logs/metrics.json` (total queries, escalation count, average similarity).

**Agent evaluation**
- `SimpleEvaluator`:
  - loads `interactions.jsonl`
  - computes summary metrics like:
    - total_interactions
    - escalation_rate
    - avg_similarity_score

**Deployment (Cloud Run / Cloud-based runtime)**
- `api_fastapi.py` exposes `/query` endpoint.
- `Dockerfile` builds a container around the FastAPI app.
- `DEPLOYMENT.md` contains commands to deploy to Cloud Run.
- This can be referenced in your write-up as “Cloud Run-ready deployment”.

**Gemini usage**
- `call_llm` is intentionally a placeholder.
- In your environment, connect Gemini in this function to earn Gemini bonus points.

---

## 🏃‍♂️ Running Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI app:

   ```bash
   uvicorn api_fastapi:app --host 0.0.0.0 --port 8080
   ```

3. Send a request:

   ```bash
   curl -X POST http://localhost:8080/query \
        -H "Content-Type: application/json" \
        -d '{
          "session_id": "session-1",
          "user_id": "user-123",
          "query": "How many paid vacation days do I get per year?"
        }'
   ```

---

## 🧪 Running the Evaluator

After a few queries have been processed, run (inside a Python shell):

```python
from src.employee_support_multiagent import SimpleEvaluator
ev = SimpleEvaluator()
print(ev.evaluate())
```

This gives you evaluation metrics you can screenshot or mention in your Kaggle write-up.

---

## ✏️ Where to Customize for Your Submission

- **LLM** – Implement Gemini in `call_llm` and mention it under “Effective Use of Gemini”.
- **Docs** – Replace `data/*.md` with your real or richer mock policies.
- **Write-up** – Use this repo plus your Kaggle notebook to explain:
  - problem
  - solution
  - architecture diagram
  - features used (multi-agent, tools, memory, observability, evaluation, deployment).

---

## 📽️ Video

You can reuse the earlier 2-minute script, but add:

- a slide for **multi-agent architecture**
- a short screen recording hitting the `/query` endpoint
- a quick glimpse of `logs/metrics.json` or the evaluator output

This will align perfectly with the “YouTube Video Submission” bonus requirement.
