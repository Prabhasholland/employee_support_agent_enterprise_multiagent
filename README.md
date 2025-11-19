# 🚀 Employee Support & Knowledge Base Agent (Enterprise Multi-Agent System)

An advanced **multi-agent employee support assistant** built for enterprise environments like **Google**, created as part of the **Google AI Agents Intensive – Capstone Project (Nov 10–14, 2025)**.

This project demonstrates **all major concepts** from the course, including:
✔ Multi-agent architecture
✔ Tools (custom tools, API tools)
✔ Parallel + Sequential Agents
✔ Sessions & Memory
✔ Long-term Memory Bank
✔ Observability (logging, tracing, metrics)
✔ Agent Evaluation
✔ Deployment (Cloud Run, Docker)
✔ Gemini-ready LLM integration

---

## 📌 Problem Statement

Large enterprises deal with thousands of repetitive HR, IT, and policy-related questions daily:

* “How many paid leaves do I get?”
* “How do I reset my password?”
* “What is the work-from-home policy?”
* “How do I report a laptop issue?”

Employees wait, support teams repeat the same answers, and productivity drops.

---

## 🎯 Solution Overview

This project implements a **multi-agent employee support system** that:

### 🔹 Understands the user’s question (Classification Agent)

Categorizes into **HR / IT / General** automatically.

### 🔹 Retrieves relevant company policies (Retrieval Agent)

Uses **TF-IDF semantic search**, running **parallel retrieval across categories**.

### 🔹 Generates accurate answers (LLM Answer Agent)

Uses a Gemini-ready wrapper function `call_llm()`.

### 🔹 Creates support tickets for unclear queries (Escalation Agent)

Simulates ServiceNow/Jira via `TicketingTool`.

### 🔹 Maintains session context (SessionStore)

Stores conversation turns like InMemorySessionService.

### 🔹 Saves long-term memory (MemoryBank)

Writes important interactions to disk.

### 🔹 Tracks logs, metrics & observability

Generates:

* `interactions.jsonl`
* `metrics.json`
* session logs
* long-term memory logs

### 🔹 Can be deployed as an API

FastAPI + Dockerfile + Cloud Run deployment instructions included.

---

## 🧠 Architecture Diagram (ASCII)

```
User Query
    │
    ▼
┌──────────────────────────┐
│  EmployeeSupportOrchestrator
└──────────────────────────┘
    │
    ▼
┌──────────────┐
│Classification│───► HR / IT / GENERAL
└──────────────┘
    │
    ▼
┌──────────────────────────┐
│ RetrievalAgent + DocumentSearchTool
│ (Parallel search across categories)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│       AnswerAgent        │ (LLM: Gemini/OpenAI)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│    EscalationAgent       │──► TicketingTool (ServiceNow-style)
└──────────────────────────┘
    │
    ▼
Logs, Metrics, MemoryBank, SessionStore
```

---

## 🧩 Features (Mapped to Capstone Rubric)

### ✔ Multi-Agent System

* **ClassificationAgent**
* **RetrievalAgent**
* **AnswerAgent**
* **EscalationAgent**
* **EmployeeSupportOrchestrator**

### ✔ Custom Tools

* **DocumentSearchTool**
* **TicketingTool (ServiceNow/Jira mock)**

### ✔ Parallel & Sequential Agents

* Sequential: classify → retrieve → answer → escalate
* Parallel: retrieval across HR/IT/GENERAL docs using ThreadPoolExecutor

### ✔ Sessions & Memory

* **SessionStore:** short-term memory
* **MemoryBank:** long-term memory JSONL
* **Context compaction:** only last 10 turns kept

### ✔ Observability

* Logs all interactions
* Tracks escalation rate, similarity score averages
* Outputs metrics JSON file

### ✔ Agent Evaluation

`SimpleEvaluator` computes:

* total interactions
* escalation rate
* avg similarity score

### ✔ Deployment (Cloud Run)

Included:

* `api_fastapi.py` — REST API
* `Dockerfile`
* `DEPLOYMENT.md` — exact commands for Cloud Run

### ✔ Gemini Integration (Bonus)

Replace `call_llm()` with Gemini API call:

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def call_llm(prompt: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()
```

---

## 📂 Project Structure

```
employee-support-agent-enterprise/
│
├── src/
│   └── employee_support_multiagent.py   # Main multi-agent system
│
├── api_fastapi.py                       # Deployment API
├── Dockerfile                           # Container deployment
├── DEPLOYMENT.md                        # Cloud Run guide
├── requirements.txt
├── README.md
│
├── data/                                # Policy documents
├── logs/                                # Interaction logs + metrics
└── memory/                              # Long-term memory
```

---

## 🔧 How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI service

```bash
uvicorn api_fastapi:app --host 0.0.0.0 --port 8080
```

### 3. Test the agent

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

## 🛠 Integrating a Real LLM (Gemini)

Inside `src/employee_support_multiagent.py`, modify:

```python
def call_llm(prompt: str) -> str:
```

Replace with any Gemini or GPT model.

---

## ☁️ Deployment (Cloud Run)

Detailed instructions in `DEPLOYMENT.md`, summary:

1. Build:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/employee-agent
```

2. Deploy:

```bash
gcloud run deploy employee-agent \
    --image gcr.io/PROJECT_ID/employee-agent \
    --platform managed \
    --region REGION \
    --allow-unauthenticated
```

## 👤 Author

**Banavath Prabhas (Prabhasholland)**
Google AI Agents Intensive — Capstone Project
Enterprise Track: Employee Support Agent
