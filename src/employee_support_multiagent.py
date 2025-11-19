import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LLM CALL (Gemini / GPT)
# =========================
# IMPORTANT: For security, no keys are included.
# Replace this function with a real Gemini / OpenAI call in your environment.

def call_llm(prompt: str) -> str:
    """Placeholder LLM call.

    To enable Gemini, replace this with something like:

    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    def call_llm(prompt: str) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    """
    return (
        "I'm an internal assistant running in demo mode. "
        "Please connect me to Gemini or another LLM for production answers."
    )


# =========================
# DATA MODELS
# =========================

@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    content: str
    category: str  # HR / IT / GENERAL


@dataclass
class Ticket:
    ticket_id: str
    user_id: str
    category: str
    query: str
    context: str
    created_at: str
    status: str = "OPEN"


@dataclass
class SessionTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


# =========================
# MEMORY & SESSIONS
# =========================

class SessionStore:
    """In-memory session state (short-term memory)."""

    def __init__(self):
        self.sessions: Dict[str, List[SessionTurn]] = {}

    def add_turn(self, session_id: str, role: str, content: str):
        turns = self.sessions.setdefault(session_id, [])
        turns.append(
            SessionTurn(
                role=role,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

    def get_history(self, session_id: str, max_turns: int = 10) -> List[SessionTurn]:
        turns = self.sessions.get(session_id, [])
        # Simple context compaction: keep only last N turns
        return turns[-max_turns:]


class MemoryBank:
    """Long-term memory persisted to disk as JSONL."""

    def __init__(self, path: str = "memory/long_term_memory.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def add_memory(self, memory: Dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(memory, ensure_ascii=False) + "\n")


# =========================
# OBSERVABILITY
# =========================

class Observability:
    """Simple logging + metrics to illustrate observability."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interactions_path = self.log_dir / "interactions.jsonl"
        self.metrics_path = self.log_dir / "metrics.json"

        if not self.metrics_path.exists():
            self.metrics_path.write_text(
                json.dumps(
                    {"total_queries": 0, "escalations": 0, "avg_similarity": 0.0},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def log_interaction(self, record: Dict):
        with self.interactions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def update_metrics(self, similarity: float, escalated: bool):
        data = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        total = data.get("total_queries", 0) + 1
        esc = data.get("escalations", 0) + (1 if escalated else 0)
        avg_sim = data.get("avg_similarity", 0.0)
        # incremental average
        avg_sim = avg_sim + (similarity - avg_sim) / total
        data.update(
            {"total_queries": total, "escalations": esc, "avg_similarity": avg_sim}
        )
        self.metrics_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# =========================
# KNOWLEDGE BASE & TOOLS
# =========================

class KnowledgeBase:
    """Loads HR / IT / General policy documents and supports TF-IDF search."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunks: List[DocumentChunk] = []
        self._load_docs()
        self._build_indexes()

    def _load_docs(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        for file in self.data_dir.glob("*.md"):
            category = "GENERAL"
            if "hr" in file.name.lower():
                category = "HR"
            elif "it" in file.name.lower():
                category = "IT"
            content = file.read_text(encoding="utf-8")
            self._chunk_document(
                doc_id=file.name, source=str(file), content=content, category=category
            )

    def _chunk_document(self, doc_id: str, source: str, content: str, category: str, max_chars: int = 600):
        paragraphs = content.split("\n\n")
        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(buffer) + len(para) < max_chars:
                buffer += para + "\n\n"
            else:
                if buffer.strip():
                    self.chunks.append(
                        DocumentChunk(
                            doc_id=doc_id,
                            source=source,
                            content=buffer.strip(),
                            category=category,
                        )
                    )
                buffer = para + "\n\n"
        if buffer.strip():
            self.chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    source=source,
                    content=buffer.strip(),
                    category=category,
                )
            )

    def _build_indexes(self):
        # Build three separate TF-IDF models per category to illustrate parallel agents
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.matrices: Dict[str, any] = {}

        for category in ["HR", "IT", "GENERAL"]:
            corpus = [c.content for c in self.chunks if c.category == category]
            if not corpus:
                continue
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform(corpus)
            self.vectorizers[category] = vec
            self.matrices[category] = mat

    def search(self, query: str, category: Optional[str] = None, top_k: int = 3):
        """If category is given, search within that category; otherwise search all in parallel."""
        if category and category in self.vectorizers:
            return self._search_single(query, category, top_k)
        # parallel search across all categories
        results: List[tuple] = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {}
            for cat in self.vectorizers.keys():
                futures[ex.submit(self._search_single, query, cat, top_k)] = cat
            for fut in futures:
                res = fut.result()
                results.extend(res)
        # sort globally
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _search_single(self, query: str, category: str, top_k: int):
        vec = self.vectorizers[category]
        mat = self.matrices[category]
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, mat)[0]
        ranked = sims.argsort()[::-1]
        results = []
        # gather top results
        filtered_chunks = [c for c in self.chunks if c.category == category]
        for idx in ranked[:top_k]:
            score = float(sims[idx])
            chunk = filtered_chunks[idx]
            results.append((score, chunk))
        return results


class DocumentSearchTool:
    """Custom tool that wraps KnowledgeBase.search for use by agents."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def search(self, query: str, category: Optional[str] = None, top_k: int = 3):
        return self.kb.search(query, category, top_k)


class TicketingTool:
    """Custom tool that creates internal tickets (mock ServiceNow / Jira)."""

    def __init__(self):
        self.counter = 0

    def create_ticket(self, user_id: str, category: str, query: str, context: str) -> Ticket:
        self.counter += 1
        ticket_id = f"TCKT-{self.counter:05d}"
        return Ticket(
            ticket_id=ticket_id,
            user_id=user_id,
            category=category,
            query=query,
            context=context,
            created_at=datetime.utcnow().isoformat(),
            status="OPEN",
        )


# =========================
# AGENTS
# =========================

class ClassificationAgent:
    """Classifies queries into HR / IT / GENERAL."""

    def classify(self, query: str) -> str:
        q = query.lower()
        hr_keywords = [
            "leave", "vacation", "holiday", "salary", "benefits",
            "hr", "payroll", "parental", "sick", "policy", "time off",
        ]
        it_keywords = [
            "laptop", "password", "vpn", "network", "email",
            "wifi", "system", "login", "software", "hardware",
        ]
        if any(w in q for w in hr_keywords):
            return "HR"
        if any(w in q for w in it_keywords):
            return "IT"
        return "GENERAL"


class RetrievalAgent:
    """Uses the DocumentSearchTool to fetch relevant snippets."""

    def __init__(self, search_tool: DocumentSearchTool):
        self.search_tool = search_tool

    def retrieve(self, query: str, category: Optional[str]) -> Dict:
        results = self.search_tool.search(query, category, top_k=3)
        if not results:
            return {"context": "", "max_score": 0.0}
        max_score = results[0][0]
        context_chunks = [chunk.content for score, chunk in results]
        context = "\n\n".join(context_chunks)
        return {"context": context, "max_score": max_score}


class AnswerAgent:
    """LLM-powered agent that generates the final answer."""

    def generate(self, query: str, category: str, history: List[SessionTurn], context: str) -> str:
        history_str = ""
        for turn in history:
            history_str += f"[{turn.role}] {turn.content}\n"

        prompt = f"""You are an internal enterprise assistant for employees at a large tech company like Google.

Conversation so far:
{history_str}

Current category: {category}
Employee query: {query}

Retrieved policy snippets:
{context}

Using ONLY the policy snippets above and reasonable corporate common sense,
answer the employee. If you are not confident or information is missing,
clearly say that and recommend contacting HR/IT support.
"""
        return call_llm(prompt)


class EscalationAgent:
    """Decides when to escalate and uses TicketingTool to create tickets."""

    LOW_CONF_KEYWORDS = [
        "not confident",
        "not sure",
        "insufficient",
        "contact hr",
        "contact it",
    ]

    def __init__(self, ticket_tool: TicketingTool, threshold: float = 0.18):
        self.ticket_tool = ticket_tool
        self.threshold = threshold

    def maybe_escalate(
        self,
        user_id: str,
        category: str,
        query: str,
        context: str,
        answer: str,
        similarity: float,
    ) -> Optional[Ticket]:
        low_conf = similarity < self.threshold or any(
            kw in answer.lower() for kw in self.LOW_CONF_KEYWORDS
        )
        if not low_conf:
            return None
        return self.ticket_tool.create_ticket(
            user_id=user_id, category=category, query=query, context=context
        )


# =========================
# EVALUATOR (AGENT EVALUATION)
# =========================

class SimpleEvaluator:
    """Naive evaluator that inspects logs and prints basic metrics.

    In a real system, you might have human-labeled data or use automated
    evaluation against golden answers.
    """

    def __init__(self, log_path: str = "logs/interactions.jsonl"):
        self.log_path = Path(log_path)

    def evaluate(self) -> Dict:
        if not self.log_path.exists():
            return {}
        records = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        total = len(records)
        esc = sum(1 for r in records if r.get("escalated"))
        avg_sim = sum(r.get("max_similarity_score", 0.0) for r in records) / total if total else 0.0

        summary = {
            "total_interactions": total,
            "escalation_rate": esc / total if total else 0.0,
            "avg_similarity_score": avg_sim,
        }
        return summary


# =========================
# ORCHESTRATOR (MULTI-AGENT)
# =========================

class EmployeeSupportOrchestrator:
    """High-level orchestrator that wires all agents together.

    This demonstrates:
    - Sequential reasoning (classify -> retrieve -> answer -> escalate)
    - Parallel retrieval across HR/IT/GENERAL when category is unknown
    - Sessions & memory
    - Observability
    """

    def __init__(self, data_dir: str = "data", log_dir: str = "logs", memory_path: str = "memory/long_term_memory.jsonl"):
        self.kb = KnowledgeBase(data_dir=data_dir)
        self.search_tool = DocumentSearchTool(self.kb)
        self.ticket_tool = TicketingTool()

        self.classifier = ClassificationAgent()
        self.retriever = RetrievalAgent(self.search_tool)
        self.answer_agent = AnswerAgent()
        self.escalation_agent = EscalationAgent(self.ticket_tool)

        self.sessions = SessionStore()
        self.memory_bank = MemoryBank(memory_path)
        self.observability = Observability(log_dir=log_dir)

    def handle_query(self, session_id: str, user_id: str, query: str) -> Dict:
        # 1) Add user turn to session memory
        self.sessions.add_turn(session_id, "user", query)

        # 2) Classification (sequential)
        category = self.classifier.classify(query)

        # 3) Retrieval (may be category-specific or across all)
        retrieval = self.retriever.retrieve(query, category if category != "GENERAL" else None)
        context = retrieval["context"]
        max_score = retrieval["max_score"]

        # 4) Generate answer with LLM
        history = self.sessions.get_history(session_id)
        answer = self.answer_agent.generate(query, category, history, context)

        # 5) Escalation decision
        ticket = self.escalation_agent.maybe_escalate(
            user_id=user_id,
            category=category,
            query=query,
            context=context,
            answer=answer,
            similarity=max_score,
        )
        escalated = ticket is not None

        # 6) Add assistant turn to session
        self.sessions.add_turn(session_id, "assistant", answer)

        # 7) Log & metrics (observability)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "category": category,
            "answer": answer,
            "max_similarity_score": max_score,
            "escalated": escalated,
            "ticket": asdict(ticket) if ticket else None,
        }
        self.observability.log_interaction(record)
        self.observability.update_metrics(max_score, escalated)

        # 8) Long-term memory (e.g. store important Q&A)
        self.memory_bank.add_memory(
            {
                "session_id": session_id,
                "user_id": user_id,
                "query": query,
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return record


if __name__ == "__main__":
    orchestrator = EmployeeSupportOrchestrator()
    demo_query = "How many paid vacation days do I get in a year?"
    result = orchestrator.handle_query(session_id="session-demo", user_id="user-123", query=demo_query)
    print("Answer:", result["answer"])
    print("Category:", result["category"])
    print("Similarity:", result["max_similarity_score"])
    print("Escalated:", result["escalated"])
    print("Ticket:", result["ticket"])
