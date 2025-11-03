from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from models.llm import chat
from tools.vectorstore import load_index  # ⬅️ Chroma loader
from tools.web_search import WebSearch
import os

class NodeState(BaseModel):
    question: str
    plan: Optional[str] = None
    search_queries: List[str] = []
    retrieved: List[Dict[str, Any]] = []
    answer_draft: Optional[str] = None
    citations: List[str] = []
    actions: List[str] = []
    quality_score: Optional[float] = None

# Try to load a prebuilt Chroma index (data/chroma). If missing, continue without it.
try:
    vs = load_index("data/chroma")
except Exception:
    vs = None

def node_plan(state: NodeState):
    prompt = f"""You are a research planner. Break the question into 2-4 precise search queries and a short plan.
Question: {state.question}
Return JSON with keys: plan, queries (list)"""
    out = chat([{"role": "user", "content": prompt}])
    # Light parse / fallback
    state.plan = out
    if not state.search_queries:
        state.search_queries = [state.question, "site:arxiv.org " + state.question]
    return state

def node_gather(state: NodeState):
    tavi = WebSearch(api_key=os.getenv("TAVILY_API_KEY", ""))
    web_docs = []
    for q in state.search_queries[:4]:
        for d in tavi.search(q, max_results=3):
            # normalize structure
            d["metadata"] = {"source": d.get("source")}
            web_docs.append(d)
    state.retrieved.extend(web_docs)
    return state

def node_retrieve(state: NodeState):
    # Use Chroma index if available; otherwise skip gracefully
    hits = vs.similarity_search(state.question, k=6) if vs else []
    for h in hits:
        src = (getattr(h, "metadata", {}) or {}).get("source", "local")
        state.retrieved.append({
            "content": h.page_content,
            "metadata": h.metadata,
            "source": src
        })
    return state

def node_grade(state: NodeState):
    unique_sources = {r.get("source") or r.get("metadata", {}).get("source") for r in state.retrieved}
    score = min(1.0, (len(state.retrieved) / 8) * 0.7 + (len(unique_sources) / 4) * 0.3)
    state.quality_score = score
    return state

def node_summarize(state: NodeState):
    # Build a compact, cited context
    context_snips = []
    for i, r in enumerate(state.retrieved[:10]):
        src = r.get("source", "unknown")
        txt = (r.get("content") or "")[:1200]
        context_snips.append(f"[{i}] ({src}) {txt}")
    context = "\n\n".join(context_snips)

    prompt = f"""Synthesize a concise, well-cited answer to the user's question using only the context snippets.
Question: {state.question}
Context:
{context}

Return sections:
1) Key findings
2) Comparison table (if applicable)
3) Cited references [use bracket index numbers]
4) Next actions for the user (practical steps)
Keep it < 300 words."""
    state.answer_draft = chat([{"role": "user", "content": prompt}])

    # Simple sources list to show in UI
    state.citations = [r.get("source", "unknown") for r in state.retrieved[:6]]
    state.actions = [
        "Open top 3 sources to verify claims",
        "Export summary to Markdown",
        "Queue a follow-up deep dive on the highest-uncertainty point"
    ]
    return state

def node_improve_or_end(state: NodeState):
    if state.quality_score is None or state.quality_score < 0.45:
        state.search_queries.extend([
            f"site:arxiv.org {state.question}",
            f"site:ultralytics.com {state.question}",
            f"site:paperswithcode.com {state.question}"
        ])
        return "gather"
    return END

# Build the graph
graph = StateGraph(NodeState)
graph.add_node("plan", node_plan)
graph.add_node("gather", node_gather)
graph.add_node("retrieve", node_retrieve)
graph.add_node("grade", node_grade)
graph.add_node("synthesize", node_summarize)
graph.add_node("improve_or_end", node_improve_or_end)

graph.add_edge("plan", "gather")
graph.add_edge("gather", "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_edge("grade", "synthesize")
graph.add_conditional_edges("synthesize", node_improve_or_end)

graph.set_entry_point("plan")
compiled = graph.compile()
