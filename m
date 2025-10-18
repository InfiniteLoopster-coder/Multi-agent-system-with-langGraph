from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Optional: add tools (web search, wiki, vector retriever, etc.)
# from langchain_community.tools.tavily_search import TavilySearchResults

# ---------- State ----------
class PipelineState(TypedDict, total=False):
    user_query: str
    research_notes: Dict[str, Any]
    final_answer: str
    meta: Dict[str, Any]

# ---------- Models ----------
LLM_RESEARCH = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
LLM_WRITE    = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# ---------- Prompts ----------
RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous research analyst. For the user query, break it into sub-questions, "
     "gather concise bullet-point facts with sources (URLs), and add a 1-sentence justification "
     "for each fact. If evidence is weak or missing, flag it."),
    ("human", "User query: {user_query}\n"
              "Return JSON with keys: points (list of strings), citations (list of URLs), "
              "coverage_score (0-1), caveats (list of strings).")
])

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise technical writer. Synthesize the research notes into a clear, structured answer. "
     "Use only provided notes; do not invent facts. If notes are insufficient, say what’s missing."),
    ("human",
     "User query: {user_query}\n"
     "Research notes (JSON): {research_notes}\n"
     "Constraints: concise, logically ordered, cite sources inline as [#]. End with 'Sources:' list.")
])

# ---------- Nodes ----------
def research_node(state: PipelineState) -> PipelineState:
    query = state["user_query"]
    # (Optional) call tools before/within LLM to fetch snippets; here we go direct LLM for brevity.
    msg = RESEARCH_PROMPT.format_messages(user_query=query)
    out = LLM_RESEARCH.invoke(msg)  # .content expected to be JSON per prompt
    # Be defensive: parse JSON with try/except; default safe shape if parsing fails.
    import json
    try:
        payload = json.loads(out.content)
    except Exception:
        payload = {"points": [], "citations": [], "coverage_score": 0.0, "caveats": ["ParseError"]}

    return {
        "research_notes": payload,
        "meta": {"flags": {"insufficient": payload.get("coverage_score", 0) < 0.4}}
    }

def write_node(state: PipelineState) -> PipelineState:
    msg = WRITE_PROMPT.format_messages(
        user_query=state["user_query"],
        research_notes=state.get("research_notes", {})
    )
    out = LLM_WRITE.invoke(msg)
    return {"final_answer": out.content}

# ---------- Graph ----------
def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("research", research_node)
    g.add_node("write", write_node)

    g.set_entry_point("research")
    g.add_edge("research", "write")
    g.add_edge("write", END)

    return g.compile()

app = build_graph()

# ---------- Public API ----------
def run_multi_agent_system(query: str) -> str:
    state: PipelineState = {"user_query": query}
    result = app.invoke(state)
    return result.get("final_answer", "")
