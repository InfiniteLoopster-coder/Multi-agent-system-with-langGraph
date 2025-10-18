from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
import asyncio
from integrate import gather_sources_async
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class PipelineState(TypedDict, total=False):
    user_query: str
    evidence: List[Dict[str, Any]]  # <- NEW
    research_notes: Dict[str, Any]
    final_answer: str
    meta: Dict[str, Any]

LLM_RESEARCH = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
LLM_WRITE    = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
LLM_VALIDATE = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

GATHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You will be given normalized evidence entries. "
               "Your job (later) is to synthesize; this prompt is only for structure reference.")
])

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous research analyst. Using ONLY the provided evidence list, "
     "produce JSON with: points (bullet facts), citations (list of evidence ids you used per point), "
     "coverage_score (0-1), caveats (list). Do not invent facts."),
    ("human",
     "User query: {user_query}\nEvidence:\n{evidence_table}\n"
     "Return strictly JSON.")
])

VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact validator. For each point, verify it is directly supported by the cited evidence ids. "
     "Remove or rewrite unsupported points. Return JSON with points_validated (list), dropped (list of reasons).")
])

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise technical writer. Synthesize into a clear answer. "
     "Use inline citations like [id]. End with 'Sources:' and one line per id: title — url"),
    ("human",
     "User query: {user_query}\nValidated notes JSON: {validated}\nEvidence:\n{evidence_table}")
])

def evidence_to_table(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for e in evidence:
        when = e.get("published_at") or "n/a"
        lines.append(f"- [{e['id']}] {e['title']} ({e['source']}, {when}) — {e['url']}")
    return "\n".join(lines)

# ----- New nodes -----
def gather_sources_node(state: PipelineState) -> PipelineState:
    query = state["user_query"]
    evidence = asyncio.run(gather_sources_async(query))
    return {"evidence": evidence}

def research_node(state: PipelineState) -> PipelineState:
    ev_table = evidence_to_table(state["evidence"])
    msg = RESEARCH_PROMPT.format_messages(user_query=state["user_query"], evidence_table=ev_table)
    out = LLM_RESEARCH.invoke(msg)
    import json
    try:
        payload = json.loads(out.content)
    except Exception:
        payload = {"points": [], "citations": [], "coverage_score": 0.0, "caveats": ["ParseError"]}
    return {"research_notes": payload}

def validate_node(state: PipelineState) -> PipelineState:
    # light structured validation: ensure every point cites at least one evidence id
    import json
    notes = state.get("research_notes", {})
    if not notes:
        return {"research_notes": {"points": [], "citations": [], "coverage_score": 0.0, "caveats": ["Empty"]}}
    prompt = VALIDATE_PROMPT.format_messages()
    # Keep it simple; pass both notes and evidence table
    ev_table = evidence_to_table(state["evidence"])
    user = [
        {"type": "text", "text": f"Research notes JSON:\n{notes}\nEvidence:\n{ev_table}\nReturn strictly JSON."}
    ]
    out = LLM_VALIDATE.invoke(prompt + user)  # compatible composition
    try:
        payload = json.loads(out.content)
    except Exception:
        payload = {"points_validated": notes.get("points", []), "dropped": ["ValidatorParseError"]}
    # replace notes with validated
    validated = {"points": payload.get("points_validated", []), "citations": notes.get("citations", []),
                 "coverage_score": notes.get("coverage_score", 0.0), "caveats": notes.get("caveats", [])}
    return {"research_notes": validated}

def write_node(state: PipelineState) -> PipelineState:
    ev_table = evidence_to_table(state["evidence"])
    import json
    msg = WRITE_PROMPT.format_messages(
        user_query=state["user_query"],
        validated=json.dumps(state["research_notes"]),
        evidence_table=ev_table
    )
    out = LLM_WRITE.invoke(msg)
    return {"final_answer": out.content}

# ----- Graph wiring -----
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("gather_sources", gather_sources_node)
    g.add_node("research", research_node)
    g.add_node("validate", validate_node)
    g.add_node("write", write_node)

    g.set_entry_point("gather_sources")
    g.add_edge("gather_sources", "research")
    g.add_edge("research", "validate")
    g.add_edge("validate", "write")
    g.add_edge("write", END)
    return g.compile()

app = build_graph()

def run_multi_agent_system(query: str) -> str:
    state: PipelineState = {"user_query": query}
    result = app.invoke(state)
    return result.get("final_answer", "")
