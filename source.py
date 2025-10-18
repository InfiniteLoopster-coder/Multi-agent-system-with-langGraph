import os, time, hashlib, requests, arxiv
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from tavily import TavilyClient  # pip install tavily-python

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _mk_id(url: str) -> str:
    return hashlib.sha1(url.strip().lower().encode()).hexdigest()[:8]

# ---------- Web Search (Tavily) ----------
def fetch_web(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    res = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )
    items = []
    for r in res.get("results", []):
        url = r.get("url")
        if not url: 
            continue
        items.append({
            "id": _mk_id(url),
            "title": r.get("title") or url,
            "url": url,
            "source": "web",
            "published_at": None,
            "snippet": r.get("content", "")[:600],
            "content": None,
        })
    return items

# ---------- Academic: arXiv ----------
def fetch_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    items = []
    for p in search.results():
        url = p.entry_id
        published = p.published.strftime(ISO) if p.published else None
        abs_ = (p.summary or "").strip()
        items.append({
            "id": _mk_id(url),
            "title": p.title,
            "url": url,
            "source": "arxiv",
            "published_at": published,
            "snippet": abs_[:600],
            "content": abs_,
        })
    return items

# ---------- Academic: Semantic Scholar ----------
def fetch_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    # No key required for light use; respect rate limits
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,year,authors,url,externalIds,abstract",
        "limit": max_results
    }
    r = requests.get(endpoint, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = []
    for p in data.get("data", []):
        url = p.get("url") or (p.get("externalIds", {}).get("ArXiv") and f"https://arxiv.org/abs/{p['externalIds']['ArXiv']}")
        if not url:
            continue
        published = f"{p.get('year')}-01-01T00:00:00Z" if p.get("year") else None
        abs_ = (p.get("abstract") or "").strip()
        items.append({
            "id": _mk_id(url),
            "title": p.get("title") or url,
            "url": url,
            "source": "semanticscholar",
            "published_at": published,
            "snippet": abs_[:600],
            "content": abs_,
        })
    return items

# ---------- News (via Tavily “news-like” or your own NewsAPI client) ----------
def fetch_news(query: str, days: int = 14, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Simple approach: bias Tavily for fresh results by including recency terms,
    or swap this for NewsAPI/GDELT if you prefer strict news sources.
    """
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    q = f"{query} site:reuters.com OR site:bbc.com OR site:bloomberg.com OR site:apnews.com after:{since}"
    return fetch_web(q, max_results=max_results)
