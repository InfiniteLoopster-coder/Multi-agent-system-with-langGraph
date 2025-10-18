from typing import List, Dict, Any
import math

def _jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def dedupe_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = [x for x in items if x.get("url")]
    items_by_url = {}
    for x in items:
        key = x["url"].strip().lower()
        if key not in items_by_url:
            items_by_url[key] = x
    # title near-dup removal
    uniq = []
    for x in items_by_url.values():
        if any(_jaccard(x["title"], u["title"]) > 0.85 for u in uniq):
            continue
        uniq.append(x)
    return uniq

def naive_rerank(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # quick cosine-ish proxy: overlap on query terms
    qset = set(query.lower().split())
    def score(x):
        title = x["title"].lower()
        snip  = (x.get("snippet") or "").lower()
        ts = len(qset & set(title.split()))
        ss = len(qset & set(snip.split()))
        freshness = 0.0
        # Prefer newer for news/papers
        if x["source"] in {"news", "web", "arxiv", "semanticscholar"} and x.get("published_at"):
            freshness = 1.0
        return ts*2 + ss*0.5 + freshness
    return sorted(items, key=score, reverse=True)
