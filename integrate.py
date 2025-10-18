import asyncio
from typing import List, Dict, Any
from sources import fetch_web, fetch_arxiv, fetch_semantic_scholar, fetch_news
from evidence import dedupe_evidence, naive_rerank

async def _run_io(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

async def gather_sources_async(query: str) -> List[Dict[str, Any]]:
    tasks = [
        _run_io(fetch_web, query, 8),
        _run_io(fetch_arxiv, query, 5),
        _run_io(fetch_semantic_scholar, query, 5),
        _run_io(fetch_news, query, 14, 8),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception): 
            continue
        items.extend(r)
    items = dedupe_evidence(items)
    items = naive_rerank(query, items)[:20]
    # Re-assign compact numeric ids for pretty citations [1], [2], ...
    for i, it in enumerate(items, 1):
        it["id"] = str(i)
    return items
