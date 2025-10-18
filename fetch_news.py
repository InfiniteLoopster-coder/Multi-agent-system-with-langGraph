from newsapi import NewsApiClient

def fetch_news_newsapi(query: str, days: int = 7, max_results: int = 20):
    client = NewsApiClient(api_key=os.environ["NEWSAPI_KEY"])
    from_ = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    res = client.get_everything(q=query, language="en", sort_by="publishedAt",
                                from_param=from_, page_size=min(max_results, 100))
    items = []
    for a in res.get("articles", []):
        url = a.get("url");  title = a.get("title") or url
        if not url: continue
        items.append({
            "id": _mk_id(url),
            "title": title,
            "url": url,
            "source": "news",
            "published_at": a.get("publishedAt"),
            "snippet": (a.get("description") or "")[:600],
            "content": None
        })
    return items
