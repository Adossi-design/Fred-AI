"""Web search skill using DuckDuckGo (no API key needed).

Search results are returned in a citation-friendly format: each result is
numbered, and the tool output includes a ready-to-use markdown "Sources" block
plus instructions telling the model to cite claims inline with [n] and to end
its answer with that Sources list. Because the web UI renders markdown, the
source links become clickable automatically.
"""

from ddgs import DDGS

CITATION_GUIDE = (
    "\n\nINSTRUCTIONS FOR YOUR ANSWER:\n"
    "- Use these results to answer, and cite them inline with bracketed numbers "
    "like [1] or [2] right after the facts they support.\n"
    "- End your answer with the exact 'Sources' block shown below so the links "
    "stay clickable. Only include the sources you actually used.\n"
)


def _format_results(query: str, results: list) -> str:
    """Render search results with a citation guide and a markdown Sources block."""
    if not results:
        return "No results found."

    # Body the model reads to write its answer
    lines = [f'Search results for "{query}":\n']
    sources_block = ["Sources:"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "Untitled").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or "").strip()
        lines.append(f"[{i}] {title}\n    {url}\n    {body[:200]}")
        # Markdown link so it renders clickable in the web UI
        sources_block.append(f"[{i}] [{title}]({url})")

    return "\n".join(lines) + CITATION_GUIDE + "\n" + "\n".join(sources_block)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted, citation-ready search results as a string
    """
    try:
        # Ensure max_results is an integer
        max_results = int(max_results) if max_results else 5

        with DDGS() as ddgs:
            # ddgs v9 API
            search_results = ddgs.text(query, max_results=max_results)
            results = list(search_results) if search_results else []

        return _format_results(query, results)

    except Exception as e:
        return f"Search failed: {str(e)}"


def get_current_news(topic: str) -> str:
    """
    Get current news and recent information about a topic.
    Use this for questions about current events, recent news, or anything that
    requires up-to-date information (politics, sports, technology, celebrities, etc).

    Args:
        topic: The topic to search for current news about

    Returns:
        Recent news and information about the topic, citation-ready
    """
    try:
        with DDGS() as ddgs:
            # Search news specifically (ddgs v9 API)
            news_results = ddgs.news(topic, max_results=5)
            results = list(news_results) if news_results else []

        if not results:
            # Fallback to regular search with date qualifier
            return web_search(f"{topic} 2026 latest news", max_results=5)

        # Normalize news results to the same shape as text results, keeping date/source
        lines = [f'Latest news about "{topic}":\n']
        sources_block = ["Sources:"]
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "Untitled").strip()
            url = (r.get("url") or r.get("href") or "").strip()
            date = r.get("date", "Recent")
            source = r.get("source", "Unknown")
            body = (r.get("body") or "").strip()
            lines.append(
                f"[{i}] {title}\n"
                f"    Date: {date} | Source: {source}\n"
                f"    {url}\n"
                f"    {body[:200]}"
            )
            sources_block.append(f"[{i}] [{title}]({url})")

        return "\n".join(lines) + CITATION_GUIDE + "\n" + "\n".join(sources_block)

    except Exception:
        # Fallback to regular search
        return web_search(f"{topic} 2026 latest", max_results=5)
