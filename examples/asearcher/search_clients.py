"""
Async local search and visit clients for ASearcher.

Communicates with the local_search_server.py (or any compatible server)
that exposes /retrieve and /access endpoints.
"""

import asyncio

import aiohttp


SEMAPHORE = asyncio.Semaphore(256)

# Reuse ClientSession per event loop to avoid TCP connection churn under high concurrency
_SESSIONS: dict[tuple[int, str | None], aiohttp.ClientSession] = {}


def _get_session(proxy: str | None = None) -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    key = (id(loop), proxy)
    if key not in _SESSIONS or _SESSIONS[key].closed:
        kwargs = {
            "connector": aiohttp.TCPConnector(
                limit=500,
                limit_per_host=256,
                ttl_dns_cache=300,
            )
        }
        if proxy:
            kwargs["proxy"] = proxy
        _SESSIONS[key] = aiohttp.ClientSession(**kwargs)
    return _SESSIONS[key]


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 120,
    proxy: str | None = None,
) -> list[dict]:
    """
    Call local retrieval server (/retrieve) and return formatted results.

    Args:
        search_url: URL of the local retrieval server (e.g. "http://127.0.0.1:8000/retrieve")
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds
        proxy: Optional proxy URL

    Returns:
        List of dicts with keys: title, text, contents
    """
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,
    }
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    try:
        async with SEMAPHORE:
            session = _get_session(proxy)
            async with session.post(search_url, json=payload, timeout=timeout_obj) as resp:
                resp.raise_for_status()
                result = await resp.json()
    except Exception as e:
        print(f"Error calling local search at {search_url}: {e}")
        return []

    retrieval_results = result.get("result", [[]])[0]
    contexts = []
    for item in retrieval_results:
        if isinstance(item, dict):
            # Some servers wrap in {"document": doc, "score": score}
            if "document" in item:
                doc = item["document"]
            else:
                doc = item
            contents = doc.get("contents", "")
            title = doc.get("title", "")
            text = doc.get("text", "")
            url = doc.get("url", "")
            contexts.append({"title": title, "text": text, "contents": contents, "url": url})
    return contexts


async def local_visit(
    access_url: str,
    url: str,
    timeout: int = 120,
    proxy: str | None = None,
) -> str:
    """
    Call local page access server (/access) and return page content.

    Args:
        access_url: URL of the access endpoint (e.g. "http://127.0.0.1:8000/access")
        url: The Wikipedia page URL to visit
        timeout: Request timeout in seconds
        proxy: Optional proxy URL

    Returns:
        Page content string, or empty string if not found.
    """
    payload = {"urls": [url]}
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    try:
        async with SEMAPHORE:
            session = _get_session(proxy)
            async with session.post(access_url, json=payload, timeout=timeout_obj) as resp:
                resp.raise_for_status()
                result = await resp.json()
    except Exception as e:
        print(f"Error calling local access at {access_url} for url={url}: {e}")
        return ""

    pages = result.get("result", [])
    if not pages:
        return ""

    page = pages[0]
    if isinstance(page, dict):
        contents = page.get("contents", "")
        if not contents and "page" in page:
            contents = page["page"]
        return contents
    return str(page)


def format_search_results(results: list[dict]) -> str:
    """Format retrieval results into a string for the agent context."""
    if not results:
        return "No relevant passages found."

    lines = []
    for idx, doc in enumerate(results, start=1):
        title = doc.get("title", "No title")
        text = doc.get("text", "")
        url = doc.get("url", "")
        url_line = f"\nURL: {url}" if url else ""
        lines.append(f"[{idx}] Title: {title}{url_line}\n{text}")
    return "\n\n".join(lines)


def format_visit_result(content: str, url: str) -> str:
    """Format visit result into a string for the agent context."""
    if not content:
        return f"Failed to retrieve content from {url}."
    # Truncate very long pages to avoid context overflow
    max_len = 250000
    if len(content) > max_len:
        content = content[:max_len] + "\n... [content truncated]"
    return f"Content from {url}:\n{content}"
