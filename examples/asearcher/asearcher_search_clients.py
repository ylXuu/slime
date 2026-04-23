import asyncio

import aiohttp


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
        List of dicts with keys: title, text, contents, url
    """
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,
    }
    timeout_obj = aiohttp.ClientTimeout(total=timeout, sock_connect=timeout)

    cnt = 0
    last_exception = None
    while cnt < 10:
        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {"json": payload, "timeout": timeout_obj}
                if proxy:
                    kwargs["proxy"] = proxy
                async with session.post(search_url, **kwargs) as response:
                    response.raise_for_status()
                    res = await response.json()

            retrieval_results = res.get("result", [[]])[0]
            contexts = []
            for item in retrieval_results:
                if isinstance(item, dict):
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
        except Exception as e:
            print("local_search", e.__class__.__name__, e.__cause__)
            last_exception = e
            print(f"Search Engine switched to {search_url}")
            cnt += 1
            await asyncio.sleep(10)

    raise RuntimeError("Fail to post search query to RAG server") from last_exception


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
    timeout_obj = aiohttp.ClientTimeout(total=timeout, sock_connect=timeout)

    cnt = 0
    last_exception = None
    while cnt < 10:
        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {"json": payload, "timeout": timeout_obj}
                if proxy:
                    kwargs["proxy"] = proxy
                async with session.post(access_url, **kwargs) as response:
                    response.raise_for_status()
                    res = await response.json()

            pages = res.get("result", [])
            if not pages:
                return ""

            page = pages[0]
            if isinstance(page, dict):
                contents = page.get("contents", "")
                if not contents and "page" in page:
                    contents = page["page"]
                return contents
            return str(page)
        except Exception as e:
            print("local_visit", e.__class__.__name__, e.__cause__)
            last_exception = e
            print(f"Search Engine switched to {access_url}")
            cnt += 1
            await asyncio.sleep(10)

    raise RuntimeError("Fail to post access request to RAG server") from last_exception


def format_search_results(results: list[dict]) -> str:
    """Format retrieval results into a string for the agent context."""
    if not results:
        return "No relevant passages found."

    lines = []
    for idx, doc in enumerate(results, start=1):
        url = doc.get("url", "")
        content = doc.get("contents", "")
        lines.append(f"[{idx}] URL: {url}\nContent: {content}")
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
