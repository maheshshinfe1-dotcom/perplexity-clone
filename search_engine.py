import asyncio
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
}

SEARCH_BACKENDS = ["google", "auto", "duckduckgo", "bing", "yahoo", "brave"]

FOCUS_QUERY_HINTS = {
    "web": "",
    "academic": "research paper study arxiv",
    "code": "documentation github code example",
    "news": "latest news update",
}


def _apply_focus_query(query: str, focus: str) -> str:
    """Steers web search queries toward the user's selected Focus Mode."""
    hint = FOCUS_QUERY_HINTS.get(focus, "")
    return f"{query} {hint}".strip() if hint else query


def _run_ddgs_with_failover(query: str, max_results: int) -> list[dict]:
    """Tries multiple DDGS search backends sequentially until one returns results."""
    for backend in SEARCH_BACKENDS:
        try:
            print(f"🌐 Searching via DDGS backend: {backend}...")
            results = DDGS(timeout=10).text(
                query,
                max_results=max_results,
                backend=backend,
            )
            if results:
                print(f"✅ Got {len(results)} results from {backend}")
                return results
        except Exception as e:
            print(f"⚠️ DDGS '{backend}' failed ({type(e).__name__}). Failing over...")
    return []


async def _fallback_google_news_rss(client: httpx.AsyncClient, query: str, limit: int = 5) -> list[dict]:
    """Fallback 1: Queries Google News RSS directly (no API key, rarely rate-limited)."""
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = await client.get(url, headers=HEADERS, timeout=8.0, follow_redirects=True)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        items = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "Untitled").strip()
            link = (item.findtext("link") or "").strip()
            desc_html = item.findtext("description") or ""
            desc_text = " ".join(BeautifulSoup(desc_html, "html.parser").get_text(" ").split())
            source_el = item.find("source")
            pub_date = (item.findtext("pubDate") or "").strip()

            source_url = source_el.attrib.get("url", "") if source_el is not None else ""
            source_name = source_el.text.strip() if (source_el is not None and source_el.text) else ""
            domain = urlparse(source_url).netloc.replace("www.", "") if source_url else "news.google.com"

            snippet = f"{desc_text} ({source_name}, {pub_date})".strip()
            items.append({
                "title": title,
                "href": link,
                "domain": domain,
                "body": snippet or title,
            })
        if items:
            print(f"✅ Fallback Google News RSS returned {len(items)} results")
        return items
    except Exception:
        print("⚠️ Google News RSS fallback failed")
        return []


async def _fallback_wikipedia(client: httpx.AsyncClient, query: str, limit: int = 3) -> list[dict]:
    """Fallback 2: Queries Wikipedia's free public search API."""
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote_plus(query)}&utf8=&format=json&srlimit={limit}"
    )
    try:
        resp = await client.get(url, headers=HEADERS, timeout=8.0)
        resp.raise_for_status()
        data = resp.json()
        items = []
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title", "Wikipedia")
            snippet = BeautifulSoup(hit.get("snippet", ""), "html.parser").get_text(" ")
            page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            items.append({
                "title": f"{title} — Wikipedia",
                "href": page_url,
                "domain": "en.wikipedia.org",
                "body": " ".join(snippet.split()),
            })
        if items:
            print(f"✅ Fallback Wikipedia API returned {len(items)} results")
        return items
    except Exception:
        print("⚠️ Wikipedia fallback failed")
        return []


async def fetch_page_text(client: httpx.AsyncClient, url: str, max_chars: int = 2000) -> str:
    """Downloads a single webpage asynchronously and strips HTML noise."""
    if not url or "news.google.com/rss/articles" in url:
        return ""
    try:
        response = await client.get(url, headers=HEADERS, timeout=5.0, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception:
        return ""


async def search_and_scrape(query: str, max_results: int = 5, focus: str = "web") -> list[dict]:
    """Queries live web search with Focus Mode + failover and scrapes returned URLs concurrently."""
    focused_query = _apply_focus_query(query, focus)
    raw_results = await asyncio.to_thread(_run_ddgs_with_failover, focused_query, max_results)

    async with httpx.AsyncClient() as client:
        if not raw_results:
            print("🔄 All DDGS backends rate-limited. Activating direct RSS + Wikipedia failover...")
            rss_hits, wiki_hits = await asyncio.gather(
                _fallback_google_news_rss(client, query, limit=max_results),
                _fallback_wikipedia(client, query, limit=2),
            )
            raw_results = (rss_hits + wiki_hits)[:max_results]

        if not raw_results:
            return []

        tasks = [fetch_page_text(client, item.get("href", "")) for item in raw_results]
        scraped_pages = await asyncio.gather(*tasks)

    sources = []
    for idx, (item, full_text) in enumerate(zip(raw_results, scraped_pages), start=1):
        url = item.get("href", "")
        domain = item.get("domain") or urlparse(url).netloc.replace("www.", "")
        snippet = item.get("body", "")

        sources.append({
            "id": idx,
            "title": item.get("title", "Untitled Source"),
            "url": url,
            "domain": domain,
            "snippet": snippet,
            "content": full_text if len(full_text) > 150 else snippet,
        })

    return sources
