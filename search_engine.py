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
            print(f"🌐 Searching via DDGS backend ({backend}): {query[:60]}")
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


async def _staggered_ddgs(query: str, max_results: int, delay: float = 0.0) -> list[dict]:
    """Adds a small stagger before parallel searches to reduce rate-limit spikes."""
    if delay > 0:
        await asyncio.sleep(delay)
    return await asyncio.to_thread(_run_ddgs_with_failover, query, max_results)


def _run_ddgs_images(query: str, max_results: int = 8) -> list[dict]:
    """Queries DDGS image search and normalizes image results."""
    try:
        raw = DDGS(timeout=8).images(query, max_results=max_results) or []
        images = []
        for item in raw:
            img_url = (item.get("image") or "").strip()
            thumb_url = (item.get("thumbnail") or img_url).strip()
            source_url = (item.get("url") or img_url).strip()
            if not (img_url.startswith("http://") or img_url.startswith("https://")):
                continue
            domain = urlparse(source_url).netloc.replace("www.", "") or "web"
            images.append({
                "title": (item.get("title") or "Image").strip(),
                "image": img_url,
                "thumbnail": thumb_url if thumb_url.startswith("http") else img_url,
                "url": source_url,
                "domain": domain,
            })
        if images:
            print(f"🖼️ DDGS Images returned {len(images)} results")
        return images
    except Exception as e:
        print(f"⚠️ DDGS Images failed ({type(e).__name__}). Using fallback...")
        return []


async def _fallback_wikipedia_images(client: httpx.AsyncClient, query: str, limit: int = 6) -> list[dict]:
    """Fallback image source using Wikipedia's free pageimages API."""
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&generator=search&gsrsearch={quote_plus(query)}&gsrlimit={limit + 4}"
        "&prop=pageimages|info&piprop=thumbnail&pithumbsize=640&inprop=url&format=json"
    )
    try:
        resp = await client.get(url, headers=HEADERS, timeout=6.0)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            thumb = (page.get("thumbnail") or {}).get("source", "")
            if not thumb:
                continue
            title = (page.get("title") or "Wikipedia Image").strip()
            page_url = (page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}")
            images.append({
                "title": title,
                "image": thumb,
                "thumbnail": thumb,
                "url": page_url,
                "domain": "en.wikipedia.org",
            })
            if len(images) >= limit:
                break
        if images:
            print(f"🖼️ Fallback Wikipedia Images returned {len(images)} results")
        return images
    except Exception:
        return []


async def search_images(query: str, max_results: int = 8) -> list[dict]:
    """Fetches live images for a query with automatic Wikimedia fallback and deduplication."""
    images = await asyncio.to_thread(_run_ddgs_images, query, max_results)
    if len(images) < 4:
        async with httpx.AsyncClient() as client:
            wiki_imgs = await _fallback_wikipedia_images(client, query, limit=max_results)
            images.extend(wiki_imgs)

    deduped = []
    seen = set()
    for img in images:
        key = img.get("image") or img.get("thumbnail")
        if key and key not in seen:
            seen.add(key)
            deduped.append(img)
            if len(deduped) >= max_results:
                break
    return deduped


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


async def _scrape_and_format(client: httpx.AsyncClient, raw_results: list[dict]) -> list[dict]:
    """Scrapes all deduplicated URLs concurrently and formats final source objects."""
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


async def search_and_scrape(query: str, max_results: int = 5, focus: str = "web") -> list[dict]:
    """Standard single-query live search with Focus Mode + failover."""
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

        return await _scrape_and_format(client, raw_results)


async def multi_search_and_scrape(
    queries: list[str],
    max_total_sources: int = 8,
    focus: str = "web",
) -> list[dict]:
    """Pro Search: executes multiple decomposed queries in parallel, deduplicates URLs, and scrapes."""
    clean_queries = [q.strip() for q in queries if q and q.strip()][:3]
    if not clean_queries:
        return []

    per_query = max(3, (max_total_sources // len(clean_queries)) + 1)
    search_tasks = [
        _staggered_ddgs(_apply_focus_query(q, focus), per_query, delay=idx * 0.25)
        for idx, q in enumerate(clean_queries)
    ]
    batches = await asyncio.gather(*search_tasks)

    deduped: list[dict] = []
    seen_urls: set[str] = set()
    max_len = max((len(b) for b in batches), default=0)

    for slot in range(max_len):
        for batch in batches:
            if slot < len(batch):
                item = batch[slot]
                url = (item.get("href") or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduped.append(item)
                    if len(deduped) >= max_total_sources:
                        break
        if len(deduped) >= max_total_sources:
            break

    async with httpx.AsyncClient() as client:
        if not deduped:
            print("🔄 Pro Search DDGS rate-limited. Activating parallel RSS + Wikipedia failover...")
            rss_hits, wiki_hits = await asyncio.gather(
                _fallback_google_news_rss(client, clean_queries[0], limit=6),
                _fallback_wikipedia(client, clean_queries[0], limit=3),
            )
            deduped = (rss_hits + wiki_hits)[:max_total_sources]

        return await _scrape_and_format(client, deduped)
