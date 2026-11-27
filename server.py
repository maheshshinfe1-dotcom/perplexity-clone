import asyncio
import io
import json
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

from search_engine import search_and_scrape, multi_search_and_scrape, search_images
from threads_db import (
    init_db,
    create_thread,
    thread_exists,
    save_turn,
    list_threads,
    get_thread,
    delete_thread,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOC_CHARS = 24000
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts"}


def load_api_key() -> str:
    """Loads GEMINI_API_KEY from local .env first, falling back to cloud environment variables."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val

    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()

    raise RuntimeError("GEMINI_API_KEY not found. Add it to perplexity_clone/.env")


API_KEY = load_api_key()
ai_client = genai.Client(api_key=API_KEY)
init_db()

app = FastAPI(title="Perplexity Clone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AttachedDocument(BaseModel):
    filename: str
    content: str
    pages: int | None = None


class SearchRequest(BaseModel):
    query: str
    history: list[dict] = []
    thread_id: str | None = None
    focus: str = "web"
    pro_mode: bool = False
    document: AttachedDocument | None = None


FOCUS_STYLE_PROMPTS = {
    "web": "Give a clear, well-structured, direct answer suitable for general web research.",
    "academic": (
        "Adopt a rigorous academic research tone. Highlight methodologies, empirical findings, "
        "consensus vs. open questions, and key limitations."
    ),
    "code": (
        "Adopt a Senior Software Engineer tone. Prioritize architecture, clean fenced code examples, "
        "performance trade-offs, and official documentation references."
    ),
    "news": (
        "Adopt a crisp news-briefing style. Lead with the latest developments, key dates, "
        "organizations involved, and why it matters now."
    ),
}

NO_AFC_CONFIG = types.AutomaticFunctionCallingConfig(disable=True)


def get_flash_models() -> list[str]:
    """Discovers valid text-generation Flash models available on your API key."""
    excluded_keywords = ("image", "audio", "tts", "live", "native", "embedding", "robotics", "computer")
    candidates = ["gemini-3.8-flash"]
    try:
        for m in ai_client.models.list():
            name = m.name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            if "flash" in name.lower() and not any(bad in name.lower() for bad in excluded_keywords):
                if name not in candidates:
                    candidates.append(name)
    except Exception as e:
        print(f"⚠️ Could not list models (check API key): {type(e).__name__}: {e}")

    for fallback in ("gemini-2.5-flash", "gemini-2.0-flash"):
        if fallback not in candidates:
            candidates.append(fallback)

    return candidates[:5]


FLASH_MODELS = get_flash_models()
print(f"🔑 API Key Loaded: {API_KEY[:6]}...{API_KEY[-4:]}")
print(f"🤖 Active Flash Failover Chain: {FLASH_MODELS}")


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def extract_document_text(raw_bytes: bytes, filename: str) -> tuple[str, int | None]:
    """Extracts readable text from PDF or plain-text documents."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        page_count = len(reader.pages)
        chunks = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                chunks.append(f"[Page {idx}]\n{page_text}")
        return "\n\n".join(chunks).strip(), page_count

    text = raw_bytes.decode("utf-8", errors="ignore").strip()
    return text, None


def plan_pro_subqueries(
    query: str,
    focus: str,
    history: list[dict],
    doc: AttachedDocument | None = None,
) -> list[str]:
    """Decomposes a complex user question into 3 targeted parallel search queries."""
    history_snippet = ""
    if history:
        last_turns = history[-2:]
        history_snippet = "Recent context:\n" + "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')[:200]}" for m in last_turns
        ) + "\n\n"

    doc_snippet = ""
    if doc:
        doc_snippet = (
            f"User also attached document '{doc.filename}'. Excerpt:\n"
            f"{doc.content[:600]}\n\n"
        )

    prompt = (
        f"{history_snippet}"
        f"{doc_snippet}"
        f"User question: {query}\n"
        f"Focus mode: {focus}\n\n"
        "You are a search query planner for an AI research engine. "
        "Break the user's question into exactly 3 distinct, concise web search queries "
        "that together cover different angles, facts, comparisons, or recent updates needed "
        "to answer thoroughly. Keep each query under 10 words.\n"
        "Return ONLY a JSON array of 3 strings."
    )
    for model_name in FLASH_MODELS:
        try:
            resp = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    automatic_function_calling=NO_AFC_CONFIG,
                ),
            )
            items = json.loads(resp.text)
            if isinstance(items, list):
                cleaned = [str(q).strip() for q in items if str(q).strip()][:3]
                if cleaned:
                    print(f"⚡ Pro Search planned queries: {cleaned}")
                    return cleaned
        except Exception:
            continue
    return [query]


def build_grounded_prompt(
    query: str,
    sources: list[dict],
    history: list[dict],
    focus: str,
    pro_mode: bool = False,
    has_document: bool = False,
) -> str:
    """Builds a strict citation-grounded prompt combining attached files and live web sources."""
    context_blocks = []
    for src in sources:
        url_line = f"URL: {src['url']}\n" if src.get("url") else "Type: User-uploaded document\n"
        context_blocks.append(
            f"Source [{src['id']}]: {src['title']} ({src['domain']})\n"
            f"{url_line}"
            f"Content:\n{src['content']}\n"
        )
    sources_text = "\n---\n".join(context_blocks)

    history_text = ""
    if history:
        recent = history[-4:]
        lines = [f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}" for m in recent]
        history_text = "Previous conversation context:\n" + "\n".join(lines) + "\n\n"

    style_guidance = FOCUS_STYLE_PROMPTS.get(focus, FOCUS_STYLE_PROMPTS["web"])
    depth_guidance = (
        "PRO RESEARCH MODE: Cross-reference multiple sources, organize your answer with clear "
        "Markdown subheadings (##), compare competing viewpoints or benchmarks when present, "
        "and include concrete specifics (numbers, dates, trade-offs)."
        if pro_mode
        else "Keep the response concise, scannable, and directly on-point."
    )
    doc_guidance = (
        "ATTACHED DOCUMENT MODE: Source [1] is the user's uploaded file. Prioritize answering directly "
        "from Source [1] when the question refers to the document, and cross-check or enrich with "
        "the live web sources ([2]+) whenever helpful. Always cite [1] for facts from the file."
        if has_document
        else ""
    )

    return f"""You are a precise, real-time AI Answer Engine (like Perplexity AI).
FOCUS MODE ({focus.upper()}): {style_guidance}
{depth_guidance}
{doc_guidance}

{history_text}AVAILABLE SOURCES:
{sources_text}

USER QUESTION: {query}

STRICT INSTRUCTIONS:
1. Answer clearly using Markdown headings or bullet points when helpful.
2. Base factual claims strictly on the AVAILABLE SOURCES provided above.
3. Add inline numerical citations like [1] or [2][4] immediately after every factual claim.
4. Never fabricate URLs, statistics, or source numbers beyond the sources provided.
5. If the sources do not cover a detail, state that plainly.
"""


def generate_followups(query: str, answer: str) -> list[str]:
    """Generates 3 short related follow-up questions in structured JSON."""
    prompt = (
        f"User asked: {query}\n\n"
        f"Answer summary: {answer[:800]}\n\n"
        "Generate exactly 3 concise, high-value follow-up questions the user would likely ask next. "
        "Return ONLY a JSON array of 3 strings."
    )
    for model_name in FLASH_MODELS:
        try:
            resp = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                    automatic_function_calling=NO_AFC_CONFIG,
                ),
            )
            items = json.loads(resp.text)
            if isinstance(items, list):
                return [str(q).strip() for q in items[:3]]
        except Exception:
            continue
    return []


@app.get("/")
async def serve_frontend():
    index_file = Path("index.html")
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "Perplexity Clone API is live", "models": FLASH_MODELS, "docs": "/docs"}


@app.head("/", include_in_schema=False)
async def health_head():
    return Response(status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "document.txt").name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {allowed}")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 10 MB size limit.")

    try:
        extracted_text, page_count = await asyncio.to_thread(extract_document_text, raw_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {type(e).__name__}: {e}")

    if not extracted_text:
        raise HTTPException(
            status_code=400,
            detail="No selectable text found in this file (scanned image-only PDFs are not supported yet).",
        )

    truncated = len(extracted_text) > MAX_DOC_CHARS
    trimmed_text = extracted_text[:MAX_DOC_CHARS]
    print(f"📄 Uploaded '{filename}' ({len(trimmed_text)} chars, pages={page_count}, truncated={truncated})")

    return {
        "filename": filename,
        "pages": page_count,
        "chars": len(trimmed_text),
        "truncated": truncated,
        "content": trimmed_text,
    }


@app.get("/api/threads")
async def api_list_threads():
    return {"threads": await asyncio.to_thread(list_threads)}


@app.get("/api/threads/{thread_id}")
async def api_get_thread(thread_id: str):
    thread = await asyncio.to_thread(get_thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.delete("/api/threads/{thread_id}")
async def api_delete_thread(thread_id: str):
    deleted = await asyncio.to_thread(delete_thread, thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"deleted": True, "thread_id": thread_id}


@app.post("/api/search")
async def stream_search(req: SearchRequest):
    async def event_generator():
        query = req.query.strip()
        focus = req.focus if req.focus in FOCUS_STYLE_PROMPTS else "web"
        pro_mode = bool(req.pro_mode)
        doc = req.document if (req.document and req.document.content.strip()) else None

        if not query:
            yield sse_event({"type": "error", "text": "Query cannot be empty."})
            yield sse_event({"type": "done"})
            return

        focus_label = {
            "web": "the live web",
            "academic": "academic sources",
            "code": "developer docs",
            "news": "live news",
        }[focus]

        # Launch image search concurrently so it adds zero latency to web scraping
        image_task = asyncio.create_task(search_images(query, max_results=8))

        # Step 1: Standard Search OR Pro Multi-Step Query Decomposition
        if pro_mode:
            yield sse_event({"type": "status", "text": "Pro Search: planning multi-step research..."})
            subqueries = await asyncio.to_thread(plan_pro_subqueries, query, focus, req.history, doc)
            yield sse_event({"type": "pro_plan", "queries": subqueries})
            yield sse_event({
                "type": "status",
                "text": f"Pro Search: querying {len(subqueries)} parallel angles across {focus_label}...",
            })
            web_sources = await multi_search_and_scrape(
                subqueries,
                max_total_sources=7 if doc else 8,
                focus=focus,
            )
        else:
            status_msg = (
                f"Reading {doc.filename} + searching {focus_label}..."
                if doc
                else f"Searching {focus_label}..."
            )
            yield sse_event({"type": "status", "text": status_msg})
            web_sources = await search_and_scrape(
                query,
                max_results=4 if doc else 5,
                focus=focus,
            )

        # Merge uploaded document as Source [1] and re-index live web sources as [2]+
        sources: list[dict] = []
        if doc:
            doc_domain = f"Uploaded PDF ({doc.pages}p)" if doc.pages else "Uploaded File"
            sources.append({
                "id": 1,
                "title": f"📄 {doc.filename}",
                "url": "",
                "domain": doc_domain,
                "snippet": doc.content[:240],
                "content": doc.content[:MAX_DOC_CHARS],
            })

        start_idx = len(sources) + 1
        for offset, ws in enumerate(web_sources):
            sources.append({**ws, "id": start_idx + offset})

        if not sources:
            image_task.cancel()
            yield sse_event({"type": "error", "text": "No live search results found. Please retry."})
            yield sse_event({"type": "done"})
            return

        # Step 2: Create or Reuse Persistent SQLite Thread
        active_thread_id = req.thread_id
        if not active_thread_id or not await asyncio.to_thread(thread_exists, active_thread_id):
            created = await asyncio.to_thread(create_thread, query, focus)
            active_thread_id = created["id"]
            yield sse_event({
                "type": "thread",
                "thread_id": created["id"],
                "title": created["title"],
                "focus": created["focus"],
            })

        # Step 3: Push Source Cards + Live Visual Media Strip
        public_sources = [
            {"id": s["id"], "title": s["title"], "url": s["url"], "domain": s["domain"]}
            for s in sources
        ]
        yield sse_event({"type": "sources", "sources": public_sources})

        try:
            images = await asyncio.wait_for(image_task, timeout=4.0)
        except Exception:
            images = []
        if images:
            yield sse_event({"type": "images", "images": images})

        yield sse_event({
            "type": "status",
            "text": f"Synthesizing {'deep Pro' if pro_mode else ''} brief from {len(sources)} sources...".replace("  ", " "),
        })

        # Step 4: Stream Citation-Grounded Answer
        prompt = build_grounded_prompt(
            query,
            sources,
            req.history,
            focus,
            pro_mode=pro_mode,
            has_document=bool(doc),
        )
        full_answer = ""
        streamed_ok = False
        last_error = ""

        for model_name in FLASH_MODELS:
            try:
                mode_tag = f"{focus}|PRO" if pro_mode else focus
                if doc:
                    mode_tag += "|DOC"
                print(f"🤖 Synthesizing [{mode_tag}] with model: {model_name}...")
                stream = ai_client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.25,
                        automatic_function_calling=NO_AFC_CONFIG,
                    ),
                )
                for chunk in stream:
                    if chunk.text:
                        full_answer += chunk.text
                        yield sse_event({"type": "token", "text": chunk.text})
                        await asyncio.sleep(0)
                streamed_ok = True
                print(f"✅ Synthesis complete using {model_name}")
                break
            except (ServerError, APIError) as e:
                last_error = f"HTTP {e.code}: {e.message}"
                print(f"❌ Model '{model_name}' failed -> {last_error}")
                if e.code in (400, 401, 403) and "API_KEY" in str(e.message).upper():
                    break
                if full_answer:
                    break
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"❌ Unexpected error on '{model_name}' -> {last_error}")
                break

        if not streamed_ok and not full_answer:
            yield sse_event({"type": "error", "text": f"AI synthesis failed ({last_error or 'all models busy'})."})
            yield sse_event({"type": "done"})
            return

        # Step 5: Generate Related Follow-Up Chips
        followups = await asyncio.to_thread(generate_followups, query, full_answer)
        if followups:
            yield sse_event({"type": "followups", "questions": followups})

        # Step 6: Persist Completed Turn to SQLite
        await asyncio.to_thread(
            save_turn,
            active_thread_id,
            query,
            focus,
            public_sources,
            full_answer,
            followups,
        )

        yield sse_event({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run("server:app", host=host, port=port, reload=not bool(os.getenv("PORT")))
