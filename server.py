import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

from search_engine import search_and_scrape
from threads_db import (
    init_db,
    create_thread,
    thread_exists,
    save_turn,
    list_threads,
    get_thread,
    delete_thread,
)


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


class SearchRequest(BaseModel):
    query: str
    history: list[dict] = []
    thread_id: str | None = None
    focus: str = "web"


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


def build_grounded_prompt(query: str, sources: list[dict], history: list[dict], focus: str) -> str:
    """Builds a strict citation-grounded prompt tailored to the active Focus Mode."""
    context_blocks = []
    for src in sources:
        context_blocks.append(
            f"Source [{src['id']}]: {src['title']} ({src['domain']})\n"
            f"URL: {src['url']}\n"
            f"Content: {src['content']}\n"
        )
    sources_text = "\n---\n".join(context_blocks)

    history_text = ""
    if history:
        recent = history[-4:]
        lines = [f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}" for m in recent]
        history_text = "Previous conversation context:\n" + "\n".join(lines) + "\n\n"

    style_guidance = FOCUS_STYLE_PROMPTS.get(focus, FOCUS_STYLE_PROMPTS["web"])

    return f"""You are a precise, real-time AI Answer Engine (like Perplexity AI).
FOCUS MODE ({focus.upper()}): {style_guidance}

{history_text}LIVE WEB SOURCES:
{sources_text}

USER QUESTION: {query}

STRICT INSTRUCTIONS:
1. Answer clearly using Markdown headings or bullet points when helpful.
2. Base factual claims strictly on the LIVE WEB SOURCES provided above.
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ==========================================
# 📚 THREAD LIBRARY REST ENDPOINTS
# ==========================================
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


# ==========================================
# ⚡ LIVE STREAMING SEARCH ENDPOINT (SSE)
# ==========================================
@app.post("/api/search")
async def stream_search(req: SearchRequest):
    async def event_generator():
        query = req.query.strip()
        focus = req.focus if req.focus in FOCUS_STYLE_PROMPTS else "web"

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
        yield sse_event({"type": "status", "text": f"Searching {focus_label}..."})
        sources = await search_and_scrape(query, max_results=5, focus=focus)

        if not sources:
            yield sse_event({"type": "error", "text": "No live search results found. Please retry."})
            yield sse_event({"type": "done"})
            return

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

        public_sources = [
            {"id": s["id"], "title": s["title"], "url": s["url"], "domain": s["domain"]}
            for s in sources
        ]
        yield sse_event({"type": "sources", "sources": public_sources})
        yield sse_event({"type": "status", "text": f"Reading {len(sources)} sources & synthesizing..."})

        prompt = build_grounded_prompt(query, sources, req.history, focus)
        full_answer = ""
        streamed_ok = False
        last_error = ""

        for model_name in FLASH_MODELS:
            try:
                print(f"🤖 Synthesizing [{focus}] with model: {model_name}...")
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

        followups = await asyncio.to_thread(generate_followups, query, full_answer)
        if followups:
            yield sse_event({"type": "followups", "questions": followups})

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