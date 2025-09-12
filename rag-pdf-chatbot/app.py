import os
import time, statistics
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import ChromaRetriever

load_dotenv()

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

app = FastAPI(title="RAG PDF Chatbot")

class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = None
    sources: Optional[List[str]] = None  # optional filename filter

_retriever: Optional[ChromaRetriever] = None

def get_retriever() -> ChromaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever

def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source") or "unknown"
        page = c.get("page")
        head = f"[{i}] {src}" + (f" p.{page}" if page is not None else "")
        text = (c.get("text") or "").strip()
        parts.append(f"{head}\n{text}")
    return "\n\n---\n\n".join(parts)

# --- minimal in-memory metrics ---
METRICS: Dict[str, Any] = {
    "requests": 0,
    "retrieval_ms": [],
    "gen_ms": [],
    "tokens_in": 0,
    "tokens_out": 0,
}

def _push(arr, val, cap: int = 500):
    try:
        arr.append(val)
        if len(arr) > cap:
            del arr[: len(arr) - cap]
    except Exception:
        pass

def _pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = int((p / 100.0) * (len(s) - 1))
    return s[k]

def llm_answer(question: str, context: str):
    provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").lower()
    model = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("MAX_TOKENS", "350"))

    system = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "Be concise and include inline citations like (source: file p.X). "
        "If the answer cannot be found in the context, say you don't know."
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"

    if not _OPENAI_AVAILABLE:
        return (
            "[Mock answer]\nOpenAI client library unavailable; returning a template answer using only retrieved context.",
            None,
            0,
        )

    try:
        t0 = time.time()
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            if not api_key:
                return (
                    "[Mock answer]\nOPENROUTER_API_KEY not set; please configure .env to use OpenRouter.",
                    None,
                    0,
                )
            headers = {}
            ref = os.getenv("OPENROUTER_HTTP_REFERER")
            title = os.getenv("OPENROUTER_X_TITLE")
            if ref:
                headers["HTTP-Referer"] = ref
            if title:
                headers["X-Title"] = title
            client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
            def _call(m):
                return client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            try:
                comp = _call(model or "deepseek/deepseek-r1")
            except Exception as e:
                msg = str(e)
                if "not a valid model" in msg or "model not found" in msg or "is not valid" in msg:
                    fallback = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-r1")
                    comp = _call(fallback)
                else:
                    raise
            usage = getattr(comp, "usage", None)
            return (comp.choices[0].message.content, usage, int((time.time() - t0) * 1000))

        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if not api_key:
                return (
                    "[Mock answer]\nDEEPSEEK_API_KEY not set; please configure .env to use DeepSeek.",
                    None,
                    0,
                )
            client = OpenAI(api_key=api_key, base_url=base_url)
            # Prefer Responses API (works with deepseek-reasoner); fallback to chat.completions
            try:
                resp = client.responses.create(
                    model=model,
                    input=f"System:\n{system}\n\nUser:\n{user}",
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                # openai>=1.38 provides output_text convenience
                text = getattr(resp, "output_text", None) or resp.output[0].content[0].text
                return (text, None, int((time.time() - t0) * 1000))
            except Exception:
                comp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = getattr(comp, "usage", None)
                return (comp.choices[0].message.content, usage, int((time.time() - t0) * 1000))

        # Default: OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "[Mock answer]\nOPENAI_API_KEY not set; configure .env or switch provider.",
                None,
                0,
            )
        client = OpenAI(api_key=api_key)
        def _call_default(m):
            return client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        try:
            comp = _call_default(model)
        except Exception as e:
            msg = str(e)
            if "not a valid model" in msg or "model not found" in msg or "is not valid" in msg:
                fallback = os.getenv("FALLBACK_MODEL", "gpt-4.1-mini")
                comp = _call_default(fallback)
            else:
                raise
        usage = getattr(comp, "usage", None)
        return (comp.choices[0].message.content, usage, int((time.time() - t0) * 1000))

    except Exception as e:
        return (f"[LLM error] {type(e).__name__}: {e}", None, 0)

@app.get("/healthz")
def healthz():
    try:
        r = get_retriever()
        cnt = r.collection.count()
        return {"ok": True, "vectors": cnt}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ingest")
def ingest():
    return {"message": "Run `python ingest.py` to (re)build the vector store from data/pdfs."}

@app.post("/chat")
def chat(req: ChatRequest):
    r = get_retriever()
    t0 = time.time()
    docs = r.query(req.question, top_k=req.k, sources=req.sources)
    retrieval_ms = int((time.time() - t0) * 1000)

    context = _format_context(docs)

    answer, usage, gen_ms = llm_answer(req.question, context)

    METRICS["requests"] += 1
    _push(METRICS["retrieval_ms"], retrieval_ms)
    _push(METRICS["gen_ms"], gen_ms)

    # Aggregate token usage if available
    if usage:
        try:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if pt is None and isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
            METRICS["tokens_in"] += int(pt or 0)
            METRICS["tokens_out"] += int(ct or 0)
        except Exception:
            pass

    # Normalize and dedupe citations for a cleaner UI
    seen = set()
    citations = []
    for d in docs:
        key = (d.get("source"), d.get("page"))
        if key in seen:
            continue
        seen.add(key)
        score = d.get("score")
        # If upstream returned negative values (e.g., raw distances), make a bounded similarity
        if isinstance(score, (int, float)) and score < 0:
            try:
                score = 1.0 / (1.0 + abs(float(score)))
            except Exception:
                pass
        citations.append({"source": d.get("source"), "page": d.get("page"), "score": score})
    return {
        "answer": answer,
        "citations": citations,
        "retrieved": docs,
        "timing_ms": {"retrieval": retrieval_ms, "generation": gen_ms},
        "usage": usage if usage else {},
    }

@app.get("/debug/config")
def debug_config():
    prov = os.getenv("LLM_PROVIDER")
    model = os.getenv("MODEL_NAME")
    base = (
        os.getenv("OPENROUTER_BASE_URL")
        if (prov or "").lower() == "openrouter"
        else os.getenv("DEEPSEEK_BASE_URL")
    )
    key_present = bool(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    return {"provider": prov, "model": model, "base_url": base, "key_present": key_present}

@app.get("/metrics")
def metrics():
    return {
        "requests": METRICS["requests"],
        "retrieval_ms": {"p50": _pct(METRICS["retrieval_ms"], 50), "p95": _pct(METRICS["retrieval_ms"], 95)},
        "generation_ms": {"p50": _pct(METRICS["gen_ms"], 50), "p95": _pct(METRICS["gen_ms"], 95)},
        "tokens": {"in": METRICS["tokens_in"], "out": METRICS["tokens_out"]},
    }

@app.get("/sources")
def sources():
    try:
        r = get_retriever()
        # Pull all metadatas (bounded) and count by source
        all_md = r.collection.get(include=["metadatas"], limit=100000)
        counts: Dict[str, int] = {}
        for md in (all_md.get("metadatas") or [[]])[0]:
            src = (md or {}).get("source") or "unknown"
            counts[src] = counts.get(src, 0) + 1
        items = sorted(({"source": k, "chunks": v} for k, v in counts.items()), key=lambda x: x["chunks"], reverse=True)
        return {"total_vectors": r.collection.count(), "sources": items}
    except Exception as e:
        return {"error": str(e)}
