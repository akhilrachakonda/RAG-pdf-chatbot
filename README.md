
# RAG PDF Chatbot (FastAPI + Chroma + LangChain)

A minimal Retrieval-Augmented Generation service to chat with your PDFs.

## Features
- Ingest PDFs -> split -> embed -> store in ChromaDB
- `POST /chat` returns grounded answers with citations
- Pluggable LLMs (OpenAI, OpenRouter, DeepSeek; or mock if unset)
- Clean modules: `ingest.py`, `retriever.py`, `app.py`

## Tech
- FastAPI, Uvicorn
- LangChain, ChromaDB
- Sentence-Transformers (`all-MiniLM-L6-v2` default)
- Optional LLM providers (bring your own key)

## Quickstart
```bash
cd rag-pdf-chatbot

python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Configure environment
cp .env.example .env
# edit .env (OPENAI_API_KEY=... optional; if not provided, service returns retrieved chunks with a mock answer)

# 2) Add PDFs
mkdir -p data/pdfs
# copy your files into data/pdfs

# 3) Build vector store
python ingest.py

# 4) Run the API
uvicorn app:app --reload --port 8000
```

### Endpoints
- `GET /healthz` – basic health and vector count
- `GET /metrics` – p50/p95 retrieval/gen latencies; token counts
- `POST /chat` – body: `{ "question": "...", "k": 5, "sources": ["optional.pdf"] }`

Example:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say about X?"}'
```

## Streamlit UI (optional)
```bash
cd rag-pdf-chatbot
streamlit run streamlit_app.py  # Upload PDFs, trigger ingestion, ask questions
```

## Docker
```bash
cd rag-pdf-chatbot
docker build -t rag-pdf-chatbot .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data":/app/data \
  --env-file .env \
  rag-pdf-chatbot
```

## Folder Structure
```
rag-pdf-chatbot/
  app.py
  ingest.py
  retriever.py
  streamlit_app.py
  eval.py
  requirements.txt
  .env.example
  data/
    pdfs/    # your PDFs (gitignored)
    chroma/  # vector store (gitignored)
  BENCHMARKS.md
  Dockerfile
```
See also: `rag-pdf-chatbot/BENCHMARKS.md`.

## GitHub Hygiene
- `.venv`, `data/pdfs/`, and `data/chroma/` are gitignored
- `.env` is gitignored; use `.env.example` to document variables
- Avoid committing large PDFs or vector stores; mount `data/` as a volume in Docker

## Notes
- If you don’t set an LLM key, the service returns top-k retrieved chunks with a simple mock answer to demonstrate RAG.
- Swap providers by changing `LLM_PROVIDER`/`MODEL_NAME` and related keys in `.env`.
- For advanced retrieval, toggle `HYBRID=true` and/or `RERANK=true` in `.env`.
