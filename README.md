
# RAG PDF Chatbot (FastAPI + Chroma + LangChain)

A minimal Retrieval-Augmented Generation service to chat with your PDFs.

## Features
- Ingest PDFs -> split -> embed -> store in **ChromaDB**.
- **/ingest** endpoint to index PDFs from a local folder.
- **/chat** endpoint that takes a user question and returns a grounded answer with retrieved chunks.
- Pluggable LLM provider (OpenAI by default, but you can implement others).
- Clear components: `ingest.py`, `retriever.py`, `app.py`.

## Tech
- FastAPI, Uvicorn
- LangChain, ChromaDB
- Sentence-Transformers for embeddings (default: `all-MiniLM-L6-v2`)
- OpenAI (optional for LLM calls) — bring your own key

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Add keys
cp .env.example .env
# edit .env (OPENAI_API_KEY=... optional; if not provided, will return retrieved chunks w/o generation)

# 2) Put PDFs
mkdir -p data/pdfs
# add your PDF files into data/pdfs

# 3) Ingest
python ingest.py

# 4) Run API
uvicorn app:app --reload --port 8000
```

### Endpoints
- `POST /ingest` -> (re)builds the Chroma index from `data/pdfs`.
- `POST /chat` -> body: `{ "question": "..." }`

### Example `curl`
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"What does the document say about X?"}'
```

## Folder Structure
```
rag-pdf-chatbot/
  app.py
  ingest.py
  retriever.py
  requirements.txt
  .env.example
  data/
    pdfs/   # your PDFs
    chroma/ # vector store (auto-created)
  README.md
```

## Notes
- If you don't set an LLM key, the service returns top-k retrieved chunks and a "mock" answer. This still demonstrates RAG.
- Replace OpenAI with any provider by implementing `llm_answer()` in `app.py`.
- Production tips are listed at the bottom of `app.py` as comments.
