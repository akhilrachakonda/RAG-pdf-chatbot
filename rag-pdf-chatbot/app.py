
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import ChromaRetriever

load_dotenv()

try:
    from openai import OpenAI
    OPENAI = True
except Exception:
    OPENAI = False

class ChatRequest(BaseModel):
    question: str

app = FastAPI(title="RAG PDF Chatbot")

retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = ChromaRetriever()
    return retriever

def llm_answer(prompt: str, context: str) -> str:
    # If OpenAI key is available, use real generation. Otherwise, return a simple templated answer.
    api_key = os.getenv("OPENAI_API_KEY")
    if OPENAI and api_key:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers strictly from the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer from context only."}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    else:
        return f"[Mock Answer] Based on the retrieved context, here is a concise answer to: '{prompt}'.
(Provide a real API key for full generations.)"

@app.post("/ingest")
def ingest():
    # Trigger ingest via shell to keep example simple; users should run ingest.py beforehand in practice.
    return {"message": "Run `python ingest.py` to (re)build the vector store from data/pdfs."}

@app.post("/chat")
def chat(req: ChatRequest):
    r = get_retriever()
    docs = r.query(req.question)
    context = "\n\n".join([d["text"] for d in docs])
    answer = llm_answer(req.question, context)
    return {"answer": answer, "retrieved": docs}
