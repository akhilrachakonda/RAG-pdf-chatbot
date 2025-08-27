
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

VECTOR_DIR = os.getenv("VECTOR_DIR", "data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

class ChromaRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DIR, settings=Settings())
        self.col = self.client.get_collection("pdf_chunks")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def query(self, question: str):
        qvec = self.model.encode([question], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=qvec, n_results=TOP_K, include=["documents", "metadatas", "distances"])
        docs = []
        for i in range(len(res["ids"][0])):
            docs.append({
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "score": 1 - res["distances"][0][i] if res.get("distances") else None
            })
        return docs
