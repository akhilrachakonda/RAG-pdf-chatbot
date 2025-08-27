
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
VECTOR_DIR = os.getenv("VECTOR_DIR", "data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def embedder():
    model = SentenceTransformer(EMBEDDING_MODEL)
    def _embed(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()
    return _embed

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DIR, settings=Settings(allow_reset=True))
    if "pdf_chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("pdf_chunks")
    col = client.create_collection("pdf_chunks", metadata={"hnsw:space": "cosine"})

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    emb = embedder()

    texts, metadatas, ids = [], [], []
    counter = 0
    for fname in os.listdir(PDF_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, fname)
        pages = PyPDFLoader(path).load()
        chunks = splitter.split_documents(pages)
        for ch in chunks:
            texts.append(ch.page_content)
            metadatas.append({"source": fname, "page": ch.metadata.get("page", None)})
            ids.append(f"{fname}-{counter}")
            counter += 1

    print(f"Indexing {len(texts)} chunks from {PDF_DIR} ...")
    batch = 128
    for i in tqdm(range(0, len(texts), batch)):
        xs = texts[i:i+batch]
        vecs = emb(xs)
        col.add(documents=xs, embeddings=vecs, metadatas=metadatas[i:i+batch], ids=ids[i:i+batch])
    print("Done.")

if __name__ == "__main__":
    main()
