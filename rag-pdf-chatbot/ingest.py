import os
import glob
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import hashlib

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def load_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(pages)
    for d in docs:
        text = (d.page_content or "").strip()
        meta = d.metadata or {}
        source = os.path.basename(meta.get("source") or pdf_path)
        page = meta.get("page") or meta.get("page_number")
        # Coerce metadata to Chroma-compatible scalar types (no None)
        try:
            page_i = int(page) if page is not None else 0
        except Exception:
            page_i = 0
        yield {"text": text, "source": str(source), "page": page_i}

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=VECTOR_DIR)
    col = client.get_or_create_collection(name="pdf_chunks", metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    texts, metas, ids = [], [], []
    for pdf in pdfs:
        for c in load_and_split(pdf):
            if not c["text"]:
                continue
            did = sha16(f"{c['source']}|{c['page']}|{c['text'][:200]}")
            ids.append(did)
            texts.append(c["text"])
            metas.append({"source": str(c["source"]), "page": int(c["page"])})

    print(f"Encoding {len(texts)} chunks with {EMBEDDING_MODEL} ...")
    embs = embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).tolist()

    # Best-effort upsert: delete existing ids then add
    for i in range(0, len(ids), 1000):
        try:
            col.delete(ids=ids[i:i+1000])
        except Exception:
            pass

    for i in tqdm(range(0, len(texts), 256), desc="Adding to Chroma"):
        col.add(
            ids=ids[i:i+256],
            documents=texts[i:i+256],
            embeddings=embs[i:i+256],
            metadatas=metas[i:i+256],
        )

    print("Done. Vector store at", VECTOR_DIR)

if __name__ == "__main__":
    main()
