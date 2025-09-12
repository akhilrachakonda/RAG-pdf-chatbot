import os
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import hashlib

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

class ChromaRetriever:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "pdf_chunks"):
        self.persist_dir = persist_dir or os.getenv("VECTOR_DIR", "./data/chroma")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embed_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(self.embed_model_name)

        self.top_k = int(os.getenv("TOP_K", "5"))
        self.hybrid = os.getenv("HYBRID", "false").lower() == "true"
        self.rerank = os.getenv("RERANK", "false").lower() == "true"

        # Optional BM25 index for HYBRID
        self.bm25 = None
        if self.hybrid:
            if BM25Okapi is None:
                raise RuntimeError("HYBRID=true requires rank-bm25 (`pip install rank-bm25`).")
            all_docs = self.collection.get(include=["documents", "metadatas"], limit=100000)
            self._bm25_docs = all_docs.get("documents", []) or []
            self._bm25_metas = all_docs.get("metadatas", []) or []
            tokenized = [ (doc or "").split() for doc in self._bm25_docs ]
            self.bm25 = BM25Okapi(tokenized)

    def _embed(self, texts: List[str]):
        return self.embedder.encode(texts, normalize_embeddings=True)

    def query(self, question: str, top_k: Optional[int] = None, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k

        # Dense search
        q_emb = self._embed([question])[0].tolist()
        dense = self.collection.query(
            query_embeddings=[q_emb],
            n_results=max(k, 10),
            include=["documents", "metadatas", "distances"],
        )
        dense_docs = []
        for doc, meta, dist in zip(
            dense.get("documents", [[]])[0],
            dense.get("metadatas", [[]])[0],
            dense.get("distances", [[]])[0],
        ):
            if doc is None:
                continue
            # Convert distance to a non-negative similarity-like score for display
            d = float(dist)
            score = max(0.0, 1.0 - d)
            src = (meta or {}).get("source")
            page = (meta or {}).get("page")
            hid = hashlib.sha256(f"{src}|{page}|{doc}".encode("utf-8")).hexdigest()[:16]
            dense_docs.append({
                "text": doc,
                "source": src,
                "page": page,
                "score_dense": score,
                "id": hid,
            })
        candidates = dense_docs

        # Hybrid: RRF fuse dense + BM25
        if self.hybrid and self.bm25 is not None and getattr(self, "_bm25_docs", []):
            m = max(k, 10)
            tokenized = question.split()
            bm25_scores = self.bm25.get_scores(tokenized)
            top_idx = np.argsort(bm25_scores)[::-1][:m]
            bm25_docs = []
            for idx in top_idx:
                doc = self._bm25_docs[idx]
                meta = (self._bm25_metas[idx] or {})
                src = meta.get("source")
                page = meta.get("page")
                hid = hashlib.sha256(f"{src}|{page}|{doc}".encode("utf-8")).hexdigest()[:16]
                bm25_docs.append({
                    "text": doc,
                    "source": src,
                    "page": page,
                    "score_bm25": float(bm25_scores[idx]),
                    "id": hid,
                })

            def rrf(rank): return 1.0 / (60 + rank)
            r = defaultdict(float); id2doc = {}
            for i,d in enumerate(dense_docs): r[d["id"]] += rrf(i+1); id2doc[d["id"]] = d
            for i,d in enumerate(bm25_docs): r[d["id"]] += rrf(i+1); id2doc.setdefault(d["id"], d)

            fused = [{**id2doc[id_], "score_fused": float(score)} for id_, score in r.items()]
            candidates = sorted(fused, key=lambda x: x["score_fused"], reverse=True)[:k]

        # Optional rerank
        if self.rerank and candidates:
            from sentence_transformers import CrossEncoder
            model = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            ce = CrossEncoder(model)
            pairs = [[question, c["text"]] for c in candidates]
            scores = ce.predict(pairs).tolist()
            for c, s in zip(candidates, scores): c["score_rerank"] = float(s)
            candidates = sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:k]

        out = []
        for c in candidates[:k]:
            score = c.get("score_rerank") or c.get("score_fused") or c.get("score_dense") or c.get("score_bm25") or 0.0
            out.append({"text": c.get("text"), "source": c.get("source"), "page": c.get("page"), "score": float(score)})
        # Optional filename filter
        if sources:
            allow = {os.path.basename(s) for s in sources}
            out = [r for r in out if os.path.basename(r.get("source") or "") in allow]
        return out
