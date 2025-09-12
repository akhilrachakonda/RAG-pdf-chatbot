import os
import json
import httpx
import streamlit as st
import subprocess
from pathlib import Path

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("📄🔎 RAG PDF Chatbot")
st.caption("FastAPI backend + Chroma + SentenceTransformers + OpenAI (optional)")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base", API_BASE)
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1)
    st.markdown("---")
    st.markdown("**Backend health:**")
    try:
        r = httpx.get(f"{api_base}/healthz", timeout=5)
        ok = r.json().get("ok", False)
        st.success("Healthy" if ok else "Unhealthy")
    except Exception as e:
        st.error(f"Error: {e}")

data_dir = Path("data/pdfs")
data_dir.mkdir(parents=True, exist_ok=True)

with st.expander("Upload PDFs and (re)ingest", expanded=False):
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            out_path = data_dir / f.name
            with open(out_path, "wb") as fh:
                fh.write(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to {data_dir}")
    if st.button("Rebuild index (ingest.py)"):
        with st.spinner("Running ingestion..."):
            try:
                # Run using the same interpreter
                res = subprocess.run([os.environ.get("PYTHON", "python"), "ingest.py"], capture_output=True, text=True, timeout=600)
                if res.returncode == 0:
                    st.success("Ingestion complete. Vector store updated.")
                else:
                    st.error(f"Ingestion failed (code {res.returncode}).\n\nStdout:\n{res.stdout}\n\nStderr:\n{res.stderr}")
            except Exception as e:
                st.error(f"Error running ingest: {e}")

q = st.text_area("Ask a question about your PDFs:", height=120, placeholder="e.g., Summarize section 2 and cite pages.")
file_options = ["All files"] + sorted([p.name for p in data_dir.glob("*.pdf")])
selected = st.selectbox("Restrict to file (optional)", options=file_options)

if st.button("Ask", type="primary") and q.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            payload = {"question": q, "k": int(top_k)}
            if selected and selected != "All files":
                payload["sources"] = [selected]
            r = httpx.post(f"{api_base}/chat", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            st.subheader("Answer")
            st.write(data.get("answer", ""))
            st.subheader("Citations")
            cits = data.get("citations", [])
            if not cits:
                st.info("No citations returned.")
            else:
                for i, c in enumerate(cits, 1):
                    score = c.get('score')
                    score_txt = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
                    st.write(f"[{i}] {c.get('source')} p.{c.get('page')}{score_txt}")
            with st.expander("Retrieved Chunks"):
                st.json(data.get("retrieved", []))
        except Exception as e:
            st.error(f"Request failed: {e}")
