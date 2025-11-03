# app.py
import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Environment / Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")  # fast & solid

DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "chroma_offline")
Path(DATA_DIR).mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# RAG stack (offline): loaders, splitters, embeddings, vector store
# ──────────────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    emb = get_embeddings()
    os.makedirs(INDEX_DIR, exist_ok=True)
    return Chroma(persist_directory=INDEX_DIR, embedding_function=emb)

def clear_index():
    import shutil
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)
    # reset cached instance
    get_vectorstore.clear()
    return get_vectorstore()

def load_pdf_pages(path: str) -> List[Document]:
    """
    Try PyMuPDF first (handles many tricky or image-heavy PDFs), fallback to PyPDF.
    Filter out empty pages.
    """
    # 1) Try PyMuPDF
    try:
        pages = PyMuPDFLoader(path).load()
        pages = [p for p in pages if (p.page_content or "").strip()]
        if pages:
            return pages
    except Exception:
        pass
    # 2) Fallback to PyPDF
    pages = PyPDFLoader(path).load()
    pages = [p for p in pages if (p.page_content or "").strip()]
    return pages

def add_pdfs_to_index(files) -> int:
    """Save uploaded PDFs, load (robust), split, and add to Chroma. Returns #chunks."""
    docs: List[Document] = []
    for file in files:
        tmp_path = os.path.join(DATA_DIR, f"upload_{file.name}")
        with open(tmp_path, "wb") as f:
            f.write(file.read())

        try:
            pages = load_pdf_pages(tmp_path)
        except Exception as e:
            st.error(f"Failed to load {file.name}: {e}")
            continue

        if not pages:
            st.warning(f"{file.name}: 0 extractable text pages (might be scanned images).")
            continue

        # tag metadata & quick stats
        total_chars = 0
        for p in pages:
            p.metadata = (p.metadata or {}) | {"source": tmp_path}
            total_chars += len(p.page_content or "")
        st.info(f"{file.name}: {len(pages)} text pages, ~{total_chars} chars")

        docs.extend(pages)

    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    st.info(f"Split into {len(chunks)} chunks total")

    vs = get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)

def retrieve(query: str, k: int = 6) -> List[Dict]:
    """
    Use MMR to diversify results across documents.
    """
    vs = get_vectorstore()
    hits = vs.max_marginal_relevance_search(query, k=k, fetch_k=24, lambda_mult=0.3)
    out = []
    for h in hits:
        out.append({
            "content": h.page_content,
            "source": (h.metadata or {}).get("source", "local"),
        })
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Ollama chat (robust) – uses /api/chat, handles non-JSON responses gracefully
# ──────────────────────────────────────────────────────────────────────────────
def ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_MODEL, timeout_s: int = 180) -> str:
    """
    Call Ollama's /api/chat (non-stream).
    messages = [{"role":"system"/"user"/"assistant", "content":"..."}]
    """
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout_s
    )
    r.raise_for_status()
    # Prefer JSON, but some setups may return text
    if "application/json" in r.headers.get("Content-Type", ""):
        data = r.json()
        # Typical response shape has "message": {"content": "..."}
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "").strip()
        return data.get("response", "").strip()
    return r.text

def synthesize_answer(question: str, retrieved: List[Dict], word_budget: int = 300) -> str:
    if not retrieved:
        return "No retrieved context yet. Upload PDFs and try again."
    context = "\n\n".join([f"[{i}] ({r['source']}) {r['content'][:1200]}" for i, r in enumerate(retrieved)])
    prompt = f"""You are a precise research assistant. Use ONLY the context to answer.

Question: {question}
Context:
{context}

Return sections:
1) Key findings
2) Comparison table (if applicable)
3) Cited references [use bracket index numbers]
4) Next actions for the user (practical steps)
Keep it < {word_budget} words."""
    return ollama_chat([{"role": "user", "content": prompt}], model=OLLAMA_MODEL)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoResearcher (Local)", page_icon="🤖", layout="wide")
st.title("AutoResearcher: Local RAG with Ollama + Chroma")
st.caption("Upload PDFs → local vector DB (Chroma + SentenceTransformers) → retrieve → synthesize with a local LLM via Ollama (/api/chat)")

with st.sidebar:
    st.subheader("Ollama")
    st.code(f"HOST:  {OLLAMA_HOST}\nMODEL: {OLLAMA_MODEL}", language="bash")
    if st.button("Test Ollama connection"):
        try:
            rv = requests.get(f"{OLLAMA_HOST}/api/version", timeout=10)
            ok = rv.status_code
            ping = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": [{"role":"user","content":"Say 'pong'."}], "stream": False},
                timeout=20
            )
            st.success(f"Ollama OK (version status {ok}, chat {ping.status_code})")
        except Exception as e:
            st.error(f"Ollama not reachable: {e}")

    st.write("---")
    st.subheader("Index Controls")
    if st.button("Clear & Recreate Index"):
        clear_index()
        st.success("Index cleared. Re-upload PDFs to rebuild.")

    st.write("---")
    files = st.file_uploader("Upload PDFs to index", type=["pdf"], accept_multiple_files=True)
    if files:
        with st.spinner("Indexing PDFs..."):
            n = add_pdfs_to_index(files)
        if n > 0:
            st.success(f"Indexed {n} chunks. Ready to ask questions!")

st.write("### Ask a question")
q = st.text_input("e.g., Summarize YOLOv11 vs YOLOv8 differences for robotics vision on embedded devices")
col1, col2 = st.columns([1,1], vertical_alignment="bottom")
with col1:
    detail = st.radio("Answer length", ["Concise", "Detailed"], index=0, horizontal=True)
with col2:
    word_budget = 350 if detail == "Concise" else 700
    run = st.button("Run", type="primary")

if run and q.strip():
    with st.status("🔍 Retrieving relevant chunks...", expanded=False) as status:
        retrieved = retrieve(q, k=6)
        status.update(label=f"Found {len(retrieved)} chunks. 🧠 Generating with {OLLAMA_MODEL}...", state="running")
        try:
            t0 = time.time()
            ans = synthesize_answer(q, retrieved, word_budget=word_budget)
            dt = time.time() - t0
            status.update(label=f"Done in {dt:.1f}s", state="complete")
        except Exception as e:
            st.error(f"Generation failed: {e}")
            ans = ""

    if ans:
        st.markdown("### Answer")
        st.write(ans)

        # Index stats (whole DB)
        with st.expander("Index stats (all documents)"):
            try:
                vs = get_vectorstore()
                raw = vs._collection.get(include=["metadatas"], limit=5000)
                metas = raw.get("metadatas", []) or []
                files_all = [os.path.basename((m or {}).get("source","unknown")) for m in metas]
                counts_all = Counter(files_all)
                if counts_all:
                    for name, n in counts_all.most_common():
                        st.markdown(f"- **{name}** — {n} chunks in index")
                else:
                    st.warning("Index is empty. Upload PDFs and re-index.")
            except Exception as e:
                st.warning(f"Could not read index stats: {e}")

        # Sources / Chunks (for this answer)
        with st.expander("Sources / Chunks (this answer)"):
            filenames = [os.path.basename(r['source']) for r in retrieved]
            counts = Counter(filenames)
            st.markdown("**Files contributing to the answer:**")
            for name, n in counts.most_common():
                st.markdown(f"- **{name}** — {n} chunk{'s' if n > 1 else ''}")

            show_snips = st.checkbox("Show retrieved text snippets", value=False)
            if show_snips:
                st.write("---")
                for i, r in enumerate(retrieved):
                    fname = os.path.basename(r['source'])
                    snippet = (r['content'] or "")[:600].replace("\n", " ")
                    st.markdown(f"**[{i}] {fname}**")
                    st.code(snippet + ("..." if len(r.get('content','')) > 600 else ""), language="markdown")

else:
    st.info("Upload a few PDFs in the sidebar, then ask a question.")