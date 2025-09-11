import os, pickle
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from flask import Flask, request, redirect, url_for, render_template, flash, session
from collections import defaultdict
from uuid import uuid4

# ---------------- Config ----------------
EMBED_MODEL   = "text-embedding-3-small"
GEN_MODEL     = "gpt-4o-mini"
CHUNK_CHARS   = 1200
CHUNK_OVERLAP = 200
TOP_K         = 5

# ---------------- Setup -----------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024

# Trim Jinja whitespace
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

client = OpenAI()

@dataclass
class Chunk:
    text: str
    source: str
    page: int  # 1-based

index = None
chunks: List[Chunk] = []
ready = False

# ---------------- Per-session chat histories ----------------
# Each visitor gets a session cookie 'sid'; we store their chat in-memory under that key.
histories = defaultdict(list)  # sid -> List[dict], each dict like {"role": "...", "text": "...", "sources": [...]}

@app.before_request
def ensure_session():
    if "sid" not in session:
        session["sid"] = str(uuid4())

def get_history() -> List[dict]:
    return histories[session["sid"]]

# ---------------- RAG helpers ----------------
def read_pdfs_to_chunks(docs_dir: str) -> List[Chunk]:
    cks: List[Chunk] = []
    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(docs_dir, fname)
        try:
            pdf = PdfReader(path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        for p_i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = " ".join(text.split())
            if not text:
                continue
            start = 0
            while start < len(text):
                end = start + CHUNK_CHARS
                chunk_txt = text[start:end]
                if not chunk_txt.strip():
                    break
                cks.append(Chunk(chunk_txt, fname, p_i + 1))
                start += CHUNK_CHARS - CHUNK_OVERLAP
    return cks

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    embs = []
    B = 128
    for i in tqdm(range(0, len(texts), B), desc="Embedding"):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
    X = np.array(embs, dtype="float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def build_or_load_index() -> Tuple[faiss.Index, List[Chunk]]:
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.pkl")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        ix = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return ix, meta
    cks = read_pdfs_to_chunks(DOCS_DIR)
    if not cks:
        dim = 1536
        ix = faiss.IndexFlatIP(dim)
        return ix, []
    X = embed_texts([c.text for c in cks])
    ix = faiss.IndexFlatIP(X.shape[1])
    ix.add(X)
    faiss.write_index(ix, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(cks, f)
    return ix, cks

def rebuild_index():
    global index, chunks, ready
    ready = False
    ix, cks = build_or_load_index()
    index = ix
    chunks = cks
    ready = True

def retrieve(ix: faiss.Index, cks: List[Chunk], query: str, k=TOP_K):
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    import numpy as np
    q = np.array(q, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-12)
    D, I = ix.search(q.reshape(1, -1), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(cks):
            continue
        hits.append((float(score), cks[idx]))
    return hits

SYS_PROMPT = """You are a helpful assistant answering ONLY from the provided context.
If the answer does not exist in the context, say you don't know.
Always include a bullet list of sources as [filename p. N] for anything you claim.
Be concise and accurate."""

def answer_question(query: str, contexts: List[Tuple[float, Chunk]]):
    context_blocks = []
    for score, ch in contexts:
        context_blocks.append(f"[{ch.source} p.{ch.page}] {ch.text}")
    context = "\n\n---\n\n".join(context_blocks)
    user_msg = f"Question: {query}\n\nContext:\n{context}\n"
    resp = client.responses.create(
        model=GEN_MODEL,
        input=[{"role":"system","content":SYS_PROMPT},{"role":"user","content":user_msg}],
        temperature=0.1,
    )
    answer_text = resp.output_text
    sources = [f"[{ch.source} p. {ch.page}]" for _, ch in contexts]
    return answer_text, sources

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    docs_count = sum(1 for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    status_text = "Ready" if ready else "Loading…"
    chat_history = get_history()
    return render_template(
        "index.html",
        status_text=status_text,
        is_busy=not ready,
        docs_count=docs_count,
        chat_history=chat_history
    )

@app.route("/upload", methods=["POST"])
def upload():
    chat_history = get_history()
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if f and f.filename.lower().endswith(".pdf"):
            path = os.path.join(DOCS_DIR, f.filename)
            f.save(path)
            saved.append(f.filename)
    # clear old FAISS files so we rebuild fresh
    for p in ["faiss.index", "meta.pkl"]:
        fp = os.path.join(INDEX_DIR, p)
        if os.path.exists(fp):
            os.remove(fp)
    rebuild_index()
    if saved:
        chat_history.append({"role":"system","text":f"Uploaded: {', '.join(saved)}. Index rebuilt with {len(chunks)} chunks.","sources":[]})
    else:
        flash("No PDF files uploaded.","warn")
    return redirect(url_for("home") + "#chat")

@app.route("/ask", methods=["POST"])
def ask():
    chat_history = get_history()
    q = (request.form.get("question") or "").strip()
    if not q:
        flash("Please enter a question.","warn")
        return redirect(url_for("home") + "#chat")
    if not ready or index is None:
        rebuild_index()
    chat_history.append({"role":"user","text":q,"sources":[]})
    hits = retrieve(index, chunks, q, k=TOP_K)
    if not hits:
        chat_history.append({"role":"bot","text":"I don't see any relevant text in your PDFs yet. Try uploading PDFs first.","sources":[]})
        return redirect(url_for("home") + "#chat")
    ans_text, sources = answer_question(q, hits)
    chat_history.append({"role":"bot","text":ans_text,"sources":sources})
    return redirect(url_for("home") + "#chat")

# Reset only the current user's chat
@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    histories.pop(sid, None)
    session.clear()  # force a new sid
    return redirect(url_for("home") + "#chat")

if __name__ == "__main__":
    print("Starting server and (re)building index if needed…")
    rebuild_index()
    app.run(host="0.0.0.0", port=5000, debug=True) 