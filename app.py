import streamlit as st
import os
import tempfile
import re

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Active Groq models (2025) ──────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]

st.set_page_config(page_title="PDF Chat", page_icon="📚", layout="wide")

st.markdown("""
<style>
  /* Hide Streamlit default header/footer */
  #MainMenu, footer, header {visibility: hidden;}

  /* Full page dark background */
  .stApp { background-color: #212121; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background-color: #2f2f2f !important;
  }
  section[data-testid="stSidebar"] * { color: #ececec !important; }

  /* Chat container */
  .chat-wrap { max-width: 800px; margin: 0 auto; padding-bottom: 20px; }

  /* User bubble */
  .user-bubble {
      background: #2f2f2f;
      color: #ececec;
      border-radius: 18px 18px 4px 18px;
      padding: 12px 18px;
      margin: 8px 0 8px 80px;
      font-size: .95rem;
      line-height: 1.6;
      word-break: break-word;
  }

  /* Assistant bubble */
  .bot-bubble {
      background: #1a1a2e;
      color: #e8e8f0 !important;
      border-left: 3px solid #6c63ff;
      border-radius: 0 18px 18px 18px;
      padding: 14px 18px;
      margin: 8px 80px 8px 0;
      font-size: .95rem;
      line-height: 1.8;
      word-break: break-word;
  }

  /* Source chips */
  .src-chip {
      display: inline-block;
      background: #2a2a4a;
      color: #a5b4fc !important;
      border: 1px solid #4f46e5;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: .75rem;
      margin: 3px 3px 0 0;
  }

  /* Title */
  .app-title {
      font-size: 1.6rem;
      font-weight: 800;
      color: #a5b4fc;
      margin-bottom: 4px;
  }
  .app-sub {
      font-size: .85rem;
      color: #6b7280;
      margin-bottom: 20px;
  }

  /* Status pill */
  .pill-green {
      background:#065f46; color:#6ee7b7 !important;
      border-radius:20px; padding:3px 12px;
      font-size:.78rem; font-weight:700;
  }
  .pill-gray {
      background:#374151; color:#9ca3af !important;
      border-radius:20px; padding:3px 12px;
      font-size:.78rem; font-weight:700;
  }

  /* Input box override */
  .stChatInput textarea {
      background:#2f2f2f !important;
      color:#ececec !important;
      border:1px solid #4b5563 !important;
      border-radius:12px !important;
  }

  /* Buttons */
  .stButton>button {
      border-radius:10px !important;
      font-weight:700 !important;
      background:#4f46e5 !important;
      color:white !important;
      border:none !important;
  }
  .stButton>button:hover { background:#4338ca !important; }

  div[data-testid="stExpander"] {
      background:#1e1e2e;
      border:1px solid #374151;
      border-radius:8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("vectorstore", None),
    ("bm25", None),
    ("all_docs", []),
    ("chat_history", []),   # [{role, content, sources}]
    ("pdf_names", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Embeddings ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Text splitter ──────────────────────────────────────────────────────────────
def make_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Smaller chunks = more precise retrieval
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
    )


# ── Process PDFs → FAISS + BM25 (Hybrid RAG) ──────────────────────────────────
def process_pdfs(files):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from rank_bm25 import BM25Okapi

    splitter = make_splitter()
    all_docs, names = [], []

    with tempfile.TemporaryDirectory() as tmp:
        for uf in files:
            path = os.path.join(tmp, uf.name)
            with open(path, "wb") as f:
                f.write(uf.read())
            try:
                pages = PyPDFLoader(path).load()
                if not pages:
                    st.warning(f"⚠️ No text in `{uf.name}` — scanned PDF?")
                    continue
                for p in pages:
                    p.metadata["source_file"] = uf.name
                chunks = splitter.split_documents(pages)
                all_docs.extend(chunks)
                names.append(uf.name)
                st.caption(f"✅ {uf.name} → {len(chunks)} chunks")
            except Exception as e:
                st.warning(f"Error reading {uf.name}: {e}")

    if not all_docs:
        return None, None, [], []

    # FAISS — dense semantic search
    vs = FAISS.from_documents(all_docs, get_embeddings())

    # BM25 — keyword / sparse search
    tokenized = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)

    return vs, bm25, all_docs, names


# ── Hybrid retrieval: BM25 + FAISS re-ranked ──────────────────────────────────
def hybrid_retrieve(question, vectorstore, bm25, all_docs, top_k=8):
    from rank_bm25 import BM25Okapi

    # 1. Dense retrieval (semantic)
    dense_hits = vectorstore.similarity_search(question, k=top_k)
    dense_set  = {d.page_content: d for d in dense_hits}

    # 2. Sparse retrieval (BM25 keyword)
    tokens     = question.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_hits  = [all_docs[i] for i in top_bm25_idx]
    bm25_set   = {d.page_content: d for d in bm25_hits}

    # 3. Merge: union of both sets, dense hits ranked first
    merged = {}
    for d in dense_hits:
        merged[d.page_content] = d
    for d in bm25_hits:
        if d.page_content not in merged:
            merged[d.page_content] = d

    # 4. Score each merged doc: dense_rank + bm25_rank (reciprocal rank fusion)
    dense_rank = {d.page_content: i for i, d in enumerate(dense_hits)}
    bm25_rank  = {all_docs[i].page_content: i for i in top_bm25_idx}

    def rrf_score(text):
        dr = dense_rank.get(text, 9999)
        br = bm25_rank.get(text, 9999)
        return 1 / (60 + dr) + 1 / (60 + br)

    ranked = sorted(merged.values(), key=lambda d: rrf_score(d.page_content), reverse=True)
    return ranked[:top_k]


# ── Build conversation messages for Groq ──────────────────────────────────────
def build_messages(question, context, chat_history):
    system = """You are a helpful, accurate document assistant. Your job is to answer questions based on the provided document context.

Rules:
- Answer ONLY from the document context provided
- Be specific, detailed and accurate  
- If the exact info is in the context, quote or reference it directly
- If info is NOT in context, say clearly: "This information is not found in the uploaded documents."
- Maintain conversation context — if user says "he", "she", "it", use previous messages to understand who/what they mean
- Format answers clearly with bullet points or numbered lists when appropriate
- Never hallucinate or make up information"""

    messages = [{"role": "system", "content": system}]

    # Add last 4 conversation turns for memory
    for entry in chat_history[-4:]:
        messages.append({"role": entry["role"], "content": entry["content"]})

    # Current question with fresh context
    user_msg = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Answer based strictly on the context above:"""

    messages.append({"role": "user", "content": user_msg})
    return messages


# ── Get answer ─────────────────────────────────────────────────────────────────
def get_answer(question, vectorstore, bm25, all_docs, chat_history, top_k=8):
    from groq import Groq

    # Hybrid retrieve
    hits = hybrid_retrieve(question, vectorstore, bm25, all_docs, top_k)
    if not hits:
        return "No relevant content found in your PDFs.", []

    sources, ctx_parts = [], []
    for doc in hits:
        src  = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        ctx_parts.append(f"[File: {src} | Page: {page}]\n{doc.page_content.strip()}")
        sources.append({"file": src, "page": page})

    context  = "\n\n---\n\n".join(ctx_parts)
    messages = build_messages(question, context, chat_history)

    client     = Groq(api_key=GROQ_API_KEY)
    last_error = None

    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip(), sources
        except Exception as e:
            last_error = str(e)
            if any(x in last_error for x in ["decommissioned", "model_not_found", "404", "not found"]):
                continue
            raise Exception(last_error)

    raise Exception(f"All models failed. Last: {last_error}")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📚 PDF Chat")

    # Status
    if st.session_state.pdf_names:
        st.markdown(f'<span class="pill-green">✅ {len(st.session_state.pdf_names)} PDF(s) ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-gray">⚪ No PDFs loaded</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📂 Upload PDFs")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    top_k    = st.slider("Retrieval depth", 5, 15, 8,
                         help="Higher = more context retrieved per question")

    if st.button("⚡ Process PDFs", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Building Hybrid RAG index…"):
                vs, bm25, docs, names = process_pdfs(uploaded)
            if vs:
                st.session_state.vectorstore  = vs
                st.session_state.bm25         = bm25
                st.session_state.all_docs     = docs
                st.session_state.pdf_names    = names
                st.session_state.chat_history = []
                st.success(f"✅ {len(names)} PDF(s) indexed!")
            else:
                st.error("No text extracted.")

    st.divider()
    st.subheader("📋 Indexed Files")
    if st.session_state.pdf_names:
        for n in st.session_state.pdf_names:
            st.markdown(f"- `{n}`")
        if st.button("🗑️ Clear Chat & Docs", use_container_width=True):
            for k in ["vectorstore", "bm25", "all_docs", "chat_history", "pdf_names"]:
                st.session_state[k] = None if k in ["vectorstore","bm25"] else []
            st.rerun()
    else:
        st.info("No PDFs yet.")

    st.divider()
    st.markdown("""
**Hybrid RAG:**
- 🔍 BM25 keyword search
- 🧠 FAISS semantic search  
- 🔀 Reciprocal Rank Fusion
- 💬 Conversation memory (4 turns)

**Model:** llama-3.3-70b  
**Speed:** ~1-2 sec ⚡  
**Cost:** Free
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="app-title">💬 PDF Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Ask anything about your uploaded documents</p>', unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#6b7280;">
        <div style="font-size:4rem;">📄</div>
        <h3 style="color:#9ca3af;">Upload your PDFs to get started</h3>
        <p>Use the sidebar to upload and process your PDF files</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Render chat history ────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 {entry["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            # Format answer — convert **bold** and newlines to HTML
            ans = entry["content"]
            ans = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', ans)
            ans = ans.replace("\n", "<br>")

            st.markdown(f'<div class="bot-bubble">🤖 {ans}</div>',
                        unsafe_allow_html=True)

            # Sources
            if entry.get("sources"):
                seen = set()
                chips = ""
                for s in entry["sources"]:
                    key = f"{s['file']}|{s['page']}"
                    if key not in seen:
                        seen.add(key)
                        chips += f'<span class="src-chip">📄 {s["file"]} p.{s["page"]}</span>'
                if chips:
                    st.markdown(f'<div style="margin:4px 0 12px 0">{chips}</div>',
                                unsafe_allow_html=True)

    # ── Chat input ─────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your PDFs…")
    if question:
        # Show user message immediately
        st.markdown(f'<div class="user-bubble">👤 {question}</div>',
                    unsafe_allow_html=True)

        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})

        # Get answer
        with st.spinner("Thinking…"):
            try:
                answer, sources = get_answer(
                    question,
                    st.session_state.vectorstore,
                    st.session_state.bm25,
                    st.session_state.all_docs,
                    st.session_state.chat_history,
                    top_k,
                )
            except Exception as e:
                answer  = f"❌ Error: {str(e)}"
                sources = []

        # Show answer
        ans_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer)
        ans_html = ans_html.replace("\n", "<br>")
        st.markdown(f'<div class="bot-bubble">🤖 {ans_html}</div>', unsafe_allow_html=True)

        # Show sources
        if sources:
            seen, chips = set(), ""
            for s in sources:
                key = f"{s['file']}|{s['page']}"
                if key not in seen:
                    seen.add(key)
                    chips += f'<span class="src-chip">📄 {s["file"]} p.{s["page"]}</span>'
            st.markdown(f'<div style="margin:4px 0 12px 0">{chips}</div>', unsafe_allow_html=True)

        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.rerun()