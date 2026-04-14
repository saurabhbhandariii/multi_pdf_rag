#  Multi-PDF RAG System

A **100% free**, fully local Retrieval-Augmented Generation (RAG) app.  
Upload any number of PDFs and ask questions — answers are grounded strictly in your documents.

--

##  Project Structure

```
multi_pdf_rag/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

##  Quick Setup in VS Code

### Step 1 — Open project in VS Code
```
File → Open Folder → select the multi_pdf_rag folder
```

### Step 2 — Open the integrated terminal
```
Terminal → New Terminal   (or Ctrl + ` )
```

### Step 3 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⏳ First install takes 3–5 minutes (downloading PyTorch, sentence-transformers, etc.)

### Step 5 — (Recommended) Install Ollama for best LLM quality
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows → download from https://ollama.com/download

# After install, pull a free model:
ollama pull llama3       # best quality  (~4 GB)
# OR
ollama pull mistral      # faster        (~4 GB)
# OR
ollama pull phi3         # lightest      (~2 GB)

# Start Ollama server (keep this terminal open):
ollama serve
```
> **Without Ollama:** The app auto-downloads `google/flan-t5-base` from HuggingFace (~300 MB) on first question.

### Step 6 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** 🎉

---

## 🖥️ VS Code Tips

| Action | Shortcut |
|--------|----------|
| Open terminal | Ctrl + ` |
| Open file | Ctrl + P → type filename |
| Stop the app | Ctrl + C in terminal |
| Restart app | Ctrl + C → `streamlit run app.py` |

**Recommended VS Code extensions:**
- Python (Microsoft)
- Pylance
- Python Indent

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDF (text extraction)
    │
    ▼
RecursiveCharacterTextSplitter
    chunk_size=800, overlap=150
    │
    ▼
HuggingFace MiniLM-L6-v2 (embeddings)
    │
    ▼
FAISS Vector Store (in-memory)
    │
    ▼
User Question → MiniLM embed → FAISS top-K search
    │
    ▼
Relevance filter (score < 1.6)
    │
    ▼
Ollama / Flan-T5 (strict grounded prompt)
    │
    ▼
Answer + Source citations
```

---

## 🛡️ Anti-Hallucination Design

| Measure | How |
|---------|-----|
| Strict prompt | LLM only allowed to use retrieved context |
| Score filter | Chunks with poor similarity are discarded |
| Not-found reply | Explicit message if no relevant chunk found |
| Source attribution | Every answer shows PDF filename + page number |

---

## 🔧 LLM Priority (all free)

| Priority | LLM | Quality | Size |
|----------|-----|---------|------|
| 1st | Ollama llama3 | ⭐⭐⭐⭐⭐ | ~4 GB |
| 2nd | Ollama mistral | ⭐⭐⭐⭐ | ~4 GB |
| 3rd | Ollama phi3 | ⭐⭐⭐ | ~2 GB |
| Fallback | Flan-T5-base | ⭐⭐ | ~300 MB |

---

## ❓ Troubleshooting

**"No module named streamlit"**
→ Make sure your virtual environment is activated: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

**Slow first run**
→ Normal — sentence-transformers model is downloading (~90 MB)

**Ollama not working**
→ Make sure `ollama serve` is running in a separate terminal

**PDF shows no results**
→ Some PDFs are image-based (scanned). Try a text-based PDF first.

**Port already in use**
→ `streamlit run app.py --server.port 8502`
#
