# 🤖 AutoResearcher: Local RAG with Ollama + Streamlit

> 🧠 A fully local, privacy-preserving AI research assistant built with **Ollama**, **LangChain**, and **Streamlit** no cloud APIs, no cost, no rate limits.

---

## 🚀 Overview

**AutoResearcher** is a local Retrieval-Augmented Generation (RAG) system that can read, understand, and summarize multiple PDFs using a locally running LLM from **Ollama** (such as *Llama 3*, *Mistral*, or *Qwen*).  

It builds a local **Chroma** vector database using **SentenceTransformer embeddings**, retrieves relevant document chunks for a given question, and synthesizes a concise, cited answer via **Ollama’s LLMs**.

---

## ✨ Key Features

- 📄 **PDF Upload**: Upload one or multiple research papers or reports  
- 🧩 **Offline Vector Index**: Uses Chroma + SentenceTransformers for local embeddings  
- 💬 **Local LLM (Ollama)**: Runs Llama 3 or any supported model via `/api/chat`  
- 🔍 **RAG Pipeline**: Retrieves top-k context chunks for precise, cited synthesis  
- 🧠 **Diverse Retrieval (MMR)**: Ensures variety across documents  
- 📊 **Index Stats View**: Inspect all indexed sources and chunk counts  
- 🧾 **Clean UI**: Streamlit interface for research-style question answering  

---

## 🧰 Tech Stack

| Component | Description |
|------------|--------------|
| **Ollama** | Local LLM engine (Llama 3 / Mistral / Qwen etc.) |
| **Streamlit** | Interactive web UI |
| **LangChain + Chroma** | Retrieval-augmented memory backend |
| **SentenceTransformers** | Local text embeddings |
| **PyMuPDF / PyPDF** | Robust PDF parsing |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Iyanuoluwa007/AutoResearcher.git
cd AutoResearcher
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate    # (Windows)
# or: source .venv/bin/activate (Linux/macOS)
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install & start Ollama
Download Ollama from https://ollama.ai
Then pull and serve your preferred model (e.g. Llama 3.2):
```bash
ollama pull llama3.2
ollama serve
```

### 5️⃣ Configure environment
Create a .env file (or copy .env.example):
```bash
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
EMBED_MODEL_NAME=all-MiniLM-L6-v2
```

#### 6️⃣ Run the app
```bash
streamlit run app.py
```
Then open http://localhost:8501

### 💡 Usage
1. Start Ollama in a terminal:
```bash
ollama serve
```
2. Launch Streamlit in another terminal.
3. Upload one or more PDFs (research papers, docs, manuals).
4. Ask a question (e.g. “Compare YOLOv11 vs YOLOv8 for robotics vision on embedded devices”).
5. View a structured, cited summary, fully local!

### 📸 Example Output
![AutoResearcher Screenshot](https://raw.githubusercontent.com/Iyanuoluwa007/AutoResearcher/main/Screenshot%202025-11-03%20124329.png)

### 🧾 Folder Structure
```bash
AutoResearcher/
│
├── app.py               # Streamlit application
├── requirements.txt     # Dependencies
├── .env.example         # Example environment config
├── data/                # Auto-created (PDFs + Chroma DB)
└── README.md
```

### 🧠 Future Enhancements
- 🔄 Multi-Agent version using LangGraph (plan → gather → synthesize)
- 🌐 Web Search integration (Tavily API)
- 🗣️ Text-to-Speech / voice query input
- 📊 Export to Markdown / PDF summaries
