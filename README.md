# 🧠 AI-Powered Multi-Agent Research & Decision Assistant

A production-ready, fully local-capable AI research pipeline built with **LangChain**, **LangGraph**, and **Ollama**. Submit any research question and watch five specialized agents collaborate, self-correct, and deliver a verified, source-cited answer.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│          LangGraph Stateful Orchestrator             │
│                                                      │
│  Searcher → Retriever → Summarizer → Critic ──┐     │
│                                    ↑           │     │
│                                    └── RETRY ──┘     │
│                                         │            │
│                                    Finalizer         │
└─────────────────────────────────────────────────────┘
     │
     ▼
Accurate Summary + Sources + SQLite Session Record
```

| Agent | Role |
|---|---|
| **Searcher** | Fetches raw data from Tavily web search + ChromaDB local docs |
| **Retriever** | Filters and ranks results to keep only relevant facts |
| **Summarizer** | Compresses facts into key points via Ollama LLM |
| **Critic** | Fact-checks for hallucinations; triggers retry if needed |
| **Finalizer** | Synthesizes final answer with inline citations |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Tavily API key (free at [tavily.com](https://tavily.com))

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/ai-research-assistant.git
cd ai-research-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

### 3. Pull Ollama Model
```bash
ollama pull llama3.2:3b
```

### 4. Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

### 5. Run the FastAPI Server (optional)
```bash
uvicorn app.api:app --reload --port 8000
```
API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker

```bash
# Build and start everything (app + Ollama sidecar)
docker-compose up --build

# Stop
docker-compose down
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
├── src/
│   ├── config.py          # Pydantic settings
│   ├── state.py           # LangGraph shared state schema
│   ├── graph.py           # LangGraph pipeline definition
│   ├── vectorstore.py     # ChromaDB integration
│   ├── tools.py           # LangChain tool wrappers
│   ├── database.py        # SQLite session storage
│   └── agents/
│       ├── searcher.py    # Web + doc search agent
│       ├── retriever.py   # Relevance filter agent
│       ├── summarizer.py  # Ollama summarization agent
│       ├── critic.py      # Hallucination checker agent
│       └── finalizer.py   # Answer synthesis agent
├── app/
│   ├── streamlit_app.py   # Interactive UI
│   └── api.py             # FastAPI REST endpoints
├── tests/
│   └── test_graph.py
├── data/                  # ChromaDB + SQLite (gitignored)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/deploy.yml
```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TAVILY_API_KEY` | Tavily web search API key | Required |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2:3b` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/chroma` |
| `DATABASE_URL` | SQLite file path | `./data/sessions.db` |
| `MAX_RETRY` | Max critic retry attempts | `3` |

---

## 🌐 Deployment

Deployed on **Railway** via GitHub Actions CD. Every push to `main` triggers:
1. Lint (ruff + black)
2. Tests (pytest)
3. Docker build & push to GHCR
4. Railway redeploy

---

## 📄 License

MIT
