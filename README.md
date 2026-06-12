<div align="center">

# 🏦 BANKAI
### Intelligent Customer Service Bot — Union Bank of India

> An AI-powered banking assistant built with RAG, Web Search Fallback, Sentiment Analysis, Smart Escalation, and a premium dark-mode UI — developed during Summer Internship at **Union Bank of India**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq_LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Live Demo](https://img.shields.io/badge/Live_Demo-BANKAI-22c55e?style=for-the-badge&logo=streamlit)](https://github.com/sourav2212/BANKBOT)

</div>

---

## 📌 Project Overview

**BANKAI** is an intelligent customer service chatbot built for **Union Bank of India**. It answers customer queries using the bank's own policy documents via a RAG pipeline, falls back to live web search when documents are insufficient, detects customer frustration in real time using sentiment analysis, and automatically escalates distressed conversations to human agents — all wrapped in a premium dark-mode UI.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔍 **RAG-Powered Answers** | Answers strictly from Union Bank PDF documents — zero hallucination risk |
| 🌐 **Web Search Fallback** | Auto-searches web via Tavily API when bank documents don't have the answer |
| 🧠 **Sentiment Analysis** | Scores every message from -1.0 (frustrated) to +1.0 (positive) using VADER |
| ⚡ **Smart Escalation** | Automatically routes frustrated customers (score ≤ -0.6) to human agents |
| 💬 **Conversation Memory** | Sliding window memory (last 5 turns) for contextual multi-turn dialogues |
| 👤 **User Authentication** | Login + registration with profile picture upload and session management |
| 📊 **Agent Dashboard** | Live escalation log with sentiment scores, urgent case highlighting, CSV export |
| 📈 **Analytics Dashboard** | Query trends, source breakdown, sentiment distribution, feedback summary |
| 👍 **Feedback System** | Per-response thumbs up/down logged to SQLite for fine-tuning pipeline |
| 🗄️ **SQLite Logging** | Structured DB for escalations, feedback, and query logs — compliance ready |
| 🚦 **Rate Limiting** | 20 queries/session with live progress bar indicator |
| 🌙 **Dark / Light Mode** | Full theme toggle with proper textarea and input styling |
| 📄 **Source Labels** | Every response labelled as 📄 documents or 🌐 web search |
| ⏰ **Time-based Greeting** | Good morning / afternoon / evening based on login time |
| 🎨 **Premium UI** | Gradient bubbles, animated typing indicator, profile avatar, polished CSS |

---

## 🏗️ System Architecture

```
Customer Message
       │
       ▼
┌─────────────────────┐
│  Sentiment Analysis  │  ← VADER scores every message
│      (VADER)         │
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     │            │
 Frustrated    Normal
  (≤ -0.6)      │
     │          ▼
     │   ┌─────────────┐
     │   │  RAG Chain   │  ← Top 3 chunks from ChromaDB
     │   │  LangChain   │
     │   └──────┬───────┘
     │          │
     │    Answer found?
     │    ┌─────┴──────┐
     │    │            │
     │   Yes           No
     │    │            │
     │    ▼            ▼
     │  📄 Return   🌐 Tavily
     │  Doc Answer  Web Search
     │                 │
     │            Re-answer
     │            via LLM
     │
     ▼
Escalate to Agent
+ Log to SQLite
+ Empathetic Response
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit with custom CSS, dark/light mode, gradient UI |
| **LLM** | Groq API — `llama-3.3-70b-versatile` (free tier) |
| **RAG Framework** | LangChain |
| **Vector Database** | ChromaDB (local) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Web Search** | Tavily Search API — fallback when RAG is insufficient |
| **Sentiment Analysis** | VADER (`vaderSentiment`) |
| **Memory** | LangChain `InMemoryChatMessageHistory` (k=5 window) |
| **Database** | SQLite — escalations, feedback, query logs |
| **PDF Processing** | PyMuPDF (fitz) |
| **Deployment** | Streamlit Cloud / Docker |
| **Environment** | Python 3.10+, Virtual Environment |

---

## 📁 Project Structure

```
bank-bot/
├── docs/                     ← Union Bank PDF documents (knowledge base)
├── vectorstore/              ← ChromaDB vector embeddings (auto-generated)
├── venv/                     ← Python virtual environment
├── app.py                    ← Main Streamlit chat UI
├── analytics.py              ← Analytics dashboard (port 8503)
├── agent_dashboard.py        ← Human agent escalation dashboard (port 8502)
├── ingest.py                 ← Reads PDFs and builds vector database
├── rag_chain.py              ← RAG pipeline connecting LLM to vector DB
├── sentiment.py              ← Sentiment scoring and escalation logic
├── bankai.db                 ← SQLite database (auto-generated on first run)
├── escalation_log.csv        ← Legacy CSV escalation log
├── Dockerfile                ← Docker container config
├── .env                      ← Secret API keys (never commit this)
├── .gitignore                ← Ignores .env, venv, vectorstore, bankai.db
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/sourav2212/BANKBOT.git
cd BANKBOT/bank-bot
```

### Step 2 — Create and activate virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

| Key | Where to get |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — 1000 searches/month free |

### Step 5 — Add bank PDF documents

Place Union Bank PDF documents inside the `docs/` folder:
- Customer FAQ documents
- Account opening guides
- Loan policy documents
- Service charge schedules
- KYC requirement documents

### Step 6 — Ingest PDFs into vector database

```bash
python ingest.py
```

Reads all PDFs, splits into 500-character chunks, converts to embeddings, stores in ChromaDB. Run once, or whenever new PDFs are added.

### Step 7 — Run the applications

```bash
# Terminal 1 — Main chat app
streamlit run app.py

# Terminal 2 — Agent escalation dashboard
streamlit run agent_dashboard.py --server.port 8502

# Terminal 3 — Analytics dashboard
streamlit run analytics.py --server.port 8503
```

| App | URL |
|---|---|
| Main Chat | `http://localhost:8501` |
| Agent Dashboard | `http://localhost:8502` |
| Analytics | `http://localhost:8503` |

---

## 🔐 Demo Login Credentials

| Username | Password | Role |
|---|---|---|
| `demo` | `demo123` | Customer |
| `sourav` | `sourav123` | Developer |
| `manager` | `bank2024` | Manager |
| `admin` | `admin123` | Admin |

You can also register a new account with profile picture upload directly from the login screen.

---

## 🧠 How RAG + Web Search Works

```
1. Bank PDFs  →  500-char chunks  →  HuggingFace embeddings  →  ChromaDB
2. Customer question  →  retrieve top 3 relevant chunks
3. Chunks + conversation history + question  →  Groq LLM (Llama 3.3 70B)
4. LLM answers from chunks
5. If answer contains "I don't have that information"  →  Tavily web search triggered
6. Web results  →  re-sent to LLM  →  grounded web answer returned
7. Source labelled: 📄 documents  or  🌐 web search
```

This ensures the bot **never makes up information** and **never leaves a question unanswered**.

---

## 😤 How Sentiment Escalation Works

| Score Range | Label | Action |
|---|---|---|
| -1.0 to -0.6 | 😤 FRUSTRATED | Escalate to human agent + log to SQLite |
| -0.6 to 0.05 | 😐 NEUTRAL | Answer normally via RAG / web |
| 0.05 to 1.0 | 😊 POSITIVE | Answer normally via RAG / web |

When escalation triggers:
- Customer receives an empathetic response with UBI helpline number
- Conversation logged to `escalations` table in SQLite
- Agent dashboard highlights the case in real time

---

## 📊 Dashboards

### Agent Dashboard (`port 8502`)
- Total escalation count and average sentiment score
- Most recent and most urgent escalation highlighted
- Full table of all flagged conversations
- Download escalation log as CSV

### Analytics Dashboard (`port 8503`)
- Queries over time (line chart)
- Answer source breakdown — documents vs web (bar chart)
- Sentiment distribution across all sessions
- Feedback summary — positive vs negative rate
- Recent escalations and feedback tables
- Download all data as CSV

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t bankai .

# Run container
docker run -p 8501:8501 --env-file .env bankai
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. New App → select repository → set main file: `app.py`
4. Advanced Settings → Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
```

5. Click Deploy — live URL in 2-3 minutes

---

## 📦 Dependencies

```
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-core
chromadb
pypdf
pymupdf
sentence-transformers
vaderSentiment
streamlit
pandas
python-dotenv
tavily-python
langchain-tavily
```

```bash
pip install langchain langchain-community langchain-chroma langchain-groq \
            langchain-core chromadb pypdf pymupdf sentence-transformers \
            vaderSentiment streamlit pandas python-dotenv \
            tavily-python langchain-tavily
```

---

## 💡 Why Banks Care About This Project

| Value | Impact |
|---|---|
| **Reduces call center load** | Handles thousands of routine queries automatically |
| **Zero hallucination risk** | Bot answers from verified documents first, web second |
| **Always has an answer** | Web search fallback means no query goes unanswered |
| **Protects customer satisfaction** | Frustrated customers are never left with a bot |
| **Compliance audit trail** | Every escalation logged with timestamp and score in SQLite |
| **Actionable insights** | Analytics dashboard shows query trends and feedback rates |
| **Cost effective** | Runs entirely on free-tier APIs (Groq + HuggingFace + Tavily) |

---

## 🔮 Future Enhancements

- Connect to real Union Bank database for live account information
- Add multilingual support for Hindi and regional languages
- Integrate WhatsApp Business API for mobile customer service
- Add voice input using OpenAI Whisper ASR
- Replace SQLite with PostgreSQL for production scale
- Add Redis queue for managing concurrent escalations
- Fine-tune LLM on Union Bank-specific Q&A pairs using feedback data
- Implement OAuth2 login with Union Bank employee ID

---

## 👨‍💻 Developer

**Sourav Shandilya**
B.Tech Computer Science & Engineering — VIT Bhopal University (Batch 2023–2027)
Built during **Summer Internship at Union Bank of India** as an AI-powered customer service solution.

| | |
|---|---|
| 🔗 GitHub | [github.com/sourav2212/BANKBOT](https://github.com/sourav2212/BANKBOT) |
 |

---

## 📄 License

This project is for educational and internship demonstration purposes.

---

<div align="center">
Built with ❤️ for Union Bank of India · Powered by LangChain + Groq + Streamlit
</div>
