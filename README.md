# 🏦 BANKAI — Intelligent Customer Service Bot

> An AI-powered banking assistant built with RAG, Sentiment Analysis, and Smart Escalation — developed as part of a bank internship project.

---

## 📌 Project Overview

BANKAI is an intelligent customer service chatbot designed for banks. It answers customer queries using the bank's own policy documents, detects customer frustration in real time, and automatically escalates distressed conversations to human agents.

This project was built using **LangChain**, **ChromaDB**, **Groq LLM**, **VADER Sentiment Analysis**, and **Streamlit**.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔍 **RAG-Powered Answers** | Answers questions strictly from bank PDF documents — never makes up information |
| 🧠 **Sentiment Analysis** | Scores every customer message from -1.0 (very negative) to +1.0 (very positive) |
| ⚡ **Smart Escalation** | Automatically flags frustrated customers and routes them to a human agent |
| 👤 **User Authentication** | Login and registration system with session management |
| 📊 **Agent Dashboard** | Separate dashboard for bank officers to view all escalated conversations |
| 🎨 **Premium UI** | Clean, professional banking interface built with custom CSS |

---

## 🏗️ System Architecture

```
Customer Message
       │
       ▼
Sentiment Analysis (VADER)
       │
  ┌────┴────┐
  │         │
Frustrated  Normal
  │         │
  ▼         ▼
Escalate   RAG Chain
to Agent   (LLM + ChromaDB)
  │         │
  ▼         ▼
Log to     Answer from
CSV        Bank Documents
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit with custom CSS |
| **LLM** | Groq API — llama-3.3-70b-versatile (free tier) |
| **RAG Framework** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| **Sentiment Analysis** | VADER (vaderSentiment) |
| **PDF Processing** | PyMuPDF (fitz) |
| **Environment** | Python 3.10+, Virtual Environment |

---

## 📁 Project Structure

```
bank-bot/
├── docs/                     ← Bank PDF documents (knowledge base)
├── vectorstore/              ← ChromaDB vector embeddings (auto-generated)
├── ingest.py                 ← Reads PDFs and builds vector database
├── sentiment.py              ← Sentiment scoring and escalation logic
├── rag_chain.py              ← RAG pipeline connecting LLM to vector DB
├── app.py                    ← Main Streamlit chat UI
├── agent_dashboard.py        ← Human agent escalation dashboard
├── escalation_log.csv        ← Auto-generated log of flagged conversations
├── .env                      ← Secret API keys (never commit this)
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-bot.git
cd bank-bot
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

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### Step 5 — Add bank PDF documents

Place your bank's PDF documents inside the `docs/` folder.  
These can be:
- Customer FAQ documents
- Account opening guides
- Loan policy documents
- Service charge schedules
- KYC requirement documents

### Step 6 — Ingest PDFs into vector database

```bash
python ingest.py
```

This reads all PDFs, splits them into chunks, converts them to embeddings, and stores them in ChromaDB. Run this once, or whenever you add new PDFs.

### Step 7 — Run the chat application

```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

### Step 8 — Run the agent dashboard (optional, separate terminal)

```bash
streamlit run agent_dashboard.py --server.port 8502
```

Open your browser at: `http://localhost:8502`

---

## 🔐 Demo Login Credentials

| Username | Password | Role |
|---|---|---|
| `demo` | `demo123` | Customer |
| `sourav` | `sourav123` | Developer |
| `manager` | `bank2024` | Manager |
| `admin` | `admin123` | Admin |

You can also register a new account directly from the login screen.

---

## 🧠 How RAG Works

1. Bank PDFs are loaded and split into 500-character chunks
2. Each chunk is converted into a vector embedding using sentence-transformers
3. All embeddings are stored in ChromaDB (local vector database)
4. When a customer asks a question, the top 3 most relevant chunks are retrieved
5. These chunks + the customer question are sent to the Groq LLM
6. The LLM answers strictly based on the retrieved chunks only

This ensures the bot never makes up information — it only answers from verified bank documents.

---

## 😤 How Sentiment Escalation Works

Every customer message is scored by VADER sentiment analysis:

| Score Range | Label | Action |
|---|---|---|
| -1.0 to -0.6 | 😤 FRUSTRATED | Escalate to human agent + log to CSV |
| -0.6 to 0.05 | 😐 NEUTRAL | Answer normally via RAG |
| 0.05 to 1.0 | 😊 POSITIVE | Answer normally via RAG |

When escalation is triggered:
- Customer receives an empathetic response
- Conversation is logged to `escalation_log.csv` with timestamp, message, and score
- Bank officer can view all escalations on the agent dashboard at port 8502

---

## 📊 Agent Dashboard Features

- Total escalation count
- Average sentiment score
- Most recent escalation timestamp
- Most urgent message highlighted
- Full table of all flagged conversations
- Download escalation log as CSV

---

## 🚀 Deployment

### Deploy on Streamlit Cloud (free)

1. Push your project to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub → New App → select your repository
4. Set main file path to `app.py`
5. Go to Advanced Settings → Secrets and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

6. Click Deploy — your app goes live with a public URL in 2-3 minutes

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
```

Install all at once:

```bash
pip install langchain langchain-community langchain-chroma langchain-groq \
            langchain-core chromadb pypdf pymupdf sentence-transformers \
            vaderSentiment streamlit pandas python-dotenv
```

---

## 💡 Why Banks Care About This Project

- **Reduces call center load** — handles thousands of routine queries automatically
- **Zero hallucination risk** — bot only answers from verified documents
- **Protects customer satisfaction** — frustrated customers are never left with a bot
- **Audit trail** — every escalation is logged with timestamp for compliance
- **Cost effective** — runs on free-tier APIs (Groq + HuggingFace)

---

## 🔮 Future Enhancements

- Connect to real bank database for live account information
- Add multilingual support for Hindi and regional languages
- Integrate WhatsApp API for mobile customer service
- Add voice input using Whisper ASR
- Replace CSV logging with PostgreSQL for production use
- Add Redis queue for managing concurrent escalations

---

## 👨‍💻 Developer

**Sourav**
Built during banking internship as an AI-powered customer service solution.

---

## 📄 License

This project is for educational and internship demonstration purposes.
