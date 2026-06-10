# 🏦 BANKAI — Intelligent Customer Service Bot

> An AI-powered banking assistant built with RAG, Web Search Fallback, Sentiment Analysis, and Smart Escalation — developed as part of a banking internship project at **Union Bank of India**.

---

## 📌 Project Overview

BANKAI is an intelligent customer service chatbot designed for banks. It answers customer queries using the bank's own policy documents, falls back to live web search when documents are insufficient, detects customer frustration in real time, and automatically escalates distressed conversations to human agents.

This project was built using **LangChain**, **ChromaDB**, **Groq LLM**, **Tavily Search API**, **VADER Sentiment Analysis**, and **Streamlit**.

---

## 🎯 Key Features

| Feature | Description |
|----------|-------------|
| 🔍 RAG-Powered Answers | Retrieves information from bank documents using ChromaDB |
| 🌐 Web Search Fallback | Uses Tavily when document retrieval is insufficient |
| 🧠 Conversation Memory | Maintains chat context within a session using LangChain chat history |
| 😤 Sentiment Analysis | Detects customer frustration using VADER |
| ⚡ Smart Escalation | Automatically escalates highly negative conversations |
| 👤 Authentication System | Login and registration system with session management |
| 🖼️ Profile Pictures | Optional profile image upload during registration |
| 👍 Feedback System | Collects user feedback on chatbot responses |
| 🌙 Dark / Light Mode | Theme switching with a single click |
| 🚦 Rate Limiting | Limits users to 20 queries per session |
| 📈 Session Analytics | Real-time sidebar statistics and usage metrics |
| 💾 SQLite Logging | Stores escalations, feedback, and query history in SQLite |
| 🎨 Premium Banking UI | Modern banking-themed Streamlit interface |

---

## 🏗️  System Architecture

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
Escalate   RAG Retrieval
to Agent   (ChromaDB)
  │         │
  ▼         ▼
Log to     Generate Answer
SQLite        │
              ▼
      Answer Found?
         │
    ┌────┴────┐
    │         │
   Yes        No
    │         │
    ▼         ▼
📄 Document   🌐 Tavily Search
Answer        Fallback
                  │
                  ▼
           Re-answer via LLM
                  │
                  ▼
        Store Conversation Memory

---

## 📁  Project Structure

bank-bot/
├── docs/                     ← Bank PDF documents
├── vectorstore/              ← ChromaDB vector embeddings
├── app.py                    ← Main Streamlit application
├── bankai.db                 ← SQLite database
├── .env                      ← API keys
├── requirements.txt          ← Dependencies
└── README.md                 ← Project documentation

---

## 👤 User Authentication

BANKAI includes a complete authentication system:

- Sign In with existing account
- Register a new account
- Optional profile picture upload
- Session-based authentication
- Demo accounts for testing
- Automatic session initialization

### Demo Accounts

| Username | Password | Role |
|----------|----------|------|
| demo | demo123 | Customer |
| sourav | sourav123 | Developer |
| manager | bank2024 | Manager |
| admin | admin123 | Admin |

---

## 🧠 Conversation Memory

BANKAI uses LangChain's InMemoryChatMessageHistory to maintain context throughout a user's session.

Benefits:

- Remembers previous questions
- Supports follow-up queries
- Provides contextual responses
- Improves customer experience

Example:

Customer: "What is the home loan rate?"

Bot: "The home loan rate is 8.5%."

Customer: "What documents do I need?"

Bot: Understands the customer is still referring to a home loan and answers accordingly.

---

## 💾 SQLite Database Logging

BANKAI uses SQLite instead of CSV files for persistent logging.

### Tables

#### Escalations

Stores:

- Timestamp
- Username
- Message
- Sentiment score
- Sentiment label

#### Feedback

Stores:

- Timestamp
- Username
- Question
- Answer
- Rating (👍 / 👎)
- Source

#### Query Log

Stores:

- Timestamp
- Username
- Customer question
- Response source

Database file:

bankai.db

---

## 👍 Feedback Collection

Every AI response includes:

👍 Helpful

👎 Not Helpful

Feedback is automatically stored in SQLite and can be used for:

- Performance monitoring
- Model improvement
- Customer satisfaction tracking

---

## 🚦 Rate Limiting

To prevent abuse and excessive API usage:

- Maximum 20 queries per session
- Remaining queries displayed in sidebar
- User notified when limit is reached

---

## 📈 Session Analytics

The sidebar provides real-time statistics:

- Total Queries
- Escalations
- Positive Conversations
- Web Search Hits

This helps administrators monitor usage patterns and chatbot performance.

---

## 😤 Smart Escalation

Messages with sentiment score ≤ -0.6 are automatically escalated.

When escalation occurs:

- Customer receives an apology message
- Escalation is logged to SQLite
- Conversation is flagged for human review

Sentiment ranges:

| Score Range | Label | Action |
|-------------|--------|---------|
| -1.0 to -0.6 | 😤 FRUSTRATED | Escalate |
| -0.6 to 0.05 | 😐 NEUTRAL | Answer Normally |
| 0.05 to 1.0 | 😊 POSITIVE | Answer Normally |

---

## 🌙 Dark / Light Mode

Features:

- One-click theme toggle
- Professional banking UI
- Dark mode optimized for long sessions
- Light mode for daytime usage

---

## ⚠️ Note on Agent Dashboard

The current version focuses on the customer-facing chatbot application.

Future versions may include:

- Dedicated agent dashboard
- Escalation management
- Advanced analytics
- Ticket assignment workflow

---

## 🚀 Additional Enhancements Implemented

✔ Session-based conversation memory

✔ SQLite logging system

✔ User registration

✔ Profile picture upload

✔ Feedback collection

✔ Query rate limiting

✔ Real-time analytics

✔ Enhanced banking UI

✔ Dark/Light mode

✔ Web search fallback

✔ Smart escalation system
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

Install all at once:

```bash
pip install langchain langchain-community langchain-chroma langchain-groq \
            langchain-core chromadb pypdf pymupdf sentence-transformers \
            vaderSentiment streamlit pandas python-dotenv \
            tavily-python langchain-tavily
```

---

## 💡 Why Banks Care About This Project

- **Reduces call center load** — handles thousands of routine queries automatically
- **Zero hallucination risk** — bot answers from verified documents first, web search second
- **Always has an answer** — web search fallback means no query goes unanswered
- **Protects customer satisfaction** — frustrated customers are never left with a bot
- **Audit trail** — every escalation is logged with timestamp for compliance
- **Cost effective** — runs entirely on free-tier APIs (Groq + HuggingFace + Tavily)

---

## 🔮 Future Enhancements

- Connect to real bank database for live account information
- Add multilingual support for Hindi and regional languages
- Integrate WhatsApp API for mobile customer service
- Add voice input using Whisper ASR
- Replace CSV logging with PostgreSQL for production use
- Add Redis queue for managing concurrent escalations
- Persistent dark mode preference saved to user profile

---

## 👨‍💻 Developer

**Sourav Shandilya** — VIT Bhopal University (B.Tech CSE, Batch 2023–2027)
Built during Summer Internship at **Union Bank of India** as an AI-powered customer service solution.

- 🔗 GitHub: [github.com/sourav2212/BANKBOT](https://github.com/sourav2212/BANKBOT)
- 🌐 Portfolio: [portfoliosouravs.netlify.app](https://portfoliosouravs.netlify.app)

---

## 📄 License

This project is for educational and internship demonstration purposes.
