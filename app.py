import streamlit as st
import csv
import os
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BANKAI — Intelligent Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp { background: #f7f8fc !important; font-family: 'Inter', sans-serif; }

/* ── LOGIN PAGE ── */
.login-wrap {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
}
.login-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 48px 44px 40px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 32px 80px rgba(0,0,0,0.35);
}
.login-logo {
    font-family: 'Sora', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -1.5px;
    margin-bottom: 4px;
}
.login-tagline {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 32px;
    font-weight: 400;
}
.login-label {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
    display: block;
}

/* ── CHAT HEADER ── */
.chat-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    position: sticky;
    top: 0;
    z-index: 100;
}
.chat-logo {
    font-family: 'Sora', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -1px;
}
.chat-logo span { color: #2563eb; }
.user-pill {
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid #bfdbfe;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.sidebar-section {
    background: #1e293b;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.sidebar-section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b !important;
    margin-bottom: 10px;
}
.quick-chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8 !important;
    font-size: 12px;
    padding: 5px 10px;
    border-radius: 6px;
    margin: 3px;
    cursor: pointer;
    transition: all 0.15s;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
.msg-user {
    background: #2563eb;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 70%;
    margin-left: auto;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}
.msg-bot {
    background: #ffffff;
    color: #1e293b;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── SENTIMENT BADGES ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 6px;
    letter-spacing: 0.3px;
}
.badge-positive { background: #dcfce7; color: #166534; }
.badge-neutral   { background: #f1f5f9; color: #475569; }
.badge-frustrated{ background: #fee2e2; color: #991b1b; }

/* ── ESCALATION BANNER ── */
.escalation-banner {
    background: linear-gradient(135deg, #fef2f2, #fff5f5);
    border: 1px solid #fca5a5;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
    color: #7f1d1d;
}

/* ── STAT CARDS ── */
.stat-row { display: flex; gap: 10px; margin-bottom: 16px; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.stat-num { font-size: 24px; font-weight: 700; color: #0f172a; }
.stat-lbl { font-size: 11px; color: #64748b; font-weight: 500; margin-top: 2px; }

/* ── WELCOME CARD ── */
.welcome-card {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 20px;
}
.welcome-title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
.welcome-sub { font-size: 13px; opacity: 0.85; line-height: 1.5; }

/* ── INPUT AREA ── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    background: #ffffff !important;
    transition: border-color 0.2s;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTextInput > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.stTextInput > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
div[data-baseweb="tab-list"] { gap: 4px; }
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.block-container { padding-top: 0 !important; }
header { display: none !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Demo accounts — add more as needed
DEMO_ACCOUNTS = {
    "sourav": "sourav123",
    "manager": "bank2024",
    "demo": "demo123",
    "admin": "admin123"
}

# ── LOGIN PAGE ─────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-bottom: 32px;">
            <div style="font-family:'Sora',sans-serif; font-size:42px; font-weight:800;
                        color:#0f172a; letter-spacing:-2px; line-height:1;">
                BANK<span style="color:#2563eb;">AI</span>
            </div>
            <div style="font-size:13px; color:#64748b; margin-top:6px; font-weight:400;">
                Intelligent Customer Service Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div style="background:#ffffff; border-radius:20px; padding:36px 36px 28px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.12); border:1px solid #e2e8f0;">
            """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["Sign In", "New Account"])

            with tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter your username", key="li_user")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Sign In →", use_container_width=True, type="primary"):
                    uname = username.strip().lower()
                    if uname in DEMO_ACCOUNTS and DEMO_ACCOUNTS[uname] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_name = username.strip().title()
                        st.session_state.messages = []
                        st.rerun()
                    elif uname in st.session_state.get("registered_users", {}):
                        if st.session_state.registered_users[uname]["password"] == password:
                            st.session_state.authenticated = True
                            st.session_state.user_name = st.session_state.registered_users[uname]["name"]
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
                    else:
                        st.error("Account not found. Please register first.")

                st.markdown("""
                <div style="text-align:center; margin-top:16px; font-size:12px; color:#94a3b8;">
                    Demo: username <b>demo</b> / password <b>demo123</b>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                new_name     = st.text_input("Full Name", placeholder="Your full name", key="reg_name")
                new_username = st.text_input("Choose Username", placeholder="Pick a username", key="reg_user")
                new_password = st.text_input("Create Password", type="password", placeholder="Min 6 characters", key="reg_pass")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Create Account →", use_container_width=True, type="primary"):
                    if not new_name or not new_username or not new_password:
                        st.warning("Please fill all fields.")
                    elif len(new_password) < 6:
                        st.warning("Password must be at least 6 characters.")
                    else:
                        ukey = new_username.strip().lower()
                        if "registered_users" not in st.session_state:
                            st.session_state.registered_users = {}
                        if ukey in DEMO_ACCOUNTS or ukey in st.session_state.registered_users:
                            st.error("Username already taken. Choose another.")
                        else:
                            st.session_state.registered_users[ukey] = {
                                "password": new_password,
                                "name": new_name.strip().title()
                            }
                            st.success(f"Account created! Sign in with username: {new_username}")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-top:20px; font-size:11px; color:#94a3b8;">
            🔒 Secured with end-to-end encryption &nbsp;·&nbsp; © 2024 BankAI
        </div>
        """, unsafe_allow_html=True)

# ── MAIN CHAT APP ──────────────────────────────────────────────────────────────
else:
    user_name = st.session_state.user_name

    # ── Load models ──
    @st.cache_resource
    def load_rag():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="C:/Users/advsu/bank-bot/bank-bot/vectorstore/",
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        BANK_PROMPT = """You are a helpful, professional customer service assistant for a bank.
Answer ONLY using the bank information provided below. Be concise and clear.
If the answer is not in the context, say: "I don't have that information. Please visit your nearest branch or call our helpline."

Bank Information:
{context}

Customer Question: {question}

Answer:"""
        prompt = PromptTemplate(template=BANK_PROMPT, input_variables=["context", "question"])
        return retriever, (prompt | llm)

    @st.cache_resource
    def load_sentiment():
        return SentimentIntensityAnalyzer()

    def get_answer(question, retriever, chain):
        try:
            docs = retriever.invoke(question)
            context = "\n".join([d.page_content for d in docs])
            result = chain.invoke({"context": context, "question": question})
            return result.content
        except Exception as e:
            return f"I'm having trouble connecting right now. Please try again. (Error: {str(e)})"

    def analyze_sentiment(text, analyzer):
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound <= -0.6:
            return {"compound": round(compound, 3), "label": "FRUSTRATED", "should_escalate": True}
        elif compound < 0.05:
            return {"compound": round(compound, 3), "label": "NEUTRAL", "should_escalate": False}
        else:
            return {"compound": round(compound, 3), "label": "POSITIVE", "should_escalate": False}

    def log_escalation(message, sentiment):
        log_file = "escalation_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "User", "Message", "Score", "Label"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_name,
                message,
                sentiment["compound"],
                sentiment["label"]
            ])

    retriever, chain = load_rag()
    analyzer         = load_sentiment()

    # ── Top bar ──
    st.markdown(f"""
    <div class="chat-topbar">
        <div class="chat-logo">BANK<span>AI</span></div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;
                        box-shadow:0 0 6px #22c55e;"></div>
            <span style="font-size:12px;color:#64748b;font-weight:500;">AI Online</span>
            <div class="user-pill">👤 {user_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 8px 10px;">
            <div style="font-family:'Sora',sans-serif;font-size:20px;font-weight:800;
                        color:#f1f5f9;letter-spacing:-0.5px;">BANKAI</div>
            <div style="font-size:11px;color:#475569;margin-top:2px;">Control Panel</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Stats
        total_q    = len([m for m in st.session_state.messages if m["role"] == "user"])
        escalated  = len([m for m in st.session_state.messages
                         if m.get("sentiment", {}).get("should_escalate")])
        positive_c = len([m for m in st.session_state.messages
                         if m.get("sentiment", {}).get("label") == "POSITIVE"])

        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Session Stats</div>
            <div style="display:flex;gap:8px;">
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#f1f5f9;">{total_q}</div>
                    <div style="font-size:10px;color:#475569;">Queries</div>
                </div>
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#ef4444;">{escalated}</div>
                    <div style="font-size:10px;color:#475569;">Escalated</div>
                </div>
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#22c55e;">{positive_c}</div>
                    <div style="font-size:10px;color:#475569;">Positive</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick topics
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Quick Topics</div>
        </div>
        """, unsafe_allow_html=True)

        quick_topics = [
            ("🏦", "Open a savings account"),
            ("💳", "Block my debit card"),
            ("🏠", "Apply for home loan"),
            ("📱", "Reset internet banking"),
            ("💰", "Fixed deposit rates"),
            ("📋", "KYC documents needed"),
            ("💸", "NEFT and RTGS charges"),
            ("📞", "File a complaint"),
            ("🏧", "ATM withdrawal limit"),
            ("📊", "Personal loan eligibility"),
        ]

        for icon, topic in quick_topics:
            if st.button(f"{icon} {topic}", use_container_width=True, key=f"qt_{topic}"):
                st.session_state.pending_input = topic

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.session_state.messages = []
            st.rerun()

    # ── Main chat area ──
    main_col, _ = st.columns([1, 0.001])
    with main_col:
        st.markdown("<div style='padding: 20px 24px 0;'>", unsafe_allow_html=True)

        # Welcome card (only when no messages)
        if len(st.session_state.messages) == 0:
            st.markdown(f"""
            <div class="welcome-card">
                <div class="welcome-title">👋 Welcome back, {user_name}!</div>
                <div class="welcome-sub">
                    I'm your AI banking assistant. Ask me anything about accounts,
                    loans, KYC, transfers, or any other banking service.
                    I'm available 24×7.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature chips
            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">
                <span style="background:#eff6ff;color:#1d4ed8;font-size:12px;font-weight:600;
                             padding:6px 14px;border-radius:20px;border:1px solid #bfdbfe;">
                    📄 RAG-powered answers
                </span>
                <span style="background:#f0fdf4;color:#166534;font-size:12px;font-weight:600;
                             padding:6px 14px;border-radius:20px;border:1px solid #bbf7d0;">
                    🧠 Sentiment detection
                </span>
                <span style="background:#fff7ed;color:#9a3412;font-size:12px;font-weight:600;
                             padding:6px 14px;border-radius:20px;border:1px solid #fed7aa;">
                    ⚡ Smart escalation
                </span>
                <span style="background:#faf5ff;color:#6b21a8;font-size:12px;font-weight:600;
                             padding:6px 14px;border-radius:20px;border:1px solid #e9d5ff;">
                    🔒 Secure & private
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="msg-bot">{msg["content"]}</div>',
                                unsafe_allow_html=True)
                    if "sentiment" in msg:
                        s = msg["sentiment"]
                        if s["label"] == "FRUSTRATED":
                            st.markdown(f'<span class="badge badge-frustrated">😤 {s["label"]} · {s["compound"]}</span>',
                                        unsafe_allow_html=True)
                        elif s["label"] == "POSITIVE":
                            st.markdown(f'<span class="badge badge-positive">😊 {s["label"]} · {s["compound"]}</span>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="badge badge-neutral">😐 {s["label"]} · {s["compound"]}</span>',
                                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle quick topic click ──
    def process_input(user_input):
        sentiment = analyze_sentiment(user_input, analyzer)
        st.session_state.messages.append({"role": "user", "content": user_input})

        if sentiment["should_escalate"]:
            bot_response = (
                f"I understand your frustration, {user_name}, and I sincerely apologize. "
                "I'm escalating this to a senior agent right away who will resolve your issue personally. "
                "Expected response time: under 5 minutes."
            )
            log_escalation(user_input, sentiment)
        else:
            with st.spinner("Searching knowledge base..."):
                bot_response = get_answer(user_input, retriever, chain)

        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response,
            "sentiment": sentiment
        })

        if sentiment["should_escalate"]:
            st.markdown("""
            <div class="escalation-banner">
                ⚠️ <b>Escalated to human agent</b> — A senior representative
                will contact you shortly. Your complaint has been logged.
            </div>
            """, unsafe_allow_html=True)

    # Handle quick topic sidebar button
    if "pending_input" in st.session_state and st.session_state.pending_input:
        topic = st.session_state.pending_input
        st.session_state.pending_input = None
        process_input(topic)
        st.rerun()

    # Chat input box
    if user_input := st.chat_input(f"Ask your banking question, {user_name}..."):
        process_input(user_input)
        st.rerun()