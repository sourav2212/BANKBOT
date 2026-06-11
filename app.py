import streamlit as st
import csv
import os
import sqlite3
import base64
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

web_search = TavilySearchResults(max_results=3)

st.set_page_config(
    page_title="BANKAI — Intelligent Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "profile_pic" not in st.session_state:
    st.session_state.profile_pic = None
if "memory" not in st.session_state:
    st.session_state.memory = InMemoryChatMessageHistory()
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signin"

MAX_QUERIES_PER_SESSION = 20

def init_db():
    conn = sqlite3.connect("bankai.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, username TEXT, message TEXT, score REAL, label TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, username TEXT, question TEXT,
            answer TEXT, rating TEXT, source TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, username TEXT, question TEXT, source TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

is_dark = st.session_state.get("dark_mode", False)

if is_dark:
    theme = """
    .stApp { background: #0f172a !important; color: #e2e8f0 !important; }
    .chat-topbar { background: #1e293b !important; border-color: #334155 !important; }
    .stat-card { background: #1e293b !important; border-color: #334155 !important; }
    .login-card { background: #1e293b !important; }
    .login-logo { color: #f1f5f9 !important; }
    """
else:
    theme = """
    .stApp { background: #f7f8fc !important; color: #0f172a !important; }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@700;800&display=swap');
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
{theme}

.login-card {{
    background: #ffffff; border-radius: 20px; padding: 48px 44px 40px;
    width: 100%; max-width: 420px; box-shadow: 0 32px 80px rgba(0,0,0,0.35);
}}
.chat-topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; background: #0f172a; border-bottom: 1px solid #1e293b;
    position: sticky; top: 0; z-index: 100;
}}
.chat-logo {{
    font-family: 'Sora', sans-serif; font-size: 22px; font-weight: 800;
    color: #f1f5f9; letter-spacing: -1px;
}}
.chat-logo span {{ color: #3b82f6; }}
.user-pill {{
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: #ffffff; font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 20px; border: none;
}}
[data-testid="stSidebar"] {{ background: #0f172a !important; border-right: none !important; }}
[data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
.sidebar-section {{ background: #1e293b; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.sidebar-section-title {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #64748b !important; margin-bottom: 10px;
}}
[data-testid="stChatMessage"] {{ background: transparent !important; border: none !important; padding: 4px 0 !important; }}
.msg-user {{
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: #ffffff; border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; max-width: 70%; margin-left: auto;
    font-size: 14px; line-height: 1.5;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35);
}}
.msg-bot {{
    background: #1e293b; color: #e2e8f0; border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; max-width: 75%; font-size: 14px; line-height: 1.6;
    border: 1px solid #334155; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.badge {{
    display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
    font-weight: 600; padding: 3px 10px; border-radius: 20px;
    margin-top: 6px; margin-right: 4px; letter-spacing: 0.3px;
}}
.badge-positive  {{ background: #dcfce7; color: #166534; }}
.badge-neutral   {{ background: #f1f5f9; color: #475569; }}
.badge-frustrated{{ background: #fee2e2; color: #991b1b; }}
.badge-web       {{ background: #1e40af20; color: #60a5fa; border: 1px solid #1e40af40; }}
.badge-docs      {{ background: #16653420; color: #4ade80; border: 1px solid #16653440; }}
.escalation-banner {{
    background: linear-gradient(135deg, #fef2f2, #fff5f5);
    border: 1px solid #fca5a5; border-left: 4px solid #ef4444;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    font-size: 13px; color: #7f1d1d;
}}
.ratelimit-banner {{
    background: #fff7ed; border: 1px solid #fed7aa; border-left: 4px solid #f97316;
    border-radius: 10px; padding: 12px 18px; margin: 8px 0;
    font-size: 13px; color: #9a3412;
}}
.welcome-card {{
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #4f46e5 100%);
    border-radius: 16px; padding: 24px 28px; color: white; margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(37,99,235,0.35);
}}
.welcome-title {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
.welcome-sub {{ font-size: 13px; opacity: 0.85; line-height: 1.5; }}
.stButton > button {{
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 14px !important; border: none !important; transition: all 0.2s !important;
}}
.stTextInput > div > input {{
    border-radius: 10px !important; border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important; padding: 10px 14px !important;
}}
.block-container {{ padding-top: 0 !important; }}
header {{ display: none !important; }}
footer {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

if is_dark:
    st.markdown("""
    <style>
    textarea { background-color: #1e293b !important; color: #e2e8f0 !important; caret-color: #e2e8f0 !important; }
    textarea:focus { background-color: #1e293b !important; color: #e2e8f0 !important; }
    textarea::placeholder { color: #64748b !important; }
    input { background-color: #1e293b !important; color: #e2e8f0 !important; }
    section[data-testid="stBottom"] { background: #0f172a !important; }
    section[data-testid="stBottom"] > div { background: #0f172a !important; }
    div[data-testid="stChatInput"] { background: #1e293b !important; }
    div[data-testid="stChatInput"] > div { background: #1e293b !important; }
    div[data-testid="stChatInput"] textarea { background-color: #1e293b !important; color: #e2e8f0 !important; caret-color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

DEMO_ACCOUNTS = {
    "sourav": "sourav123",
    "manager": "bank2024",
    "demo": "demo123",
    "admin": "admin123"
}

if not st.session_state.authenticated:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-bottom:28px;">
            <div style="font-family:'Sora',sans-serif; font-size:42px; font-weight:800;
                        color:#0f172a; letter-spacing:-2px; line-height:1;">
                BANK<span style="color:#2563eb;">AI</span>
            </div>
            <div style="font-size:13px; color:#64748b; margin-top:6px; font-weight:400;">
                Union Bank of India — Intelligent Customer Service
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔑  Sign In", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "signin" else "secondary",
                         key="btn_signin"):
                st.session_state.auth_mode = "signin"
                st.rerun()
        with col_b:
            if st.button("✨  Register", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "register" else "secondary",
                         key="btn_register"):
                st.session_state.auth_mode = "register"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "signin":
            st.markdown("""
            <div style="background:#ffffff;border-radius:16px;padding:24px 24px 8px;
                        box-shadow:0 12px 40px rgba(0,0,0,0.10);border:1px solid #e2e8f0;
                        margin-bottom:8px;">
                <div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:4px;">Welcome back 👋</div>
                <div style="font-size:12px;color:#64748b;margin-bottom:16px;">Sign in to your BANKAI account</div>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Enter your username", key="li_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In →", use_container_width=True, type="primary"):
                uname = username.strip().lower()
                if uname in DEMO_ACCOUNTS and DEMO_ACCOUNTS[uname] == password:
                    st.session_state.authenticated = True
                    st.session_state.user_name = username.strip().title()
                    st.session_state.messages = []
                    st.session_state.query_count = 0
                    st.session_state.memory = InMemoryChatMessageHistory()
                    st.session_state.profile_pic = None
                    st.rerun()
                elif uname in st.session_state.get("registered_users", {}):
                    if st.session_state.registered_users[uname]["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_name = st.session_state.registered_users[uname]["name"]
                        st.session_state.profile_pic = st.session_state.registered_users[uname].get("pic")
                        st.session_state.messages = []
                        st.session_state.query_count = 0
                        st.session_state.memory = InMemoryChatMessageHistory()
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password.")
                else:
                    st.error("❌ Account not found. Please register first.")

            st.markdown("""
            <div style="text-align:center;margin-top:14px;font-size:12px;color:#94a3b8;">
                Demo: <b>demo</b> / <b>demo123</b>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#ffffff;border-radius:16px;padding:24px 24px 8px;
                        box-shadow:0 12px 40px rgba(0,0,0,0.10);border:1px solid #e2e8f0;
                        margin-bottom:8px;">
                <div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:4px;">Create your account ✨</div>
                <div style="font-size:12px;color:#64748b;margin-bottom:16px;">Join BANKAI — takes less than a minute</div>
            </div>
            """, unsafe_allow_html=True)

            new_name     = st.text_input("Full Name", placeholder="Your full name", key="reg_name")
            new_username = st.text_input("Username", placeholder="Pick a username", key="reg_user")
            new_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pass")
            new_confirm  = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")

            st.markdown("""<div style="font-size:12px;font-weight:600;color:#374151;
                        text-transform:uppercase;letter-spacing:0.6px;margin:12px 0 6px;">
                Profile Picture (optional)</div>""", unsafe_allow_html=True)
            uploaded_pic = st.file_uploader("", type=["jpg","jpeg","png"], key="reg_pic", label_visibility="collapsed")
            if uploaded_pic:
                st.image(uploaded_pic, width=80)
                st.caption("✅ Photo selected")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account →", use_container_width=True, type="primary"):
                if not new_name or not new_username or not new_password:
                    st.warning("⚠️ Please fill all required fields.")
                elif len(new_password) < 6:
                    st.warning("⚠️ Password must be at least 6 characters.")
                elif new_password != new_confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    ukey = new_username.strip().lower()
                    if "registered_users" not in st.session_state:
                        st.session_state.registered_users = {}
                    if ukey in DEMO_ACCOUNTS or ukey in st.session_state.registered_users:
                        st.error("❌ Username already taken.")
                    else:
                        pic_data = uploaded_pic.read() if uploaded_pic else None
                        st.session_state.registered_users[ukey] = {
                            "password": new_password,
                            "name": new_name.strip().title(),
                            "pic": pic_data
                        }
                        st.success(f"✅ Account created! Sign in as: {new_username}")
                        st.session_state.auth_mode = "signin"
                        st.rerun()

        st.markdown("""
        <div style="text-align:center;margin-top:20px;font-size:11px;color:#94a3b8;">
            🔒 Secured with end-to-end encryption &nbsp;·&nbsp; © 2024 BankAI
        </div>
        """, unsafe_allow_html=True)

else:
    user_name = st.session_state.user_name

    @st.cache_resource
    def load_rag():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="vectorstore/",
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0)
        BANK_PROMPT = """You are BANKAI, an intelligent customer service assistant for Union Bank of India.
Always refer to the bank as "Union Bank of India". Be concise and professional.
If the answer is not in the context, say:
"I don't have that specific information for Union Bank of India. Please visit your nearest Union Bank branch or call 1800 22 2244."

Conversation History:
{history}

Bank Information:
{context}

Customer Question: {question}

Answer:"""
        prompt = PromptTemplate(template=BANK_PROMPT, input_variables=["history", "context", "question"])
        return retriever, llm, prompt

    @st.cache_resource
    def load_sentiment():
        return SentimentIntensityAnalyzer()

    def log_escalation_db(message, sentiment):
        conn = sqlite3.connect("bankai.db")
        c = conn.cursor()
        c.execute("INSERT INTO escalations (timestamp, username, message, score, label) VALUES (?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, message, sentiment["compound"], sentiment["label"]))
        conn.commit(); conn.close()

    def log_feedback_db(question, answer, rating, source):
        conn = sqlite3.connect("bankai.db")
        c = conn.cursor()
        c.execute("INSERT INTO feedback (timestamp, username, question, answer, rating, source) VALUES (?,?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, question, answer, rating, source))
        conn.commit(); conn.close()

    def log_query_db(question, source):
        conn = sqlite3.connect("bankai.db")
        c = conn.cursor()
        c.execute("INSERT INTO query_log (timestamp, username, question, source) VALUES (?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, question, source))
        conn.commit(); conn.close()

    def get_answer(question, retriever, llm, prompt):
        try:
            docs = retriever.invoke(question)
            context = "\n".join([d.page_content for d in docs])
            history_msgs = st.session_state.memory.messages
            history = ""
            for m in history_msgs[-6:]:
                role = "Customer" if m.type == "human" else "Assistant"
                history += f"{role}: {m.content}\n"
            chain = prompt | llm
            result = chain.invoke({"context": context, "question": question, "history": history})
            rag_answer = result.content
            fallback_phrases = ["i don't have that information", "please visit your nearest branch",
                                "i don't know", "not available", "no information"]
            if any(phrase in rag_answer.lower() for phrase in fallback_phrases):
                web_results = web_search.invoke(question)
                web_context = "\n\n".join([r["content"] for r in web_results])
                web_result = chain.invoke({"context": web_context, "question": question, "history": history})
                answer = web_result.content
                source = "web"
            else:
                answer = rag_answer
                source = "documents"
            st.session_state.memory.add_user_message(question)
            st.session_state.memory.add_ai_message(answer)
            return answer, source
        except Exception as e:
            return f"I'm having trouble connecting right now. Please try again. (Error: {str(e)})", "unknown"

    def analyze_sentiment(text, analyzer):
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound <= -0.6:
            return {"compound": round(compound, 3), "label": "FRUSTRATED", "should_escalate": True}
        elif compound < 0.05:
            return {"compound": round(compound, 3), "label": "NEUTRAL", "should_escalate": False}
        else:
            return {"compound": round(compound, 3), "label": "POSITIVE", "should_escalate": False}

    retriever, llm, prompt = load_rag()
    analyzer = load_sentiment()

    # Top bar with profile pic
    pic_html = ""
    if st.session_state.get("profile_pic"):
        pic_b64 = base64.b64encode(st.session_state.profile_pic).decode()
        pic_html = f'<img src="data:image/jpeg;base64,{pic_b64}" style="width:34px;height:34px;border-radius:50%;border:2px solid #3b82f6;object-fit:cover;">'
    else:
        pic_html = f'<div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#fff;">{user_name[0].upper()}</div>'

    st.markdown(f"""
    <div class="chat-topbar">
        <div class="chat-logo">BANK<span>AI</span>
            <span style="font-size:11px;font-weight:400;color:#475569;margin-left:10px;letter-spacing:0;">Union Bank of India</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e;"></div>
            <span style="font-size:12px;color:#94a3b8;font-weight:500;">AI Online</span>
            {pic_html}
            <div class="user-pill">👤 {user_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    toggle_col, _ = st.columns([0.06, 0.94])
    with toggle_col:
        if st.button("☀️" if st.session_state.dark_mode else "🌙", help="Toggle dark/light mode", key="dark_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    with st.sidebar:
        st.markdown("""<div style="padding:20px 8px 10px;">
            <div style="font-family:'Sora',sans-serif;font-size:20px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">BANKAI</div>
            <div style="font-size:11px;color:#475569;margin-top:2px;">Union Bank of India</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        total_q    = len([m for m in st.session_state.messages if m["role"] == "user"])
        escalated  = len([m for m in st.session_state.messages if m.get("sentiment", {}).get("should_escalate")])
        positive_c = len([m for m in st.session_state.messages if m.get("sentiment", {}).get("label") == "POSITIVE"])
        web_count  = len([m for m in st.session_state.messages if m.get("source", "") == "web"])

        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Session Stats</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
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
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#60a5fa;">{web_count}</div>
                    <div style="font-size:10px;color:#475569;">Web hits</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pct = int((st.session_state.query_count / MAX_QUERIES_PER_SESSION) * 100)
        bar_color = "#22c55e" if pct < 60 else "#f97316" if pct < 85 else "#ef4444"
        remaining = MAX_QUERIES_PER_SESSION - st.session_state.query_count
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Query Limit</div>
            <div style="background:#0f172a;border-radius:6px;height:6px;margin-bottom:8px;overflow:hidden;">
                <div style="background:linear-gradient(90deg,#3b82f6,#6366f1);width:{pct}%;height:100%;border-radius:6px;"></div>
            </div>
            <div style="font-size:11px;color:#64748b;">
                {st.session_state.query_count} / {MAX_QUERIES_PER_SESSION} used &nbsp;·&nbsp;
                <span style="color:{bar_color};font-weight:600;">{remaining} remaining</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class="sidebar-section"><div class="sidebar-section-title">Quick Topics</div></div>""", unsafe_allow_html=True)

        quick_topics = [
            ("🏦", "Open a savings account"), ("💳", "Block my debit card"),
            ("🏠", "Apply for home loan"),    ("📱", "Reset internet banking"),
            ("💰", "Fixed deposit rates"),    ("📋", "KYC documents needed"),
            ("💸", "NEFT and RTGS charges"),  ("📞", "File a complaint"),
            ("🏧", "ATM withdrawal limit"),   ("📊", "Personal loan eligibility"),
        ]
        for icon, topic in quick_topics:
            if st.button(f"{icon} {topic}", use_container_width=True, key=f"qt_{topic}"):
                st.session_state.pending_input = topic

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.memory = InMemoryChatMessageHistory()
            st.rerun()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.memory = InMemoryChatMessageHistory()
            st.rerun()

    main_col, _ = st.columns([1, 0.001])
    with main_col:
        st.markdown("<div style='padding: 20px 24px 0;'>", unsafe_allow_html=True)

        if len(st.session_state.messages) == 0:
            greeting = get_greeting()
            st.markdown(f"""
            <div class="welcome-card" style="position:relative;overflow:hidden;">
                <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                            background:rgba(255,255,255,0.06);border-radius:50%;"></div>
                <div style="position:absolute;bottom:-40px;right:60px;width:80px;height:80px;
                            background:rgba(255,255,255,0.04);border-radius:50%;"></div>
                <div class="welcome-title">{greeting}, {user_name}! 👋</div>
                <div class="welcome-sub">
                    I'm BANKAI, Union Bank of India's AI assistant. Ask me anything about
                    accounts, loans, KYC, transfers, or any other banking service. Available 24×7.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">
                <span style="background:#1e40af20;color:#60a5fa;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;border:1px solid #1e40af40;">📄 RAG-powered answers</span>
                <span style="background:#16653420;color:#4ade80;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;border:1px solid #16653440;">🧠 Sentiment detection</span>
                <span style="background:#92400e20;color:#fb923c;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;border:1px solid #92400e40;">⚡ Smart escalation</span>
                <span style="background:#4c1d9520;color:#a78bfa;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;border:1px solid #4c1d9540;">🌐 Web search fallback</span>
                <span style="background:#78350f20;color:#fbbf24;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;border:1px solid #78350f40;">🧠 Conversation memory</span>
            </div>
            """, unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
                    if "sentiment" in msg:
                        s = msg["sentiment"]
                        if s["label"] == "FRUSTRATED":
                            st.markdown(f'<span class="badge badge-frustrated">😤 Frustrated · {s["compound"]}</span>', unsafe_allow_html=True)
                        elif s["label"] == "POSITIVE":
                            st.markdown(f'<span class="badge badge-positive">😊 Positive · {s["compound"]}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="badge badge-neutral">😐 Neutral · {s["compound"]}</span>', unsafe_allow_html=True)
                    if "source" in msg:
                        src = msg["source"]
                        if src == "web":
                            st.markdown('<span class="badge badge-web">🌐 web search</span>', unsafe_allow_html=True)
                        elif src == "documents":
                            st.markdown('<span class="badge badge-docs">📄 documents</span>', unsafe_allow_html=True)
                    if msg["role"] == "assistant" and "feedback" not in msg:
                        col_t, col_f, col_sp = st.columns([0.08, 0.08, 0.84])
                        with col_t:
                            if st.button("👍", key=f"thumb_up_{i}"):
                                log_feedback_db(st.session_state.messages[i-1]["content"] if i > 0 else "", msg["content"], "positive", msg.get("source", ""))
                                st.session_state.messages[i]["feedback"] = "positive"
                                st.rerun()
                        with col_f:
                            if st.button("👎", key=f"thumb_dn_{i}"):
                                log_feedback_db(st.session_state.messages[i-1]["content"] if i > 0 else "", msg["content"], "negative", msg.get("source", ""))
                                st.session_state.messages[i]["feedback"] = "negative"
                                st.rerun()
                    elif "feedback" in msg:
                        emoji = "✅" if msg["feedback"] == "positive" else "❌"
                        st.caption(f"{emoji} Feedback recorded")

        st.markdown("</div>", unsafe_allow_html=True)

    def process_input(user_input):
        if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
            st.markdown("""<div class="ratelimit-banner">🚫 <b>Session limit reached</b> — You have used all 20 queries. Please sign out and sign back in.</div>""", unsafe_allow_html=True)
            return

        sentiment = analyze_sentiment(user_input, analyzer)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1

        if sentiment["should_escalate"]:
            bot_response = (f"I understand your frustration, {user_name}, and I sincerely apologize on behalf of Union Bank of India. "
                           "I'm escalating this to a senior agent right away. Expected response time: under 5 minutes. You can also call 1800 22 2244.")
            source = "escalated"
            log_escalation_db(user_input, sentiment)
        else:
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div style="display:flex;align-items:center;gap:10px;padding:12px 18px;
                        background:#1e293b;border-radius:18px 18px 18px 4px;
                        max-width:220px;border:1px solid #334155;margin:4px 0;">
                <div style="display:flex;gap:5px;align-items:center;">
                    <span style="width:8px;height:8px;border-radius:50%;background:#3b82f6;display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:#3b82f6;display:inline-block;"></span>
                    <span style="width:8px;height:8px;border-radius:50%;background:#3b82f6;display:inline-block;"></span>
                </div>
                <span style="font-size:13px;color:#64748b;font-weight:500;">BANKAI is thinking...</span>
            </div>
            """, unsafe_allow_html=True)
            bot_response, source = get_answer(user_input, retriever, llm, prompt)
            log_query_db(user_input, source)
            typing_placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": bot_response, "sentiment": sentiment, "source": source})

        if sentiment["should_escalate"]:
            st.markdown("""<div class="escalation-banner">⚠️ <b>Escalated to human agent</b> — A senior representative will contact you shortly.</div>""", unsafe_allow_html=True)

    if "pending_input" in st.session_state and st.session_state.pending_input:
        topic = st.session_state.pending_input
        st.session_state.pending_input = None
        process_input(topic)
        st.rerun()

    if user_input := st.chat_input(f"Ask your banking question, {user_name}..."):
        process_input(user_input)
        st.rerun()