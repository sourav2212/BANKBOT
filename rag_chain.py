import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_groq import ChatGroq

load_dotenv()  # loads your .env file

# Load the saved vector database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="vectorstore/",
    embedding_function=embeddings
)

# This prompt tells the LLM to ONLY use the bank's documents
# and not make things up from its general knowledge
BANK_PROMPT = """You are a helpful customer service assistant for a bank.
Answer the customer's question ONLY using the information provided below.
If the answer is not in the provided information, say:
"I don't have that information. Please visit your nearest branch or call our helpline."
Do NOT make up any information.

Bank information:
{context}

Customer question: {question}

Your answer:"""

prompt = PromptTemplate(
    template=BANK_PROMPT,
    input_variables=["context", "question"]
)

# Install langchain-groq first: pip install langchain-groq
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=MODEL_NAME,
    temperature=0                 # 0 = consistent, factual answers
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

def get_answer(question: str) -> str:
    """Takes a customer question, returns answer based on bank PDFs"""
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    try:
        result = qa_chain.invoke({"context": context, "question": question})
    except Exception as exc:
        print(f"RAG error: {exc}")
        return (
            "Sorry, I couldn\'t retrieve an answer from the bank documents right now. "
            "Please try again later."
        )
    if isinstance(result, dict):
        return result.get("text", "Sorry, I could not generate an answer.")
    return str(result)


# Test it
if __name__ == "__main__":
    test_q = "What documents do I need to open a savings account?"
    print(f"Question: {test_q}")
    print(f"Answer: {get_answer(test_q)}")
