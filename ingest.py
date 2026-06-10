import fitz  # pymupdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("Step 1: Loading PDFs from docs/ folder...")

docs_folder = "docs/"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(docs_folder, filename)
        print(f"  Reading: {filename}")
        
        pdf = fitz.open(filepath)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            
            if text.strip():  # only add if page has actual text
                all_documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num + 1}
                ))
        pdf.close()

print(f"Loaded {len(all_documents)} pages with text")

if len(all_documents) == 0:
    print("ERROR: No text found in any PDF!")
    print("Your PDFs might be scanned images.")
    print("Solution: Download a text-based PDF instead.")
    exit()

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks")

print("Step 3: Saving to ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore/"
)

print("Done! Vector store saved successfully.")