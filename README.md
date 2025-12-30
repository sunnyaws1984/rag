# 📘 RAG with Gemini + FAISS + Gradio

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using:

- **FAISS** for vector storage  
- **HuggingFace embeddings** (`all-MiniLM-L6-v2`)  
- **Gemini LLM** for generating answers  
- **Gradio** for a web-based UI  

---

## Setup Instructions

### 1. Clone the project
```bash
git clone https://github.com/sunnyaws1984/rag.git
cd rag

2. Create a virtual environment (recommended)
python -m venv rag
source rag/bin/activate       # Mac/Linux
source rag/Scripts/activate   # GIT BASH

3. Install dependencies:
pip install uv
uv pip install requests pymupdf langchain langchain-community sentence-transformers faiss-cpu gradio google-genai

or
pip install -r requirements.txt

# Environment Setup
Create a .env file in the project root:
GEMINI_API_KEY=your_api_key_here
Get your Gemini API key from Google AI Studio

# How to Run
Step 1: Build FAISS index
python create_embeddings.py

This will:
Split text into chunks
Generate embeddings using HuggingFace
Store them in a local FAISS index (faiss_index/)

Step 2: Start the Gradio UI
python retrieval_with_llm.py

A browser window will open automatically
Or visit: http://127.0.0.1:7860

💻 Example Workflow
In the UI, type a question:
What is Probationary Period  ?
what are exit interviews policy ?
How are vacancies listed in org ?
..


📂 Project Structure
rag/
│── create_embeddings.py         # Creates FAISS index
│── retrieval_with_llm.py   # Gradio UI + Gemini for answering
│── requirements.txt
│── .env                    # Store your Gemini API key here
│── faiss_index/            # Saved FAISS vector store
│── README.md

#####################################################################################

Reference:

Hello world! This is a test of the RecursiveCharacterTextSplitter.

chunk_size = 20
chunk_overlap = 4

Chunk 1: "Hello world! This is "
Chunk 2: " is a test of the Recu"   <- last 4 chars including spaces from Chunk 1 (" is ") appear here

## This is the Process of creating Embeddings.

PDF Text
   ↓
Text Chunks
   ↓
HuggingFaceEmbeddings
   ↓
384-dimensional vectors
   ↓
FAISS index (fast similarity search)

## This is the Process of Retrieval (RAG)
Retrieval converts the user question into an embedding, searches FAISS for nearest vectors, and returns the most relevant document chunks as context for the LLM

User Question
   ↓
Query Embedding
   ↓
HuggingFaceEmbeddings
   ↓
384-dimensional query vector
   ↓
FAISS index (similarity search)
   ↓
Top-K most similar document chunks
   ↓
Context passed to LLM
