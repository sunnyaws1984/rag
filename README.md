# 📘 RAG with Gemini + FAISS + Gradio

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using:

- **FAISS** for vector storage  
- **HuggingFace embeddings** (`all-MiniLM-L6-v2`)  
- **Gemini LLM** for generating answers  
- **Gradio** for a web-based UI  

---

## 🚀 Setup Instructions

### 1. Clone the project
```bash
git clone https://github.com/sunnyaws1984/rag.git
cd rag

2. Create a virtual environment (recommended)
python -m venv rag
source rag/bin/activate       # Mac/Linux
source rag/Scripts/activate   # GIT BASH

3. Install dependencies
pip install -r requirements.txt

⚙️ Environment Setup
Create a .env file in the project root:
GEMINI_API_KEY=your_api_key_here
Get your Gemini API key from 👉 Google AI Studio

🛠️ How to Run
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
where Nvidia chips are used ?
when was CUDA introduced ?

FAISS retrieves the relevant document:
Nvidia chips are used in cloud platforms like AWS, Google Cloud, and Azure

📂 Project Structure
rag/
│── create_embeddings.py         # Creates FAISS index
│── retrieval_with_llm.py   # Gradio UI + Gemini for answering
│── requirements.txt
│── .env                    # Store your Gemini API key here
│── faiss_index/            # Saved FAISS vector store
│── README.md