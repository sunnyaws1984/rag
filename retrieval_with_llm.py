import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai  # new SDK

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
# Make sure GEMINI_API_KEY is set in .env
# GEMINI_API_KEY=your_key_here
client = genai.Client()  # Reads API key from env variable

# -----------------------------
# Load FAISS vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "hr_policy_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -----------------------------
# Function to answer questions
# -----------------------------
def answer_question(query):
    # Retrieve relevant documents
    results = vectorstore.similarity_search(query, k=3)
    
    # Combine retrieved chunks as context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Construct prompt for Gemini
    prompt = f"""
You are an HR assistant. Answer the user's question strictly based on the HR Policy context below.
Do NOT include any information that is not present in the provided context.
If the answer is not present in the context, respond with: "The answer is not specified in the policy."

Context:
{context}

Question: {query}

Answer:
"""
    # Generate answer using Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text.strip()  # remove extra whitespace

# -----------------------------
# Gradio UI
# -----------------------------
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=1, placeholder="Ask a question about HR policies..."),
    outputs=gr.Textbox(lines=15, label="Answer"),
    title="HR Policy Q&A (RAG + Gemini)",
    description="Ask questions about your company's HR policies. Answers are generated strictly from the policy documents."
)

iface.launch(server_name="0.0.0.0", server_port=7860)