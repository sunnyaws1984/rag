# create_embeddings.py
# pip install sentence-transformers langchain faiss-cpu

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 1️⃣ Sample documents
texts = [
    "The capital of Japan is Tokyo.",
    "Python is a programming language used in AI.",
    "The Amazon River is the largest river by discharge.",
    "The Eiffel Tower is in Paris, France.",
    "Our Favourite player is Abhishek Sharma",
    "The Great Wall of China is visible from space."
]

# 2️⃣ Split text
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
print ("Splitter",splitter)
docs = splitter.create_documents(texts)
print ("Docs", docs)
# 3️⃣ Create embeddings locally
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print ("Embeddings",embeddings)
# 4️⃣ Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Save vectorstore to disk
vectorstore.save_local("faiss_index")
print("✅ Embeddings saved to 'faiss_index' folder!")