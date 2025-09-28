from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 1️⃣ Sample documents
texts = [
    "NVIDIA Corporation was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem.",
    "NVIDIA is a global leader in GPUs, artificial intelligence, and high-performance computing.",
    "NVIDIA's GeForce series dominates the gaming GPU market.",
    "CUDA, introduced by NVIDIA in 2006, enables general-purpose computing on GPUs.",
    "NVIDIA GPUs are widely used for AI model training and inference.",
    "NVIDIA's A100 and H100 GPUs power modern AI workloads and data centers.",
    "The Blackwell GPU architecture is the next generation of NVIDIA's AI hardware.",
    "NVIDIA reached a market cap rivaling Apple and Microsoft in 2023.",
    "NVIDIA Omniverse is a platform for 3D simulation and digital twin creation.",
    "NVIDIA DGX systems are enterprise-grade AI supercomputers.",
    "TensorRT, cuDNN, and Triton are key components of NVIDIA's AI software stack.",
    "NVIDIA's chips are used in cloud platforms like AWS, Google Cloud, and Azure.",
    "NVIDIA DRIVE is a platform for autonomous vehicle computing.",
    "NVIDIA accelerates innovation in healthcare, robotics, and scientific research.",
    "NVIDIA is a foundational force in the generative AI revolution.",
    "NVIDIA is not just a chipmaker but an AI infrastructure leader."
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