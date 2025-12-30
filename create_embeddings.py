import requests
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create embeddings locally via HuggingFace Model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Download PDF from internet
pdf_url = "https://www.timelabs.in/resource/hrPolicies/HR-Policy-Manual-Template.pdf"
pdf_path = "HR_Policy_Manual_download.pdf"

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

print("Step 1 PDF downloaded")

# Extract text from PDF
doc = fitz.open(pdf_path)
full_text = ""

for page in doc:
    full_text += page.get_text()

print("Step 2 Text extracted from PDF")

#  Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,           # medium-large chunks for policies
    chunk_overlap=100,        # overlap to preserve context
    separators=["\n\n", "\n", " ", ""]  # splits on paragraphs ,single lines and spaces.
)

docs = splitter.create_documents([full_text])
print(f"Step 3 - Total chunks created: {len(docs)}")

## Below code is to inspect chunks created - optional

chunk_1 = docs[2].page_content
chunk_2 = docs[3].page_content
print("\n========== CHUNK 1 ==========")
print(chunk_1)
print("\n========== CHUNK 2 ==========")
print(chunk_2)
embedding_chunk_1 = embeddings.embed_query(chunk_1)

print("\n========== EMBEDDING FOR CHUNK 1 ==========")
print("Vector length:", len(embedding_chunk_1))
print("First 10 values:", embedding_chunk_1[:10])


#  Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

#  Save FAISS index
vectorstore.save_local("hr_policy_faiss_index")

print("HR Policy RAG index saved to 'hr_policy_faiss_index' local folder")
