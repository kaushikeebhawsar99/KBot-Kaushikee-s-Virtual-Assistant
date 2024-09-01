from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import os, warnings
warnings.filterwarnings('ignore')

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Load the PDF
loader = PyPDFLoader("Kaushikee_Bhawsar_Resume.pdf")
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
  embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
  print("Embeddings created successfully")
except Exception as e:
  print(f"Error creating embeddings: {e}")

  # Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")
# Add documents to the vector store
vector_store.add_documents(documents=texts)

# Save the embedding
vector_store.persist()