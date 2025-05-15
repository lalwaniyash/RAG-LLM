from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
import os
# Load PDF
def load_pdf(file_path):
   reader = PdfReader(file_path)
   text = ""
   for page in reader.pages:
       text += page.extract_text() or ""
   return text
# Chunk text
def chunk_text(text):
   splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   return splitter.split_text(text)
# Embed and store
def embed_and_store(docs, persist_directory="chroma_db"):
   embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   vectordb = Chroma.from_texts(docs, embedding_function, persist_directory=persist_directory)
   vectordb.persist()
   print("Documents embedded and stored!")
if __name__ == "__main__":
   raw_text = load_pdf("sample_docs/tensai.pdf")  # <-- Use your PDF path
   chunks = chunk_text(raw_text)
   embed_and_store(chunks)