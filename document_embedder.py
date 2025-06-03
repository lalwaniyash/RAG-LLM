import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
 
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
 
def embed_and_store_pdf(pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
    os.remove(tmp.name)
 
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["doc_id"] = str(uuid.uuid4())
 
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    vectordb.add_documents(chunks)
 
def delete_pdf_chunks(filename: str) -> int:
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        all_data = vectordb.get()
        ids_to_delete = []
 
        for doc_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
            if metadata and isinstance(metadata, dict):
                source = metadata.get("source")
                print(f"Checking metadata source: {source}")
                if source == filename:
                    ids_to_delete.append(doc_id)
 
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks.")
            return len(ids_to_delete)
        else:
            print("No matching chunks found for deletion.")
            return 0
    except Exception as e:
        print("Error in delete_pdf_chunks:", e)
        return 0