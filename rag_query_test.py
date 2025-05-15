from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import requests
# Load Chroma DB
def get_relevant_chunks(query, persist_directory="chroma_db", k=3):
   embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
   results = vectordb.similarity_search(query, k=k)
   return [doc.page_content for doc in results]
# Call LLaMA via Ollama
def call_llama(prompt):
   response = requests.post("http://localhost:11434/api/generate", json={
       "model": "llama3",
       "prompt": prompt,
       "stream": False
   })
   return response.json()["response"]
# Main RAG flow
def ask_question(query):
   print(f"[+] Searching for: {query}")
   chunks = get_relevant_chunks(query)
   print(f"[+] Retrieved {len(chunks)} chunks")
   context = "\n\n".join(chunks)
   final_prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}
Question: {query}
Answer:"""
   answer = call_llama(final_prompt)
   print("\n[+] Answer:")
   print(answer)
if __name__ == "__main__":
   user_query = input("Ask your question: ")
   ask_question(user_query)