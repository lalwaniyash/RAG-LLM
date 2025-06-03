from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import shutil
 
persist_directory = "chroma_db"
 
def query_from_vector_db(query: str) -> str:
    # Force vector store reload by removing and copying again (optional but safer)
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
    # Clear cache - optional if you see caching issues
    shutil.rmtree(".chroma", ignore_errors=True)
 
    # Always reload Chroma DB from disk
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
 
    retriever = vectordb.as_retriever()
    llm = Ollama(model="llama3")
 
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You must answer ONLY if the information is present in the context below.
        if the answer is even slightly uncertain or not directly found in the context , respond with:
        "Answer not found in the document"
        Context: {context}
        Question: {question}
        """,
    )
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt},
    )
 
    return qa_chain.run(query)
 