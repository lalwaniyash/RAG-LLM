from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from document_embedder import embed_and_store_pdf, delete_pdf_chunks
from rag_qa import query_from_vector_db
 
app = FastAPI()
 
# CORS configuration (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    embed_and_store_pdf(content, file.filename)
    return {
        "status": "success",
        "message": f"{file.filename} uploaded and embedded"
    }
 
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    answer = query_from_vector_db(query)
    return {"answer": answer}
 
@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    deleted_count = delete_pdf_chunks(filename)
    if deleted_count > 0 :
        return {
            "status": "success",
            "deleted_chunks": deleted_count,
            "message": f"{deleted_count} chunks deleted for {filename}"
        }
    else:
        return{"status":"not found","message":f"No chunks found for {filename}"}