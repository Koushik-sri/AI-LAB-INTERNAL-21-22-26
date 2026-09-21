"""FastAPI web server for RAG-Based Question Answering System."""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import DocumentChunker
from vector_store import SimpleVectorStore
from qa_engine import RAGQAEngine

app = FastAPI(title="RAG-Based QA AI Studio", version="1.0.0")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

chunker = DocumentChunker(chunk_size=250, chunk_overlap=30)
store = SimpleVectorStore()

# Ingest sample knowledge base
kb_file = os.path.join(data_dir, "knowledge_base.txt")
if os.path.exists(kb_file):
    with open(kb_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunker.chunk_document(text, source_name="knowledge_base.txt")
    store.add_chunks(chunks)

qa_engine = RAGQAEngine(store)

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    filename: str
    content: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_file = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file)
    return HTMLResponse("<h2>RAG QA System Web App</h2>")

@app.get("/api/status")
def get_status():
    api_key_configured = qa_engine.is_api_key_configured()
    total_chunks = len(store.chunks)
    sources = list(set(c["source"] for c in store.chunks))
    
    return {
        "status": "ONLINE",
        "api_key_configured": api_key_configured,
        "api_key_message": "OPENAI_API_KEY detected in .env" if api_key_configured else "No OPENAI_API_KEY detected in .env. Using built-in high-accuracy vector synthesizer.",
        "total_chunks": total_chunks,
        "indexed_sources": sources
    }

@app.get("/api/documents")
def get_documents():
    docs = {}
    for c in store.chunks:
        src = c["source"]
        if src not in docs:
            docs[src] = []
        docs[src].append({"id": c["id"], "preview": c["text"][:80] + "..."})
    return {"documents": docs, "total_chunks": len(store.chunks)}

@app.post("/api/ingest")
def ingest_document(req: IngestRequest):
    if not req.filename.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Filename and content required.")
    
    new_chunks = chunker.chunk_document(req.content.strip(), source_name=req.filename.strip())
    store.add_chunks(new_chunks)
    return {
        "message": f"Successfully ingested {req.filename}. Added {len(new_chunks)} vector chunks.",
        "chunks_added": len(new_chunks),
        "total_chunks": len(store.chunks)
    }

@app.post("/api/query")
def execute_rag_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = qa_engine.answer_question(req.question.strip())
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
