# Project 02 — RAG-Based Question Answering AI Studio

> **Enterprise Retrieval-Augmented Generation Powered by Google Gemini 2.5 Flash**

---

## 🎯 Overview

The **RAG-Based Question Answering AI Studio** provides grounded document intelligence. It ingests custom knowledge bases, performs automated text chunking with overlap, indexes documents via vector TF-IDF cosine similarity, and synthesizes answers with source citations.

---

## 🏗️ Architecture

```
[Knowledge Base / Document Ingestion]
        │
        ▼
[DocumentChunker (250 chars, 30 overlap)]
        │
        ▼
[SimpleVectorStore (Cosine Similarity Indexing)]
        │
        ▼
[User Question] ──► [TF-IDF Top-K Chunks Retrieval]
        │
        ▼
[Google Gemini 2.5 Flash Grounded Synthesis] ──(Fallback)──► [Local Grounded Synthesizer]
        │
        ▼
[FastAPI REST API (Port 8002)] ──► [Interactive Vector UI]
```

---

## ⚡ Key Features

- **Document Ingestion & Chunking**: Inspect chunk sizes, overlaps, and indexed metadata in real-time.
- **Cosine Vector Search**: Computes similarity scores and match percentages for retrieved text chunks.
- **Grounded Answer Synthesis**: Generates answers strictly grounded in context with `[Source | Chunk ID]` citations.
- **Dynamic Local Synthesizer**: Provides vector similarity results even if no LLM API key is present.
- **Document Management**: Ingest new text documents directly from the web UI.

---

## 🔌 API Endpoints Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the RAG AI Studio web UI. |
| `GET` | `/api/status` | Returns vector store status, chunk counts, and indexed sources. |
| `GET` | `/api/documents` | Lists indexed documents and chunk previews. |
| `POST` | `/api/ingest` | Ingests a new document content into vector store. |
| `POST` | `/api/query` | Executes RAG query and returns grounded answer with citations. |

---

## 🚀 Running Standalone Server

To run Project 02 standalone on port `8002`:

```bash
cd 02_rag_qa/src
python -m uvicorn server:app --host 127.0.0.1 --port 8002
```

Access the UI at: **http://127.0.0.1:8002**

---

## 🧪 Automated Testing

Run unit tests for Project 02:

```bash
python -m pytest 02_rag_qa/tests/
```
