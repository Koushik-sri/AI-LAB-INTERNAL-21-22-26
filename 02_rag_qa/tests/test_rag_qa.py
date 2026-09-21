import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chunker import DocumentChunker
from vector_store import SimpleVectorStore
from qa_engine import RAGQAEngine

def test_chunker():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "Sentence one. Sentence two is longer. Sentence three is here."
    chunks = chunker.chunk_document(text, "test_doc")
    assert len(chunks) > 0
    assert chunks[0]["source"] == "test_doc"

def test_vector_store():
    store = SimpleVectorStore()
    store.add_chunks([
        {"id": 0, "source": "doc1", "text": "Python programming language for AI"},
        {"id": 1, "source": "doc2", "text": "Cooking delicious pasta recipes"}
    ])
    results = store.search("Python AI programming", top_k=1)
    assert len(results) == 1
    assert results[0][0]["id"] == 0

def test_rag_qa_engine():
    store = SimpleVectorStore()
    store.add_chunks([
        {"id": 0, "source": "doc1", "text": "Acme Corp refund policy permits refunds within 30 days."}
    ])
    qa = RAGQAEngine(store)
    res = qa.answer_question("What is the refund policy?")
    assert len(res["sources"]) > 0
    assert "30 days" in res["answer"] or "refund" in res["answer"]
