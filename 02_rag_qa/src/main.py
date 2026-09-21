"""Main entry point for RAG-Based Question Answering System."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import DocumentChunker
from vector_store import SimpleVectorStore
from qa_engine import RAGQAEngine

def main():
    print("=" * 60)
    print("   RAG-BASED QUESTION ANSWERING SYSTEM - PROJECT 02")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "knowledge_base.txt")

    if not os.path.exists(data_file):
        print(f"[!] Data file not found at {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        kb_text = f.read()

    chunker = DocumentChunker(chunk_size=250, chunk_overlap=30)
    chunks = chunker.chunk_document(kb_text, source_name="knowledge_base.txt")
    print(f"[*] Ingested document. Created {len(chunks)} chunks.")

    store = SimpleVectorStore()
    store.add_chunks(chunks)
    qa_engine = RAGQAEngine(store)

    sample_questions = [
        "What are the Acme Corp AI usage guidelines regarding PII?",
        "Which databases and vector stores are used in the system architecture?",
        "What approval is required for financial refunds over $500?"
    ]

    for idx, q in enumerate(sample_questions, 1):
        print(f"\n[{idx}] Question: '{q}'")
        res = qa_engine.answer_question(q)
        print(f"    Answer: {res['answer']}")
        print(f"    Sources: {', '.join(res['sources'])}")
        print("-" * 60)

    print("\n[OK] Project 02 RAG QA execution finished successfully.")

if __name__ == "__main__":
    main()
