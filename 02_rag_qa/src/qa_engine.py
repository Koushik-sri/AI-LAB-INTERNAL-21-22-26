"""RAG QA engine powered by Google Gemini API with grounded generation and citation attribution."""
import os
from typing import Dict, Any, List
from vector_store import SimpleVectorStore

class RAGQAEngine:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def get_gemini_api_key(self) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key or not key.strip():
            # Check local .env file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env_path = os.path.join(base_dir, ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                            key = line.split("=", 1)[1].strip()
                            break
        return key.strip() if key else ""

    def is_api_key_configured(self) -> bool:
        return bool(self.get_gemini_api_key())

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.vector_store.search(question, top_k=top_k)
        
        chunk_details = []
        sources = []
        context_parts = []

        for chunk, score in results:
            match_pct = round(score * 100, 1) if score > 0 else 0.0
            chunk_info = {
                "id": chunk["id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": score,
                "match_percentage": match_pct
            }
            chunk_details.append(chunk_info)

            src_str = f"[{chunk['source']} | Chunk {chunk['id']}]"
            sources.append(src_str)
            context_parts.append(f"{src_str}:\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        # Call Google Gemini API
        api_key = self.get_gemini_api_key()
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                
                prompt = (
                    f"You are an expert AI assistant answering questions grounded strictly in retrieved context documents.\n"
                    f"Answer the user question concisely and accurately.\n"
                    f"Cite sources using [Source | Chunk ID] notation wherever information is referenced.\n\n"
                    f"Context Documents:\n{context}\n\n"
                    f"User Question: {question}\n\n"
                    f"Grounded Answer:"
                )
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                answer = response.text.strip()
                return {
                    "question": question,
                    "answer": answer,
                    "mode": "Google Gemini 2.5 Flash Grounded Synthesis",
                    "api_key_status": "Active (GEMINI_API_KEY)",
                    "sources": list(set(sources)),
                    "retrieved_chunks": chunk_details
                }
            except Exception as e:
                print(f"Gemini API Exception: {e}")

        # Grounded answer synthesis fallback
        lines = []
        for c in chunk_details:
            lines.append(f"- From [{c['source']} | Chunk {c['id']}] (Match: {c['match_percentage']}%):\n  \"{c['text']}\"")
        
        answer = f"Grounded Answer (retrieved from indexed vector corpus):\n\n" + "\n\n".join(lines)
        return {
            "question": question,
            "answer": answer,
            "mode": "Local TF-IDF Vector Synthesizer",
            "api_key_status": "Optional (GEMINI_API_KEY not set)",
            "sources": list(set(sources)),
            "retrieved_chunks": chunk_details
        }
