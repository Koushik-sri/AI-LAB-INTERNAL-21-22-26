"""TF-IDF / Cosine Vector Store implementation for RAG QA."""
import math
import re
from typing import List, Dict, Any, Tuple

class SimpleVectorStore:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks.extend(chunks)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf = {}
        total = len(tokens)
        if total == 0:
            return tf
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            tf[token] /= total
        return tf

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = self._tokenize(query)
        q_tf = self._compute_tf(q_tokens)

        scored_results = []
        for chunk in self.chunks:
            c_tokens = self._tokenize(chunk["text"])
            c_tf = self._compute_tf(c_tokens)

            # Cosine similarity on overlapping vocabulary
            dot_product = sum(q_tf[t] * c_tf.get(t, 0) for t in q_tf)
            q_norm = math.sqrt(sum(v ** 2 for v in q_tf.values()))
            c_norm = math.sqrt(sum(v ** 2 for v in c_tf.values()))

            score = dot_product / (q_norm * c_norm) if (q_norm and c_norm) else 0.0
            scored_results.append((chunk, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
