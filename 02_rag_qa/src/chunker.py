"""Document chunker module for RAG-Based Question Answering System."""
import re
from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, source_name: str = "document") -> List[Dict[str, Any]]:
        # Split document by headers or double newlines first
        sections = re.split(r'\n\s*\n', text)
        chunks = []
        chunk_id = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section fits in chunk_size
            if len(section) <= self.chunk_size:
                chunks.append({
                    "id": chunk_id,
                    "source": source_name,
                    "text": section
                })
                chunk_id += 1
            else:
                # Sliding window chunking
                start = 0
                while start < len(section):
                    end = start + self.chunk_size
                    chunk_text = section[start:end].strip()
                    if chunk_text:
                        chunks.append({
                            "id": chunk_id,
                            "source": source_name,
                            "text": chunk_text
                        })
                        chunk_id += 1
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks
