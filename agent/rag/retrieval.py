# agent/rag/retrieval.py

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class DocChunk:
    id: str          
    source: str      
    text: str


class LocalCorpusRetriever:
  

    def __init__(self, docs_dir: str, chunk_size: int = 400):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size

        self._chunks: List[DocChunk] = []
        texts: List[str] = []

        for fname in os.listdir(docs_dir):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(docs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            base = os.path.splitext(fname)[0]  # marketing_calendar, ...

            # chunk naive: split 400
            i = 0
            while i < len(content):
                chunk_text = content[i : i + chunk_size]
                chunk_id = f"{base}::chunk{len(self._chunks)}"
                self._chunks.append(DocChunk(id=chunk_id, source=base, text=chunk_text))
                texts.append(chunk_text)
                i += chunk_size

        if not self._chunks:
            raise ValueError(f"No .md docs found in {docs_dir}")

        self._vectorizer = TfidfVectorizer()
        self._tfidf = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        q_vec = self._vectorizer.transform([query])
        scores = (self._tfidf @ q_vec.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_idx:
            chunk = self._chunks[int(idx)]
            score = float(scores[int(idx)])
            results.append(
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": score,
                }
            )
        return results
