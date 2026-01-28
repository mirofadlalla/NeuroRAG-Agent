import nltk
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

nltk.download("punkt")

class BM25Indexer:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]

        tokenized = [nltk.word_tokenize(t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10):
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {"chunk_id": cid, "score": float(score)}
            for cid, score in ranked
        ]
