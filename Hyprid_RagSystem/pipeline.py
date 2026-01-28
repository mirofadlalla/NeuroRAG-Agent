import hashlib
import numpy as np
from typing import List, Dict, Any
from indexing.embedder import EmbeddingEngine
from indexing.bm25_index import BM25Indexer
from indexing.faiss_index import FaissIndex
from retrieval.fusion import RRFFusion
from retrieval.rerank import ReRanker

class SmartHybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ):
        self.embedder = EmbeddingEngine()
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}

        self.faiss = FaissIndex(vector_dim=embeddings.shape[1])
        self.faiss.add(chunks)
        self.faiss.save()

        self.bm25 = BM25Indexer(chunks)
        self.fusion = RRFFusion()
        self.reranker = ReRanker()

    def expand_query(self, query: str):
        return [
            query,
            f"Explain {query}",
            f"Detailed information about {query}"
        ]

    def retrieve(self, query: str, top_k: int = 5):
        dense_all, sparse_all = [], []

        for q in self.expand_query(query):
            q_chunk = {
                "chunk_id": hashlib.sha256(q.encode()).hexdigest(),
                "text": q,
                "metadata": {}
            }
            q_vec = self.embedder.embed([q_chunk])

            dense_all.extend(self.faiss.search(q_vec, top_k=10))
            sparse_all.extend(self.bm25.search(q, top_k=10))

        fused = self.fusion.fuse(dense_all, sparse_all, top_k=15)

        # enrich text
        enriched = []
        for r in fused:
            base = self.chunk_lookup[r["chunk_id"]]
            enriched.append({
                "chunk_id": r["chunk_id"],
                "text": base["text"],
                "metadata": base["metadata"],
                "score": r["score"]
            })

        return self.reranker.rerank(query, enriched, top_k=top_k)
