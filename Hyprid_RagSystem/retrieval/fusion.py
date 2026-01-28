from collections import defaultdict
from typing import List, Dict, Any

class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 20
    ):
        scores = defaultdict(float)

        for rank, r in enumerate(dense_results, start=1):
            scores[r["chunk_id"]] += 1 / (self.k + rank)

        for rank, r in enumerate(sparse_results, start=1):
            scores[r["chunk_id"]] += 1 / (self.k + rank)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": cid, "score": float(score)} for cid, score in fused[:top_k]]
