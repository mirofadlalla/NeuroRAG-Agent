import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any


class FaissIndex:
    def __init__(
        self,
        vector_dim: int,
        index_path: str = "data/processed/faiss.index",
        mapping_path: str = "data/processed/faiss_mapping.json"
    ):
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.mapping_path = mapping_path

        self.index = faiss.IndexFlatIP(vector_dim)
        self.mapping: Dict[int, str] = {}
        self.next_id = 0

        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = {int(k): v for k, v in json.load(f).items()}
            self.next_id = max(self.mapping.keys()) + 1 if self.mapping else 0

    def add(self, chunks: List[Dict[str, Any]]):
        """
        chunks: [{chunk_id, embedding}]
        """
        vectors = []
        for chunk in chunks:
            vectors.append(chunk["embedding"])
            self.mapping[self.next_id] = chunk["chunk_id"]
            self.next_id += 1

        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i, faiss_id in enumerate(indices[0]):
            if faiss_id == -1:
                continue
            results.append({
                "chunk_id": self.mapping[faiss_id],
                "score": float(scores[0][i])
            })
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
