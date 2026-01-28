import os
import numpy as np
import nltk
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

nltk.download("punkt")


# Embedding Cache
class EmbeddingCache:
    def __init__(self, cache_dir: str = "data/processed/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, text_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def exists(self, text_hash: str) -> bool:
        return os.path.exists(self._path(text_hash))

    def load(self, text_hash: str) -> np.ndarray:
        return np.load(self._path(text_hash))

    def save(self, text_hash: str, vector: np.ndarray):
        np.save(self._path(text_hash), vector)


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        cache: EmbeddingCache | None = None
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.cache = cache

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-12)

    def embed(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        chunks: [{chunk_id, text, metadata}]
        Side-effect:
            - يضيف key: embedding لكل chunk
        Returns:
            np.ndarray (n_chunks, dim)
        """

        id_to_chunk = {c["chunk_id"]: c for c in chunks}

        texts_to_embed = []
        ids_to_embed = []

        # 1) Load from cache if exists
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if self.cache and self.cache.exists(cid):
                chunk["embedding"] = self.cache.load(cid)
            else:
                texts_to_embed.append(chunk["text"])
                ids_to_embed.append(cid)

        # 2) Encode missing
        if texts_to_embed:
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )

            new_embeddings = self._normalize(new_embeddings)

            for cid, emb in zip(ids_to_embed, new_embeddings):
                if self.cache:
                    self.cache.save(cid, emb)
                id_to_chunk[cid]["embedding"] = emb

        # 3) Safety check
        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError(f"Missing embedding for chunk_id={chunk['chunk_id']}")

        return np.vstack([chunk["embedding"] for chunk in chunks])

