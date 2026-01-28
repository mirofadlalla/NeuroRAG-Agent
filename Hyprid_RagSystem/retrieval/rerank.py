from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class ReRanker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ):
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        for c, s in zip(chunks, scores):
            c["score"] = float(s)

        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks[:top_k]



# Code splitting the top chunks into sentences and getting score for each one
'''
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import nltk

nltk.download("punkt")

class ReRanker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform sentence-level reranking:
        1. Split each chunk into sentences
        2. Score each sentence with CrossEncoder
        3. Return top_k sentences with scores
        """
        sentence_entries = []
        for chunk in chunks:
            text = chunk["text"]
            sentences = nltk.sent_tokenize(text)  # split chunk to sentences
            for sent in sentences:
                sentence_entries.append({
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk.get("metadata", {}),
                    "text": sent
                })

        if not sentence_entries:
            return []

        # prepare pairs (query, sentence)
        pairs = [(query, s["text"]) for s in sentence_entries]
        scores = self.model.predict(pairs)

        # attach scores
        for s, score in zip(sentence_entries, scores):
            s["score"] = float(score)

        # sort by score descending
        sentence_entries.sort(key=lambda x: x["score"], reverse=True)

        return sentence_entries[:top_k]
'''