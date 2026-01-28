from ..Hyprid_RagSystem.pipeline import SmartHybridRetriever


class RAGTool:
    name = "rag"

    def __init__(self, retriever: SmartHybridRetriever):
        self.retriever = retriever

    def run(self, query: str, top_k: int = 5):
        if not query.strip():
            return {"status": "error", "output": "Empty query"}

        return {
            "status": "success",
            "query": query,
            "results": self.retriever.retrieve(query, top_k)
        }
