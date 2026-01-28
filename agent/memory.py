import time
import sqlite3
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional
import numpy as np


class AgentMemory:
    def __init__(
        self,
        max_short_term: int = 20,
        decay_lambda: float = 0.001,
        db_path: str = "agent_memory.db",
        embed_fn=None
    ):
        """
        max_short_term: LRU size for short-term memory
        decay_lambda: decay factor for old episodes
        embed_fn: function(text:str) -> np.ndarray
        """
        self.short_term = deque(maxlen=max_short_term)
        self.episodes = []
        self.tool_stats = defaultdict(lambda: {"success": 0, "fail": 0})
        self.semantic_memory = []  # list of (embedding, tool)
        self.decay_lambda = decay_lambda
        self.embed_fn = embed_fn

        self.conn = sqlite3.connect(db_path)
        self._init_db()

    # -------------------- DB --------------------

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                query TEXT,
                tool TEXT,
                success INTEGER,
                timestamp REAL
            )
        """)
        self.conn.commit()

    # -------------------- Add Interaction --------------------

    def add_interaction(self, query: str, tool: str, result: Dict[str, Any]):
        success = result.get("status") == "success"
        ts = time.time()

        # short-term (LRU)
        self.short_term.append({
            "query": query,
            "tool": tool,
            "result": result,
            "timestamp": ts
        })

        # episodic memory
        episode = {
            "query": query,
            "tool": tool,
            "success": success,
            "timestamp": ts
        }
        self.episodes.append(episode)

        # persist
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?)",
            (query, tool, int(success), ts)
        )
        self.conn.commit()

        # tool statistics
        if success:
            self.tool_stats[tool]["success"] += 1
        else:
            self.tool_stats[tool]["fail"] += 1

        # semantic memory
        if self.embed_fn is not None:
            emb = self.embed_fn(query)
            self.semantic_memory.append((emb, tool))

    # -------------------- Read Memory --------------------

    def recent_context(self, k: int = 5) -> List[Dict[str, Any]]:
        return list(self.short_term)[-k:]

    # -------------------- Tool Recommendation --------------------

    def preferred_tool(self, query: str) -> Optional[str]:
        # 1) semantic match
        if self.embed_fn and self.semantic_memory:
            q_emb = self.embed_fn(query)
            best_sim = 0.0
            best_tool = None

            for emb, tool in self.semantic_memory:
                sim = self._cosine_similarity(q_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_tool = tool

            if best_sim > 0.8:
                return best_tool

        # 2) fallback: best success rate
        best_tool = None
        best_rate = 0.0

        for tool, stats in self.tool_stats.items():
            total = stats["success"] + stats["fail"]
            if total == 0:
                continue
            rate = stats["success"] / total
            if rate > best_rate:
                best_rate = rate
                best_tool = tool

        return best_tool

    # -------------------- Decay --------------------

    def decay_weight(self, timestamp: float) -> float:
        age = time.time() - timestamp
        return float(np.exp(-self.decay_lambda * age))

    # -------------------- Utils --------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
