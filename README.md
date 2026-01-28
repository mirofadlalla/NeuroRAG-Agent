
# Hybrid RAG Agent System 🚀

A **production-style AI Agent system** that combines **Hybrid Retrieval-Augmented Generation (RAG)** with **Tool-Using, Memory-Driven Agents**.

This project is **not a demo** — it is designed with **real-world architecture patterns** used in modern AI companies.

---

## 🔥 Key Features

- **Hybrid RAG System**
  - Dense Retrieval (FAISS)
  - Sparse Retrieval (BM25)
  - Reciprocal Rank Fusion (RRF)
  - Cross-Encoder Re-ranking
  - Query Expansion

- **Agent Framework (From Scratch)**
  - Planning Agent (LLM-based)
  - Step Router (Tool / RAG / Direct reasoning)
  - Tool Execution Layer
  - Memory-Driven Decision Making

- **Advanced Agent Memory**
  - Short-Term Memory (LRU)
  - Episodic Memory (Persistent SQLite)
  - Semantic Memory (Embedding-based)
  - Tool Success Statistics
  - Temporal Decay for stale knowledge

- **Tools**
  - Python Code Execution Tool (Sandboxed)
  - Safe Calculator Tool (AST-based)
  - Hybrid RAG Tool

---

## 🧠 Architecture Overview

```
User Query
   ↓
Planning Agent (LLM)
   ↓
Step Decomposition
   ↓
Router
   ↓
┌───────────────┬───────────────┬───────────────┐
│   RAG Tool    │  Python Tool  │  Calc Tool    │
└───────────────┴───────────────┴───────────────┘
   ↓
Observations
   ↓
Agent Memory (Learn from Experience)
   ↓
Final Answer (LLM)
```

---

## 📂 Project Structure

```
.
├── agent/
│   ├── planner.py        # LLM-based planning
│   ├── router.py         # Step routing logic
│   ├── loop.py           # Agent execution loop
│   ├── memory.py         # Advanced memory system
│   └── llm.py            # LLM abstraction
│
├── tools/
│   ├── rag_tool.py
│   ├── python_tool.py
│   └── calc_tool.py
│
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── faiss_index.py
│   ├── bm25.py
│   ├── fusion.py
│   └── reranker.py
│
├── data/
│   └── *.txt
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧩 Why Hybrid RAG?

### ❌ Problem with Dense Retrieval Alone
- Fails on keyword-heavy queries
- Sensitive to embedding drift
- Misses exact term matches

### ❌ Problem with Sparse Retrieval Alone
- No semantic understanding
- Weak on paraphrasing

### ✅ Hybrid Solution
We combine:
- **Dense vectors** for semantic similarity
- **Sparse BM25** for lexical precision
- **RRF Fusion** to merge rankings robustly
- **Re-ranking** for final relevance

---

## 🔀 Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists:

```
score(d) = Σ 1 / (k + rank_i(d))
```

Advantages:
- Robust to noisy rankings
- Prevents dominance of one retriever
- Industry-proven (used by Google, Bing)

---

## 🧠 Memory-Driven Agents

The agent **learns from experience**:

- Remembers which tools worked best
- Routes future queries using semantic similarity
- Applies temporal decay to avoid stale decisions

> The agent improves its routing decisions over time using experience, semantic similarity, and decay.

---

## ▶️ Running the Project

```bash
pip install -r requirements.txt
python main.py
```

Type a query and interact with the agent.

---

## 📌 Example Queries

- "Explain attention in transformers and give code"
- "Calculate complexity of self-attention"
- "Search documents about RNN limitations"

---

## 🛠️ Current Limitations

- Single-agent execution
- No async execution
- Basic planner prompt
- No automated evaluation metrics

---

## 🚀 Planned Improvements

- ✅ Self-Reflection Agent (Critique & Improve)
- 🔄 Multi-Agent System (Planner / Researcher / Writer)
- ⚡ Async Tool Execution
- 🌐 FastAPI Deployment
- 📊 RAG Evaluation (Recall@K, MRR, Faithfulness)
- 🧠 Graph RAG Support

---

## 🎯 Use Cases

- AI Engineering Portfolio
- Research Prototyping
- Enterprise Knowledge Assistants
- Autonomous Coding Agents

---

## 👤 Author

**Omar Yasser**  
AI Engineer | LLM Systems | RAG | Agents

---

## ⭐ Final Note

This project intentionally avoids high-level frameworks (e.g., LangChain)
to demonstrate **deep understanding of LLM systems internals**.

If you're reviewing this repo:
> This is **engineering**, not a tutorial.
