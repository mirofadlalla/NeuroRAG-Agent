
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
DeepAgent/
│
├── main.py
│   └── For Deployemnt 
│
├── README.md
│   └── Project overview, setup instructions, architecture explanation, and usage
│
├── requirements.txt
│   └── Python dependencies for agent, RAG system, and LLM integration
│
├── Architecture-Digram.png
│   └── High-level system architecture of the agent + Hybrid RAG pipeline
│
├── test_agent.py
│   └── Entry point for running the autonomous AI agent and handling user interaction
│
├── agent/
│   ├── __init__.py
│   │
│   ├── llm.py
│   │   └── LLM abstraction layer (prompt handling, model calls, response parsing)
│   │
│   ├── planner.py
│   │   └── Task planning and reasoning logic for multi-step agent execution
│   │
│   ├── router.py
│   │   └── Intent-based routing of user queries to appropriate agent actions
│   │
│   ├── memory.py
│   │   └── Agent memory abstraction (short-term / extensible for long-term memory)
│   │
│   ├── loop.py
│   │   └── Core agent execution loop (Reason → Act → Observe → Decide)
│
├── Hyprid_RagSystem/
│   ├── __init__.py
│   │
│   ├── pipeline.py
│   │   └── Smart Hybrid RAG pipeline combining dense + sparse retrieval,
│   │       query expansion, fusion, and re-ranking
│   │
│   ├── embedder.py
│   │   └── Embedding engine with caching, normalization, and batch processing
│   │
│   ├── faiss_index.py
│   │   └── Dense vector index using FAISS with persistence and ID mapping
│   │
│   ├── bm25_index.py
│   │   └── Sparse lexical retrieval using BM25 for keyword-based search
│   │
│   ├── fusion.py
│   │   └── Reciprocal Rank Fusion (RRF) for combining dense and sparse results
│   │
│   ├── rerank.py
│   │   └── Cross-encoder semantic re-ranking for final context selection
│   │
│   ├── utils.py
│   │   └── Shared utility functions for chunking, preprocessing, and helpers
│   │
│   └── config.py
│       └── Centralized configuration for models, retrieval parameters, and thresholds

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
