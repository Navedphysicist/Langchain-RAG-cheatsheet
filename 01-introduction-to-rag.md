# 1. Introduction to RAG

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines the power of information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them as context for the LLM to generate accurate, grounded answers. This approach reduces hallucinations and allows LLMs to access up-to-date, domain-specific information without fine-tuning.

```python
# RAG Pipeline Overview
# 1. Load documents → 2. Split into chunks → 3. Create embeddings
# 4. Store in vector database → 5. Retrieve relevant chunks → 6. Generate answer
```

### Why use RAG?

RAG solves critical LLM limitations: **knowledge cutoff dates**, **hallucinations**, and **domain-specific knowledge gaps**. Instead of fine-tuning (expensive and slow), RAG allows you to inject external knowledge at inference time. It's cost-effective, maintainable, and enables real-time updates to your knowledge base without retraining models.

```python
# Benefits of RAG:
# ✅ Access to private/domain-specific data
# ✅ Reduced hallucinations
# ✅ No fine-tuning required
# ✅ Easy to update knowledge base
# ✅ Source attribution and citations
```

### RAG Architecture Overview

A RAG system consists of **three main phases**: **Indexing** (load, split, embed, store), **Retrieval** (query embedding, similarity search), and **Generation** (context injection, LLM response). The indexing phase happens offline, while retrieval and generation occur at query time. This architecture enables scalable, efficient question-answering over large document collections.

```python
# RAG Architecture Flow
# Indexing Phase (Offline):
# Documents → Loader → Splitter → Embeddings → Vector Store

# Query Phase (Online):
# Query → Embedding → Similarity Search → Retrieved Docs → LLM → Answer
```

---

**Navigation:**
- [← Back to README](README.md)
- [Next: Document Loading →](02-document-loading.md)
