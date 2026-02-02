# LangChain RAG Cheatsheet

> **A comprehensive guide to building Retrieval-Augmented Generation (RAG) applications with LangChain**

[![LangChain Version](https://img.shields.io/badge/LangChain-0.3+-blue.svg)](https://python.langchain.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

## 📋 Overview

This cheatsheet provides a complete reference for building production-ready RAG (Retrieval-Augmented Generation) applications using LangChain. It covers everything from basic document loading to advanced techniques like query rewriting, reranking, and agentic RAG.

**Perfect for:**
- Developers building RAG applications
- Data scientists working with LLMs
- Engineers implementing semantic search systems
- Anyone learning LangChain and RAG patterns

## ✨ Key Highlights

- **Modern LangChain Patterns**: Built for LangChain 0.3+ with LCEL (LangChain Expression Language)
- **Production-Ready Examples**: All code examples follow current best practices
- **Comprehensive Coverage**: Complete RAG pipeline from start to finish
- **55+ Code Examples**: Practical, tested code snippets throughout
- **Best Practices**: Proven strategies for optimization and evaluation

## 📚 Content Statistics

- **10 Comprehensive Sections**: Covering the entire RAG pipeline
- **55+ Practical Code Examples**: Real-world implementations for every concept
- **Complete Coverage**: From document loading to advanced optimization techniques
- **LangChain 0.3+ Compatible**: All examples use the latest LangChain patterns
- **LCEL Throughout**: Modern LangChain Expression Language patterns in every section

## 🗺️ Reading Sequence

**IMPORTANT**: Follow the files in numerical order (01-10) for a complete understanding of RAG systems. Each section builds upon previous concepts.

### Recommended Learning Path

1. **Start Here**: [01-introduction-to-rag.md](01-introduction-to-rag.md) - Understand what RAG is and why it matters
2. **Foundation**: Files 02-04 cover the core components (Document Loading, Text Splitting, Embeddings)
3. **Storage & Retrieval**: Files 05-06 explain Vector Stores and Retrievers
4. **Building RAG**: File 07 shows how to assemble everything with LCEL chains
5. **Advanced Topics**: Files 08-10 cover advanced techniques, prompt engineering, and optimization

### Learning Paths by Experience Level

**Beginners**: Follow files 01-07 sequentially. This covers the fundamentals needed to build a working RAG system.

**Intermediate**: Complete files 01-09. Adds advanced retrieval techniques and prompt engineering for better results.

**Advanced**: All files 01-10. Focus especially on files 08-10 for production optimization, evaluation metrics, and cutting-edge techniques.

## 📁 File Structure

The cheatsheet is organized into 10 numbered files, each covering a specific aspect of RAG:

### Core Fundamentals

**[01-introduction-to-rag.md](01-introduction-to-rag.md)**
- What is RAG and why use it
- RAG architecture overview
- Benefits and use cases
- Foundation concepts for understanding the rest

**[02-document-loading.md](02-document-loading.md)**
- Overview of document loaders
- WebBaseLoader for web content
- PyMuPDFLoader for PDFs
- DirectoryLoader for batch processing
- CSVLoader for structured data
- Creating custom loaders

**[03-text-splitting.md](03-text-splitting.md)**
- Why text splitting is essential
- RecursiveCharacterTextSplitter (most common)
- CharacterTextSplitter
- MarkdownHeaderTextSplitter
- Token-based splitting
- Best practices for chunk size and overlap

**[04-embeddings.md](04-embeddings.md)**
- What embeddings are and how they work
- OpenAI Embeddings (text-embedding-3-small/large)
- HuggingFace Embeddings (open-source)
- Google Generative AI Embeddings
- Embedding dimension trade-offs and selection

### Storage and Retrieval

**[05-vector-stores.md](05-vector-stores.md)**
- Vector store overview and concepts
- Chroma (local development)
- FAISS (high-performance)
- Pinecone (managed service)
- Weaviate (advanced features)
- Qdrant (open-source with cloud options)
- Creating and querying vector stores

**[06-retrievers.md](06-retrievers.md)**
- Retriever basics and interface
- Similarity search retrieval
- MMR (Maximal Marginal Relevance) retrieval
- Contextual compression retrieval
- Parent document retriever
- Ensemble retrievers (combining strategies)

### Building RAG Systems

**[07-rag-chains-lcel.md](07-rag-chains-lcel.md)**
- LCEL (LangChain Expression Language) overview
- create_stuff_documents_chain
- create_retrieval_chain (complete RAG pipeline)
- create_history_aware_retriever (conversational RAG)
- Complete end-to-end RAG pipeline example
- Streaming RAG responses

### Advanced Topics

**[08-advanced-rag-techniques.md](08-advanced-rag-techniques.md)**
- Query rewriting for better retrieval
- Reranking with cross-encoders
- Hybrid search (vector + keyword)
- Multi-query retrieval
- Self-RAG patterns
- Agentic RAG with LangChain agents

**[09-prompt-engineering-for-rag.md](09-prompt-engineering-for-rag.md)**
- RAG prompt templates and structure
- System prompts for RAG applications
- Few-shot examples in prompts
- Citation and source attribution
- Best practices for prompt design

**[10-evaluation-and-optimization.md](10-evaluation-and-optimization.md)**
- RAG evaluation metrics (RAGAS framework)
- Retrieval evaluation (precision@k, recall@k, MRR)
- Generation evaluation (faithfulness, relevance)
- Performance optimization strategies
- Caching, batching, and streaming techniques

## 🎯 Key Concepts Covered

### Core RAG Pipeline
- Document loading from various sources (web, PDF, CSV, custom)
- Intelligent text splitting with overlap strategies
- Embedding generation and storage
- Vector similarity search
- Context injection and generation

### Advanced Techniques
- **Query Rewriting**: Improve retrieval with better queries
- **Reranking**: Use cross-encoders for better relevance
- **Hybrid Search**: Combine vector and keyword search
- **Multi-Query Retrieval**: Generate multiple query variations
- **Self-RAG**: LLM decides when to retrieve
- **Agentic RAG**: Iterative retrieval and reasoning

### Production Considerations
- Evaluation metrics (RAGAS, precision@k, recall@k)
- Performance optimization (caching, batching, streaming)
- Prompt engineering for better answers
- Citation and source attribution
- Cost and latency optimization

## 🔧 Prerequisites

Before diving into this cheatsheet, you should have:

- **Python Knowledge**: Basic to intermediate Python programming skills
- **Basic ML Concepts**: Understanding of machine learning fundamentals
- **LangChain Basics**: Familiarity with LangChain concepts (helpful but not required)
- **API Access**: Access to embedding models (OpenAI, HuggingFace, or Google) and LLMs

## 📖 How to Use This Cheatsheet

1. **For Beginners**: Start with file 01 and work through sequentially to file 07. This gives you everything needed to build a working RAG system.

2. **For Quick Reference**: Jump to specific files based on what you need. Each file is self-contained with complete examples.

3. **For Production**: Focus on files 08-10 after mastering the fundamentals. These cover advanced techniques, prompt engineering, and optimization.

4. **For Troubleshooting**: Check the "Best Practices" subsections in relevant files for common issues and solutions.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain RAG Tutorials](https://python.langchain.com/docs/tutorials/retrievers)
- [LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [Vector Store Integrations](https://python.langchain.com/docs/integrations/vectorstores/)

## 📅 Version Information

- **Last Updated**: February 2025
- **LangChain Version**: 0.3+
- **Status**: All code examples verified against latest LangChain documentation
- **Python Version**: 3.8+

## 📝 File Structure

```
.
├── README.md                          # This guide (no code examples)
├── langchain_rag_cheatsheet.md        # Original master file (optional reference)
├── 01-introduction-to-rag.md          # Section 1: Introduction
├── 02-document-loading.md             # Section 2: Document Loading
├── 03-text-splitting.md               # Section 3: Text Splitting
├── 04-embeddings.md                   # Section 4: Embeddings
├── 05-vector-stores.md                # Section 5: Vector Stores
├── 06-retrievers.md                   # Section 6: Retrievers
├── 07-rag-chains-lcel.md              # Section 7: RAG Chains (LCEL)
├── 08-advanced-rag-techniques.md      # Section 8: Advanced Techniques
├── 09-prompt-engineering-for-rag.md   # Section 9: Prompt Engineering
└── 10-evaluation-and-optimization.md   # Section 10: Evaluation & Optimization
```

## 🤝 Contributing

Found an error or want to suggest improvements? Feel free to:
- Open an issue
- Submit a pull request
- Share feedback

## 📄 License

This cheatsheet is provided as-is for educational and reference purposes.

---

**Happy Building! 🚀**

*Build powerful RAG applications with confidence using this comprehensive LangChain reference.*
