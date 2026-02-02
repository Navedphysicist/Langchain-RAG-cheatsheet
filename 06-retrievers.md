# 6. Retrievers

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### Retriever Basics

**Retrievers** provide a standardized interface for fetching relevant documents from vector stores or other sources. They abstract retrieval logic and can be swapped without changing downstream code. Retrievers return lists of `Document` objects and support various search strategies (similarity, MMR, etc.). Essential components in RAG pipelines.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Retrieve documents
docs = retriever.invoke("What is machine learning?")
```

### Similarity Search Retrieval

**Similarity search retrieval** finds documents with embeddings most similar to the query embedding. Uses cosine similarity or dot product to rank results. Simple and effective for most RAG applications. Returns top-k most similar documents. Fast but may return redundant results.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

docs = retriever.invoke("query")
```

### MMR (Maximal Marginal Relevance) Retrieval

**MMR retrieval** balances relevance and diversity. It selects documents that are both relevant to the query and different from each other. Reduces redundancy in retrieved results. Use when you want diverse perspectives or when similarity search returns too many similar documents. Controlled by `fetch_k` and `lambda_mult` parameters.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Fetch more candidates
        "lambda_mult": 0.5  # 0=diversity, 1=relevance
    }
)
```

### Contextual Compression Retrieval

**Contextual compression retrieval** filters and compresses retrieved documents using an LLM. Removes irrelevant information and keeps only query-relevant content. Reduces token usage and improves answer quality. Uses `ContextualCompressionRetriever` with a base retriever and document compressor. More expensive but produces higher-quality context.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Parent Document Retriever

**Parent Document Retriever** retrieves small chunks but returns larger parent documents for context. Useful when you need fine-grained retrieval but want full document context for generation. Stores both small chunks (for retrieval) and parent documents (for context). Improves retrieval precision while maintaining context.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

# Store parent documents
store = InMemoryStore()
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=text_splitter,
    parent_splitter=parent_splitter
)
```

### Ensemble Retrievers

**Ensemble retrievers** combine multiple retrieval strategies for better results. They merge results from different retrievers (e.g., vector similarity + keyword search) and re-rank them. Often outperforms single retrieval methods. Use `EnsembleRetriever` to combine vector store retrieval with BM25 or other keyword-based retrievers.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine vector and keyword retrieval
bm25_retriever = BM25Retriever.from_documents(splits)
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
```

---

**Navigation:**
- [← Previous: Vector Stores](05-vector-stores.md)
- [Back to README](README.md)
- [Next: RAG Chains (LCEL) →](07-rag-chains-lcel.md)
