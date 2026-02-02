# 5. Vector Stores

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### Vector Store Overview

**Vector stores** are specialized databases that store document embeddings and enable fast similarity search. They index high-dimensional vectors using algorithms like approximate nearest neighbor (ANN) search. Popular options include Chroma (local), FAISS (fast), Pinecone (managed), Weaviate, and Qdrant. Choose based on scale, latency, and deployment requirements.

```python
# Vector stores enable:
# - Fast similarity search
# - Scalable storage
# - Metadata filtering
# - Hybrid search (vector + keyword)
```

### Chroma

**Chroma** is an open-source, lightweight vector database perfect for development and small-scale production. Easy to set up, runs locally or in-memory, and provides a simple Python API. Great for prototyping and learning RAG. Limited scalability compared to managed solutions but sufficient for many use cases.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### FAISS

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. Extremely fast, supports GPU acceleration, and handles millions of vectors on a single machine. Not a full database (no persistence by default), but excellent for high-performance retrieval. Best for read-heavy workloads.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

# Save and load
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings)
```

### Pinecone

**Pinecone** is a fully managed vector database service with automatic scaling, high availability, and global distribution. Production-ready with low latency and high throughput. Handles billions of vectors with ease. Requires API key and paid subscription but offers enterprise-grade reliability and features.

```python
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import pinecone

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name="my-index"
)
```

### Weaviate

**Weaviate** is an open-source vector database with built-in ML models, GraphQL API, and hybrid search capabilities. Supports both self-hosted and cloud deployments. Excellent for complex queries combining vector similarity with metadata filtering. Good choice for production systems requiring advanced features.

```python
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
import weaviate

client = weaviate.Client("http://localhost:8080")
vectorstore = Weaviate.from_documents(
    documents=splits,
    embedding=embeddings,
    client=client,
    by_text=False
)
```

### Qdrant

**Qdrant** is a vector similarity search engine with filtering, payload storage, and hybrid search. Open-source with cloud options, supports both dense and sparse vectors. Excellent performance and feature set. Good balance between open-source flexibility and production capabilities.

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
vectorstore = Qdrant.from_documents(
    documents=splits,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="my_collection"
)
```

### Creating and Querying Vector Stores

**Creating vector stores** involves loading documents, generating embeddings, and storing them. **Querying** converts queries to embeddings and performs similarity search. Use `as_retriever()` to convert vector stores into retrievers for RAG chains. Configure search parameters (k, score threshold) based on your needs.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# Query directly
docs = vectorstore.similarity_search("query", k=3)

# Use as retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

---

**Navigation:**
- [← Previous: Embeddings](04-embeddings.md)
- [Back to README](README.md)
- [Next: Retrievers →](06-retrievers.md)
