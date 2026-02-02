# 4. Embeddings

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### What are Embeddings?

**Embeddings** are dense vector representations that capture semantic meaning of text. Similar texts have similar vectors, enabling semantic search. Embeddings convert text into fixed-size numerical arrays (typically 384-1536 dimensions). They're the foundation of vector similarity search in RAG systems, allowing retrieval based on meaning rather than keywords.

```python
# Embeddings convert text to vectors
# "cat" → [0.2, -0.1, 0.5, ...]
# "dog" → [0.3, -0.1, 0.4, ...]  # Similar vectors = similar meaning
```

### OpenAI Embeddings

**OpenAI Embeddings** (text-embedding-3-small, text-embedding-3-large) provide high-quality semantic representations. Small model is cost-effective (1536 dimensions), large model offers better accuracy (3072 dimensions). Both support dimension reduction. Widely used in production RAG systems for their quality and reliability.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536  # Optional: reduce dimensions
)

# Generate embeddings
vector = embeddings.embed_query("What is RAG?")
doc_vectors = embeddings.embed_documents(["doc1", "doc2"])
```

### HuggingFace Embeddings

**HuggingFace Embeddings** offer open-source alternatives with various model sizes and languages. Models like `sentence-transformers/all-MiniLM-L6-v2` provide good quality with smaller dimensions (384). Free to use, customizable, and can run locally. Ideal for cost-sensitive or privacy-focused applications.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Google Generative AI Embeddings

**Google Generative AI Embeddings** (models/gemini-embedding-001) provide competitive embedding quality with 3072 dimensions. Integrated with Google's ecosystem and suitable for multi-modal applications. Good alternative to OpenAI embeddings with similar performance characteristics. Supports various languages and domains. Uses `GOOGLE_API_KEY` environment variable for authentication.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Generate embeddings
vector = embeddings.embed_query("What is RAG?")
doc_vectors = embeddings.embed_documents(["doc1", "doc2"])
```

### Embedding Dimensions

**Embedding dimensions** determine vector size and impact storage, search speed, and quality. Higher dimensions (1536+) capture more nuance but require more storage. Lower dimensions (384-768) are faster and cheaper but may lose semantic detail. Choose based on your accuracy requirements and infrastructure constraints.

```python
# Dimension Trade-offs:
# - 384 dims: Fast, cheap, good for simple tasks
# - 768 dims: Balanced performance
# - 1536 dims: High quality, more storage
# - 3072 dims: Maximum quality, expensive

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536  # Adjust based on needs
)
```

---

**Navigation:**
- [← Previous: Text Splitting](03-text-splitting.md)
- [Back to README](README.md)
- [Next: Vector Stores →](05-vector-stores.md)
