# 8. Advanced RAG Techniques

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### Query Rewriting

**Query rewriting** improves retrieval by reformulating user queries before searching. Techniques include query expansion (synonyms), decomposition (multi-hop questions), and clarification. Uses LLMs to generate better search queries. Significantly improves retrieval quality for ambiguous or complex questions.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite this query for better retrieval: {query}"
)

# Use LCEL pattern (pipe operator)
rewrite_chain = rewrite_prompt | llm

original_query = "Tell me about that thing"
rewritten = rewrite_chain.invoke({"query": original_query})
docs = retriever.invoke(rewritten.content)
```

### Reranking

**Reranking** improves retrieval quality by using a more sophisticated model to score and reorder initial retrieval results. Cross-encoders (like Cohere Rerank) provide better relevance scoring than cosine similarity. Expensive but significantly improves answer quality. Use when initial retrieval returns many candidates.

```python
from langchain_community.cross_encoders import CohereRerank

reranker = CohereRerank(cohere_api_key="...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Retrieve more, rerank top results
docs = retriever.invoke("query")
reranked = reranker.compress_documents(docs, "query")[:5]
```

### Hybrid Search

**Hybrid search** combines vector similarity search with keyword-based search (BM25) for better results. Vector search captures semantic meaning, keyword search handles exact matches and rare terms. Ensemble retrievers combine both approaches. Often outperforms pure vector or keyword search alone.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(splits)
vector_retriever = vectorstore.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Tune based on your data
)
```

### Multi-Query Retrieval

**Multi-query retrieval** generates multiple query variations and retrieves documents for each, then combines results. Handles ambiguous queries and improves recall. Uses an LLM to generate alternative phrasings of the original query. More comprehensive than single-query retrieval.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# Generates multiple queries and combines results
docs = multi_query_retriever.invoke("What is machine learning?")
```

### Self-RAG

**Self-RAG** (Self-Retrieval Augmented Generation) uses the LLM to decide when retrieval is needed and which documents are relevant. The model can skip retrieval for questions it can answer directly or request more specific retrieval. More efficient and intelligent than always retrieving.

```python
# Self-RAG pattern (conceptual)
# 1. LLM decides: retrieve or not?
# 2. If retrieve: what to search for?
# 3. Evaluate retrieved docs: relevant?
# 4. Generate answer with or without context
# Requires custom agent implementation
```

### Agentic RAG

**Agentic RAG** uses agents to iteratively retrieve, reason, and generate answers. Agents can perform multiple retrieval steps, filter results, and synthesize information across documents. Handles complex, multi-step questions requiring information from multiple sources. More powerful but slower than standard RAG.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

agent = create_agent(
    model=llm,
    tools=[retrieve_docs],
    system_prompt="You are a RAG agent that retrieves and answers questions."
)
```

---

**Navigation:**
- [← Previous: RAG Chains (LCEL)](07-rag-chains-lcel.md)
- [Back to README](README.md)
- [Next: Prompt Engineering for RAG →](09-prompt-engineering-for-rag.md)
