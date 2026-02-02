# 10. Evaluation and Optimization

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### RAG Evaluation Metrics

**RAG evaluation metrics** measure system quality across retrieval and generation. **Retrieval metrics**: precision@k, recall@k, MRR (Mean Reciprocal Rank). **Generation metrics**: faithfulness (grounded in context), answer relevance, context precision. Use frameworks like RAGAS or custom evaluation scripts. Essential for production systems.

```python
# Key Metrics:
# - Retrieval: Precision@k, Recall@k, MRR
# - Generation: Faithfulness, Answer Relevance
# - End-to-end: Answer Correctness

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevance]
)
```

### Retrieval Evaluation

**Retrieval evaluation** measures how well the retriever finds relevant documents. Use ground truth query-document pairs. Calculate precision@k (fraction of retrieved docs that are relevant) and recall@k (fraction of relevant docs retrieved). Optimize chunk size, embedding model, and retrieval parameters based on metrics.

```python
# Evaluate retrieval
def evaluate_retrieval(retriever, test_queries, ground_truth):
    scores = []
    for query, relevant_docs in zip(test_queries, ground_truth):
        retrieved = retriever.invoke(query)
        precision = len(set(retrieved) & set(relevant_docs)) / len(retrieved)
        scores.append(precision)
    return sum(scores) / len(scores)
```

### Generation Evaluation

**Generation evaluation** measures answer quality: correctness, faithfulness to context, and relevance to question. Use LLM-as-judge or human evaluation. Check for hallucinations and ensure answers are grounded in retrieved context. Optimize prompts and model parameters based on evaluation results.

```python
# Evaluate generation
def evaluate_generation(answer, context, question):
    # Use LLM to judge:
    # 1. Is answer correct?
    # 2. Is answer grounded in context?
    # 3. Is answer relevant to question?
    pass
```

### Performance Optimization

**Performance optimization** for RAG systems involves reducing latency, cost, and improving accuracy. Techniques: caching embeddings and retrievals, using smaller/faster models, optimizing chunk sizes, parallel processing, and efficient vector search. Monitor metrics and iterate. Balance between speed, cost, and quality based on requirements.

```python
# Optimization strategies:
# 1. Cache embeddings and retrievals
# 2. Use smaller embedding models for non-critical paths
# 3. Optimize chunk size and overlap
# 4. Use approximate nearest neighbor search
# 5. Batch processing where possible
# 6. Use streaming for better UX

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # Cache LLM calls
```

---

**Navigation:**
- [← Previous: Prompt Engineering for RAG](09-prompt-engineering-for-rag.md)
- [Back to README](README.md)
