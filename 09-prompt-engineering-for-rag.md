# 9. Prompt Engineering for RAG

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### RAG Prompt Templates

**RAG prompt templates** structure how retrieved context is presented to the LLM. Effective templates include clear instructions, context formatting, and answer requirements. Use placeholders for `{context}` and `{input}`. Well-designed prompts significantly improve answer quality and reduce hallucinations.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions using ONLY the provided context.
    
Context: {context}

If the context doesn't contain the answer, say "I don't have enough information."""),
    ("human", "{input}")
])
```

### System Prompts for RAG

**System prompts for RAG** define the assistant's role, behavior, and constraints. Specify how to use context, when to say "I don't know," and citation requirements. Clear system prompts reduce hallucinations and improve consistency. Include instructions about source attribution and answer format.

```python
system_prompt = """You are an expert assistant that answers questions using provided documents.

Rules:
1. Use ONLY information from the context
2. Cite sources when possible
3. If context is insufficient, say so
4. Be concise and accurate
"""
```

### Few-Shot Examples in RAG

**Few-shot examples** in RAG prompts demonstrate desired behavior with example question-answer pairs. Helps the model understand expected format, citation style, and answer quality. Include examples showing how to handle insufficient context and proper source attribution. Improves consistency and reduces need for prompt tuning.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer questions using context. Examples:

Q: What is Python?
Context: Python is a programming language...
A: Python is a programming language. [Source: doc1]

Q: What is Java?
Context: [No relevant information]
A: I don't have enough information to answer this question.

Now answer:"""),
    ("human", "Context: {context}\n\nQuestion: {input}")
])
```

### Citation and Source Attribution

**Citation and source attribution** in RAG systems provide transparency and allow users to verify answers. Include document sources, page numbers, or metadata in responses. Use structured output or prompt instructions to ensure consistent citation format. Critical for production RAG applications requiring trust and verifiability.

```python
prompt = ChatPromptTemplate.from_template(
    """Answer the question using the context. Include citations in format [Source: filename, page X].

Context: {context}

Question: {input}

Answer with citations:"""
)

# Or use structured output
from dataclasses import dataclass

@dataclass
class AnswerWithCitations:
    answer: str
    sources: list[str]
```

---

**Navigation:**
- [← Previous: Advanced RAG Techniques](08-advanced-rag-techniques.md)
- [Back to README](README.md)
- [Next: Evaluation and Optimization →](10-evaluation-and-optimization.md)
