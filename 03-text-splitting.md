# 3. Text Splitting

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### Why Text Splitting?

**Text splitting** breaks large documents into smaller chunks that fit within LLM context windows and improve retrieval accuracy. Chunks should be semantically meaningful and appropriately sized (typically 500-2000 tokens). Overlapping chunks preserve context across boundaries, ensuring no information is lost at split points.

```python
# Why split?
# 1. LLM context limits (e.g., 128K tokens)
# 2. Better retrieval granularity
# 3. Improved semantic search accuracy
# 4. Cost optimization (smaller chunks = fewer tokens)
```

### RecursiveCharacterTextSplitter

**RecursiveCharacterTextSplitter** intelligently splits text by recursively trying different separators (paragraphs, sentences, words, characters). It preserves semantic meaning by prioritizing larger text units. The most commonly used splitter for general-purpose RAG applications. Handles various document types effectively.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)
```

### CharacterTextSplitter

**CharacterTextSplitter** splits text by a fixed number of characters. Simpler than recursive splitting but may break sentences or words. Useful when you need precise control over chunk sizes. Less semantic-aware than recursive splitting, so use only when necessary.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator="\n\n"
)
splits = text_splitter.split_documents(docs)
```

### MarkdownHeaderTextSplitter

**MarkdownHeaderTextSplitter** splits markdown documents based on headers, preserving hierarchical structure. Chunks include header context, making them more meaningful for retrieval. Perfect for documentation, README files, and structured markdown content. Maintains parent-child relationships between sections.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
splits = markdown_splitter.split_text(markdown_doc)
```

### Token-based Splitting

**Token-based splitters** split text by token count rather than characters, ensuring chunks fit within model token limits. More accurate for LLM context management. Use when working with specific models that have strict token requirements. Provides better control over actual token usage.

```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    encoding_name="cl100k_base"  # OpenAI tokenizer
)
splits = text_splitter.split_documents(docs)
```

### Chunk Size and Overlap Best Practices

**Chunk size** depends on your use case: smaller chunks (200-500) for precise retrieval, larger chunks (1000-2000) for comprehensive context. **Overlap** (10-20% of chunk size) prevents information loss at boundaries. Test different configurations based on your document types and query patterns. Balance between granularity and context completeness.

```python
# Best Practices:
# - Chunk size: 500-1500 characters (or 200-800 tokens)
# - Overlap: 10-20% of chunk size
# - Test with your specific documents and queries
# - Consider document structure (paragraphs, sections)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Good default
    chunk_overlap=200,    # 20% overlap
    length_function=len
)
```

---

**Navigation:**
- [← Previous: Document Loading](02-document-loading.md)
- [Back to README](README.md)
- [Next: Embeddings →](04-embeddings.md)
