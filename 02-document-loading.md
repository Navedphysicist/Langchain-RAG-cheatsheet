# 2. Document Loading

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### Overview of Document Loaders

**Document loaders** convert data from various sources into LangChain's standardized `Document` format. Each loader handles source-specific parsing (PDFs, web pages, databases, etc.) and extracts text content along with metadata. LangChain provides 100+ loaders for different data sources, ensuring consistent processing regardless of origin.

```python
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_core.documents import Document

# All loaders return Document objects with:
# - page_content: The text content
# - metadata: Source information (source, page number, etc.)
```

### WebBaseLoader

**WebBaseLoader** scrapes and loads content from websites. It uses BeautifulSoup4 to parse HTML and extract text content. You can customize parsing with `bs_kwargs` to target specific HTML elements, filter content, or handle JavaScript-rendered pages. Ideal for loading blog posts, documentation, and web articles.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title")
        )
    ),
)
docs = loader.load()
```

### PyMuPDFLoader

**PyMuPDFLoader** extracts text from PDF files using the PyMuPDF (fitz) library. It's fast, supports structured text extraction, and preserves page numbers and metadata. Perfect for loading research papers, reports, and PDF documents. Handles both text-based and scanned PDFs (with OCR support).

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("document.pdf")
docs = loader.load()

# Access page content and metadata
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)  # {'source': 'document.pdf', 'page': 0}
```

### DirectoryLoader

**DirectoryLoader** loads multiple files from a directory, supporting various file types through glob patterns. It automatically detects file types and uses appropriate loaders. Useful for batch processing documents from folders. Supports recursive directory traversal and file filtering.

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all PDFs from a directory
loader = DirectoryLoader(
    "./documents/",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
docs = loader.load()
```

### CSVLoader

**CSVLoader** reads CSV files and converts rows into documents. You can specify which columns to use as content and metadata. Useful for structured data like product catalogs, user data, or tabular information. Supports custom separators and encoding options.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    source_column="product_name",  # Column to use as source
    encoding="utf-8"
)
docs = loader.load()
```

### Custom Loaders

**Custom loaders** extend LangChain's capabilities for proprietary data sources. Create a loader by inheriting from `BaseLoader` and implementing the `load()` method. Return a list of `Document` objects with `page_content` and `metadata`. This enables integration with internal databases, APIs, or custom file formats.

```python
from langchain_core.documents import BaseLoader, Document

class CustomLoader(BaseLoader):
    def __init__(self, data_source):
        self.data_source = data_source
    
    def load(self):
        # Your custom loading logic
        return [
            Document(
                page_content=content,
                metadata={"source": self.data_source}
            )
        ]
```

---

**Navigation:**
- [← Previous: Introduction to RAG](01-introduction-to-rag.md)
- [Back to README](README.md)
- [Next: Text Splitting →](03-text-splitting.md)
