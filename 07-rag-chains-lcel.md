# 7. RAG Chains (LCEL)

> **Part of the LangChain RAG Cheatsheet**  
> **Updated for LangChain 0.3+ with LCEL (LangChain Expression Language)**

---

### LCEL Overview

**LangChain Expression Language (LCEL)** is a declarative way to compose chains using the pipe (`|`) operator. It enables streaming, parallel execution, and easy debugging. LCEL chains are Runnable objects that can be invoked, streamed, or batched. Modern LangChain applications use LCEL for building RAG pipelines.

```python
# LCEL syntax: chain = component1 | component2 | component3
# Components are Runnable objects
from langchain_core.runnables import RunnablePassthrough
```

### create_stuff_documents_chain

**create_stuff_documents_chain** combines retrieved documents into a single prompt and sends them to the LLM. "Stuff" refers to stuffing all documents into the context. Simple and effective when documents fit within the context window. Returns a chain that takes documents and a question, produces an answer.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions based on context: {context}"),
    ("human", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
```

### create_retrieval_chain

**create_retrieval_chain** combines a retriever with a document chain to create a complete RAG pipeline. It automatically retrieves documents, formats them into context, and generates answers. Returns a chain that takes an input query and returns both context and answer. The standard way to build RAG applications in modern LangChain.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Create document chain
prompt = ChatPromptTemplate.from_template(
    "Answer using context: {context}\n\nQuestion: {input}"
)
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
rag_chain = create_retrieval_chain(retriever, document_chain)

# Invoke
result = rag_chain.invoke({"input": "What is RAG?"})
print(result["answer"])
```

### create_history_aware_retriever

**create_history_aware_retriever** rewrites queries using chat history to make them standalone and context-aware. Essential for conversational RAG where questions reference previous messages. Uses an LLM to reformulate queries before retrieval. Improves retrieval quality in multi-turn conversations.

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history, rephrase the question to be standalone."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

### Complete RAG Pipeline

**Complete RAG pipeline** combines all components: document loading, splitting, embedding, vector storage, retrieval, and generation. Uses LCEL for clean composition. Handles the full flow from raw documents to answers. Production-ready pattern for building RAG applications.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# 1. Load documents
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 4. Create LLM
llm = ChatOpenAI(model="gpt-4o")

# 5. Create prompt
prompt = ChatPromptTemplate.from_template(
    "Answer using only this context: {context}\n\nQuestion: {input}"
)

# 6. Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 7. Query
result = rag_chain.invoke({"input": "What is the main topic?"})
print(result["answer"])
```

### Streaming RAG Responses

**Streaming RAG responses** return tokens as they're generated, improving perceived latency and user experience. Use `.stream()` instead of `.invoke()` on LCEL chains. Retrieval happens first, then generation streams. Essential for production applications requiring real-time feedback.

```python
# Stream responses
for chunk in rag_chain.stream({"input": "What is RAG?"}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
```

---

## Quick Reference

### Installation

```python
pip install langchain langchain-community langchain-openai langchain-chroma
pip install langchain-text-splitters langchain-core
```

### Basic RAG Pipeline

```python
# 1. Load → 2. Split → 3. Embed → 4. Store → 5. Retrieve → 6. Generate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load and split
loader = WebBaseLoader("https://example.com")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Create chain
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {input}")
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Query
result = rag_chain.invoke({"input": "What is this about?"})
```

---

**Navigation:**
- [← Previous: Retrievers](06-retrievers.md)
- [Back to README](README.md)
- [Next: Advanced RAG Techniques →](08-advanced-rag-techniques.md)
