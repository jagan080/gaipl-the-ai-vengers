# 📚 RAG (Retrieval-Augmented Generation) Documentation

## 📖 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Components](#components)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Knowledge Base Management](#knowledge-base-management)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The RAG (Retrieval-Augmented Generation) system combines semantic search with a large language model (LLM) to provide contextual, knowledge-base-aware responses. It retrieves relevant documents from a vector database and passes them to the LLM for generating informed answers.

**Key Benefits:**
- Provides context-aware responses based on your knowledge base
- Reduces hallucinations by grounding responses in real data
- Falls back to foundation model knowledge when RAG context is unavailable
- Supports multiple document formats (PDF, JSON, DOCX, Excel)

---

## 🏗️ Architecture

```
User Query
    ↓
[Embedding Model] (all-MiniLM-L6-v2)
    ↓
[Vector Search] (ChromaDB)
    ↓
[Retrieved Documents] (top-k results)
    ↓
[LLM Backend] (DeepSeek-R1:7B via Ollama)
    ↓
Final Response
```

### Components Interaction:
1. **Query** → Converted to embeddings
2. **Vector DB** → Searches for similar documents
3. **Context** → Combined with user prompt
4. **LLM** → Generates response using both context and foundation knowledge

---

## 🔄 How It Works

### Step-by-Step Flow:

1. **Query Reception**
   - User provides a question/prompt

2. **Embedding Generation**
   - Query is converted to embeddings using `all-MiniLM-L6-v2`
   - Embeddings are 384-dimensional vectors

3. **Semantic Search**
   - Query embedding is compared against document embeddings in ChromaDB
   - Top-k most similar documents are retrieved (default: k=3)

4. **Context Preparation**
   - Retrieved documents are formatted as context
   - If no context found, a notification is sent to the LLM
   - LLM is instructed to use foundation model knowledge

5. **LLM Processing**
   - System prompt guides the LLM on context usage
   - User prompt with context is sent to DeepSeek-R1:7B
   - LLM generates a response

6. **Response Generation**
   - Final answer is returned to the user

### Decision Logic:
```
IF RAG_context is not empty:
    Use context as primary source
ELSE:
    Tell LLM to use foundation model knowledge
END
```

---

## 🔧 Components

### 1. **ChromaDB (Vector Database)**
- **Location**: `code/src/chroma_db/`
- **Type**: Persistent vector store
- **Purpose**: Stores document embeddings for semantic search
- **Collection**: `rag_docs` (customizable)

```python
chroma_client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")
```

### 2. **Embedding Model**
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimension**: 384
- **Size**: ~33MB
- **Lazy-loaded** to avoid startup delays

```python
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(text_list).tolist()
```

### 3. **LLM Backend (Ollama)**
- **Model**: `deepseek-r1:7b`
- **API Endpoint**: `http://localhost:11434/api/chat`
- **Format**: Message-based chat protocol
- **Temperature**: 0.7 (controllable)
- **Max Tokens**: 500 (controllable)

### 4. **Text Processing**
- **PDF Extraction**: PyMuPDF (fitz)
- **JSON Handling**: Python json module
- **DOCX Support**: python-docx
- **Excel Support**: pandas
- **Text Chunking**: LangChain RecursiveCharacterTextSplitter

---

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: For HuggingFace backend
export HUGGINGFACE_API_KEY="your-key-here"

# Optional: For Google Vertex AI
export GCP_PROJECT_ID="your-project-id"
```

### LLM Provider Selection
Edit `rag.py` line 18:
```python
LLM_PROVIDER = 2  # 1=HuggingFace, 2=Ollama, 3=Gemini Vertex AI
```

### Tunable Parameters

| Parameter | Default | Location | Usage |
|-----------|---------|----------|-------|
| `top_k` | 3 | `retrieve_context()` | Number of documents to retrieve |
| `chunk_size` | 500 | `split_text()` | Text chunk size in characters |
| `chunk_overlap` | 50 | `split_text()` | Overlap between chunks |
| `temperature` | 0.7 | `generate_response()` | LLM creativity (0=deterministic, 1=creative) |
| `max_tokens` | 500 | `generate_response()` | Maximum response length |

---

## 📡 API Reference

### Core Functions

#### 1. `add_document_to_db(doc_text_list, doc_id)`
Adds documents to the knowledge base.

**Parameters:**
- `doc_text_list` (list): List of text chunks
- `doc_id` (str): Unique document identifier

**Example:**
```python
from rag import add_document_to_db
chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
add_document_to_db(chunks, "doc_001")
```

#### 2. `retrieve_context(query, top_k=3)`
Retrieves relevant context from the knowledge base.

**Parameters:**
- `query` (str): User query/question
- `top_k` (int, optional): Number of top results to return (default: 3)

**Returns:** (str) Concatenated retrieved documents

**Example:**
```python
from rag import retrieve_context
context = retrieve_context("How to restart Docker?", top_k=5)
print(context)
```

#### 3. `generate_response(prompt, context)`
Generates an LLM response with RAG enhancement.

**Parameters:**
- `prompt` (str): User question/prompt
- `context` (str): Context from RAG (pass empty string if none)

**Returns:** (str) LLM-generated response

**Example:**
```python
from rag import generate_response
response = generate_response(
    prompt="How to deploy a service?",
    context="Service deployment involves..."
)
print(response)
```

#### 4. `load_initial_knowledge(folder_path)`
Loads documents from a folder into the knowledge base.

**Parameters:**
- `folder_path` (str, optional): Folder path (default: `data/`)

**Example:**
```python
from rag import load_initial_knowledge
load_initial_knowledge()  # Loads from code/src/data/
```

#### 5. `extract_text_from_file(file_path)`
Extracts text from various document formats.

**Supported Formats:**
- `.pdf` - PDF files
- `.json` - JSON files
- `.docx` - Word documents
- `.xlsx` - Excel spreadsheets

**Example:**
```python
from rag import extract_text_from_file
text = extract_text_from_file("document.pdf")
```

---

## 📝 Usage Examples

### Example 1: Basic RAG Query
```python
from rag import retrieve_context, generate_response

# User's question
prompt = "How do I check service health?"

# Retrieve context from knowledge base
context = retrieve_context(prompt, top_k=3)

# Generate response with context
response = generate_response(prompt, context)

print(f"Response: {response}")
```

### Example 2: Load Knowledge Base and Query
```python
from rag import load_initial_knowledge, generate_response, retrieve_context

# Load all documents from data/ folder
load_initial_knowledge()

# Query the system
prompt = "What are best practices for monitoring?"
context = retrieve_context(prompt)
response = generate_response(prompt, context)

print(response)
```

### Example 3: Add Custom Documents
```python
from rag import add_document_to_db, split_text, generate_response, retrieve_context

# Custom documents
docs = [
    "Docker is a containerization platform...",
    "Kubernetes orchestrates containers...",
    "Helm manages Kubernetes packages..."
]

# Add to knowledge base
for doc in docs:
    chunks = split_text(doc, chunk_size=300)
    add_document_to_db(chunks, f"doc_{docs.index(doc)}")

# Query
context = retrieve_context("What is Kubernetes?")
response = generate_response("What is Kubernetes?", context)
print(response)
```

### Example 4: Handle Empty RAG Context
```python
from rag import generate_response

# When context is empty, LLM will use foundation knowledge
response = generate_response(
    prompt="Explain quantum computing",
    context=""  # Empty context
)
# LLM will use its own knowledge about quantum computing
```

---

## 📦 Knowledge Base Management

### Adding Documents

**Method 1: Bulk Load (Automatic)**
1. Place documents in `code/src/data/` folder
2. Call `load_initial_knowledge()`
3. All supported formats are processed

**Method 2: Programmatic**
```python
from rag import add_document_to_db, split_text

doc = "Your document text here..."
chunks = split_text(doc)
add_document_to_db(chunks, "unique_id")
```

**Method 3: File Upload**
```python
from rag import extract_text_from_file, split_text, add_document_to_db

file_path = "path/to/document.pdf"
text = extract_text_from_file(file_path)
chunks = split_text(text)
add_document_to_db(chunks, file_path)
```

### Viewing Knowledge Base Contents
```python
import chromadb

client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = client.get_or_create_collection(name="rag_docs")

# Get all documents
all_docs = collection.get()
print(f"Total chunks: {len(all_docs['ids'])}")
```

### Clearing Knowledge Base
```python
import chromadb

client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
client.delete_collection(name="rag_docs")
print("Knowledge base cleared")
```

---

## 🚨 Troubleshooting

### Issue: "Error: Cannot connect to Ollama"
**Cause**: Ollama is not running
**Solution**:
```bash
# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull deepseek-r1:7b
```

### Issue: "Error loading embedding model"
**Cause**: Network issues or model not cached
**Solution**:
```bash
# Ensure internet connection
# First run downloads ~33MB model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: ChromaDB path not found
**Cause**: Database directory doesn't exist
**Solution**:
```python
import os
os.makedirs("gaipl-the-ai-vengers/code/src/chroma_db", exist_ok=True)
```

### Issue: RAG returns empty context
**Cause**: No similar documents in knowledge base
**Solution**:
1. Load more documents: `load_initial_knowledge()`
2. LLM will use foundation knowledge automatically
3. Check that documents are being indexed properly

### Issue: Slow response times
**Cause**: Large chunk retrievals or slow embeddings
**Solution**:
- Reduce `top_k` in `retrieve_context()` (default 3)
- Reduce `chunk_size` in text splitting
- Upgrade Ollama model: use faster quantization

---

## 📊 Performance Metrics

| Operation | Typical Time |
|-----------|--------------|
| Embedding generation | ~100ms per document |
| Vector search | ~50ms for 1000 documents |
| LLM response | 5-30 seconds (depends on length) |
| Full pipeline (query → response) | 5-30 seconds |

---

## 🔐 Best Practices

1. **Chunk Size**: Use 300-500 characters for optimal retrieval
2. **Top-K Selection**: 3-5 results usually sufficient
3. **Temperature**: Use 0.5-0.7 for factual answers, 0.8+ for creative
4. **Document Format**: Ensure clean, structured documents
5. **Regular Updates**: Keep knowledge base current
6. **Error Handling**: Always handle empty RAG contexts gracefully

---

## 📚 Additional Resources

- **ChromaDB Docs**: https://docs.trychroma.com/
- **SentenceTransformers**: https://www.sbert.net/
- **Ollama**: https://ollama.ai/
- **LangChain**: https://langchain.com/

---

**Last Updated**: May 3, 2026  
**Version**: 1.0
