# 📡 API Reference & Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [RAG API](#rag-api)
3. [Agent API](#agent-api)
4. [LLM Backend API](#llm-backend-api)
5. [Error Handling](#error-handling)
6. [Code Examples](#code-examples)

---

## 📖 Overview

The AI Vengers system provides two main API interfaces:
1. **RAG API** - Knowledge base retrieval and response generation
2. **Agent API** - Task execution and automation

Both APIs are Python-based and can be used programmatically or integrated into larger applications.

---

## 🔍 RAG API

### Module: `rag.py`

#### 1. `add_document_to_db(doc_text_list, doc_id)`

Adds text documents to the knowledge base.

**Signature:**
```python
def add_document_to_db(doc_text_list: list[str], doc_id: str) -> None
```

**Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `doc_text_list` | list[str] | List of text chunks to store | `["Part 1", "Part 2"]` |
| `doc_id` | str | Unique identifier for the document | `"doc_001"` |

**Returns:** None

**Raises:**
- `Exception`: If ChromaDB connection fails

**Example:**
```python
from rag import add_document_to_db

chunks = [
    "Docker is a containerization platform",
    "It allows packaging applications",
    "Runs consistently across environments"
]
add_document_to_db(chunks, "docker_basics")
```

---

#### 2. `retrieve_context(query, top_k=3)`

Retrieves relevant documents from the knowledge base using semantic search.

**Signature:**
```python
def retrieve_context(query: str, top_k: int = 3) -> str
```

**Parameters:**
| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `query` | str | - | Search query | `"How to restart Docker?"` |
| `top_k` | int | 3 | Number of results to retrieve | `5` |

**Returns:** (str) Concatenated retrieved document chunks

**Raises:**
- `Exception`: If embedding model fails to load

**Performance:**
- ~100ms for embedding generation
- ~50ms for vector search
- Total: ~150ms for typical queries

**Example:**
```python
from rag import retrieve_context

context = retrieve_context("Kubernetes deployment", top_k=5)
print(context)
# Output: "Kubernetes is an orchestration platform..."
```

---

#### 3. `generate_response(prompt, context)`

Generates an LLM response with RAG-enhanced context.

**Signature:**
```python
def generate_response(prompt: str, context: str) -> str
```

**Parameters:**
| Parameter | Type | Description | Notes |
|-----------|------|-------------|-------|
| `prompt` | str | User question or request | Required |
| `context` | str | Context from RAG retrieval | Pass empty string for no context |

**Returns:** (str) LLM-generated response

**Raises:**
- `requests.exceptions.ConnectionError`: If Ollama is not running
- `Exception`: For other processing errors

**Behavior:**
- If context is empty: LLM uses foundation knowledge
- If context is provided: LLM prioritizes context
- Temperature: 0.7 (balanced)
- Max tokens: 500

**Example:**
```python
from rag import generate_response, retrieve_context

prompt = "How do I scale a Kubernetes deployment?"
context = retrieve_context(prompt)
response = generate_response(prompt, context)
print(response)
```

---

#### 4. `load_initial_knowledge(folder_path=None)`

Loads all documents from a folder into the knowledge base.

**Signature:**
```python
def load_initial_knowledge(folder_path: str | None = None) -> None
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `folder_path` | str | `code/src/data/` | Folder path containing documents |

**Supported Formats:**
- `.pdf` - PDF documents
- `.json` - JSON files  
- `.docx` - Word documents
- `.xlsx` - Excel spreadsheets

**Returns:** None

**Prints:** Progress information and any errors

**Example:**
```python
from rag import load_initial_knowledge

# Load default data folder
load_initial_knowledge()

# Or specify custom folder
load_initial_knowledge("/path/to/documents")
```

---

#### 5. `extract_text_from_file(file_path)`

Extracts text from various document formats.

**Signature:**
```python
def extract_text_from_file(file_path: str) -> str | list[str]
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str | Path to document file |

**Returns:**
- For PDF/DOCX: list[str] - List of text chunks
- For JSON: list[str] - Formatted JSON as string
- For Excel: list[str] - Rows as strings

**Raises:**
- `Exception`: If file format is unsupported or cannot be read

**Supported Formats:**
| Format | Method | Supported |
|--------|--------|-----------|
| PDF | PyMuPDF (fitz) | ✅ |
| JSON | json module | ✅ |
| DOCX | python-docx | ✅ |
| XLSX | pandas | ✅ |
| TXT | Built-in | ✅ (via code extension) |

**Example:**
```python
from rag import extract_text_from_file

# PDF
pdf_text = extract_text_from_file("document.pdf")

# JSON
json_text = extract_text_from_file("config.json")

# Excel
excel_text = extract_text_from_file("data.xlsx")
```

---

#### 6. `split_text(text, chunk_size=500, chunk_overlap=50)`

Splits text into chunks for embedding.

**Signature:**
```python
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | - | Text to split |
| `chunk_size` | int | 500 | Characters per chunk |
| `chunk_overlap` | int | 50 | Overlap between chunks |

**Returns:** list[str] - List of text chunks

**Example:**
```python
from rag import split_text

text = "Very long document text..."
chunks = split_text(text, chunk_size=300, chunk_overlap=50)
```

---

## 🤖 Agent API

### Module: `agent_setup.py`

#### 1. `run_agent(user_query)`

Executes the agent with a user query.

**Signature:**
```python
def run_agent(user_query: str) -> dict
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `user_query` | str | User's request or question |

**Returns:** dict with keys:
- `status`: Execution status
- `response`: Agent's response
- `actions`: List of executed actions
- `results`: Action results

**Example:**
```python
from agent_setup import run_agent

result = run_agent("Restart the main service")
print(result)
# {'status': 'success', 'response': '...', 'actions': [...], ...}
```

---

## 🧠 LLM Backend API

### Module: `rag.py` - LLMBackend Classes

#### Base Class: `LLMBackend`

```python
class LLMBackend:
    def call(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Send messages to LLM and get response"""
        raise NotImplementedError
```

#### Implementation: `OllamaBackend` (Primary)

**Configuration:**
```python
api_url: str = "http://localhost:11434/api/chat"
model: str = "deepseek-r1:7b"
```

**Method: `call(messages, temperature=0.7, max_tokens=500)`**

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | list[dict] | - | Chat messages in OpenAI format |
| `temperature` | float | 0.7 | Response randomness (0-1) |
| `max_tokens` | int | 500 | Max response length |

**Message Format:**
```python
messages = [
    {"role": "system", "content": "You are helpful..."},
    {"role": "user", "content": "Question here"},
]
```

**Example:**
```python
from rag import llm_backend

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain Docker"}
]

response = llm_backend.call(messages, temperature=0.7, max_tokens=500)
```

---

## ⚠️ Error Handling

### Common Errors & Solutions

#### 1. Connection Error
```python
# Error: "Cannot connect to Ollama"
# Solution: Start Ollama server
# Terminal: ollama serve
```

#### 2. Model Not Found
```python
# Error: Model deepseek-r1:7b not found
# Solution: Pull the model
# Terminal: ollama pull deepseek-r1:7b
```

#### 3. Embedding Model Load Error
```python
# Error: "Error loading embedding model"
# Solution: Ensure internet connection for first download
# or use offline cached model
```

#### 4. ChromaDB Path Error
```python
# Error: "Path not found"
# Solution:
import os
os.makedirs("gaipl-the-ai-vengers/code/src/chroma_db", exist_ok=True)
```

### Error Handling Best Practice
```python
from rag import generate_response, retrieve_context

try:
    context = retrieve_context("Your query")
    response = generate_response("Your query", context)
    print(response)
except Exception as e:
    print(f"Error: {str(e)}")
    # Fallback: Direct LLM call without RAG
    response = generate_response("Your query", "")
```

---

## 💻 Code Examples

### Example 1: Complete RAG Pipeline
```python
from rag import (
    load_initial_knowledge,
    retrieve_context,
    generate_response
)

# Step 1: Load knowledge base
print("Loading knowledge base...")
load_initial_knowledge()

# Step 2: User query
user_query = "What is Kubernetes?"

# Step 3: Retrieve context
print("Retrieving context...")
context = retrieve_context(user_query, top_k=3)

# Step 4: Generate response
print("Generating response...")
response = generate_response(user_query, context)

print(f"\nResponse:\n{response}")
```

### Example 2: Document Upload & Query
```python
from rag import (
    extract_text_from_file,
    split_text,
    add_document_to_db,
    generate_response,
    retrieve_context
)

# Upload and process document
file_path = "my_document.pdf"
text_list = extract_text_from_file(file_path)
chunks = split_text(text_list[0] if isinstance(text_list, list) else text_list)
add_document_to_db(chunks, "my_doc")

# Query the document
query = "What is mentioned about security?"
context = retrieve_context(query)
response = generate_response(query, context)
print(response)
```

### Example 3: Custom LLM Configuration
```python
from rag import generate_response, retrieve_context

# Lower temperature for more factual responses
prompt = "List security best practices"
context = retrieve_context(prompt)

# Custom parameters via LLM backend
from rag import llm_backend
messages = [
    {"role": "system", "content": "You are a security expert"},
    {"role": "user", "content": prompt}
]
response = llm_backend.call(messages, temperature=0.3, max_tokens=1000)
print(response)
```

### Example 4: Batch Processing
```python
from rag import retrieve_context, generate_response

queries = [
    "How to deploy Docker?",
    "What is Kubernetes?",
    "Explain microservices"
]

for query in queries:
    context = retrieve_context(query, top_k=3)
    response = generate_response(query, context)
    print(f"\nQ: {query}")
    print(f"A: {response}\n")
    print("-" * 50)
```

### Example 5: Handling Empty RAG Context
```python
from rag import retrieve_context, generate_response

query = "Explain quantum computing"
context = retrieve_context(query)

# Context might be empty if no relevant docs in KB
if not context.strip():
    print("No context found in KB, using foundation model knowledge")

response = generate_response(query, context)
print(response)
# LLM automatically handles empty context
```

---

## 🔗 Integration Examples

### With Flask API
```python
from flask import Flask, request, jsonify
from rag import retrieve_context, generate_response

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    prompt = data.get('prompt', '')
    
    context = retrieve_context(prompt)
    response = generate_response(prompt, context)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
```

### With FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel
from rag import retrieve_context, generate_response

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/query")
async def query(q: Query):
    context = retrieve_context(q.prompt)
    response = generate_response(q.prompt, context)
    return {"response": response}
```

### With Jupyter Notebook
```python
from rag import load_initial_knowledge, retrieve_context, generate_response

# Initialize
load_initial_knowledge()

# Interactive queries
while True:
    prompt = input("Your question: ")
    context = retrieve_context(prompt)
    response = generate_response(prompt, context)
    print(f"\nResponse:\n{response}\n")
```

---

## 📊 Response Formats

### RAG Response Structure
```python
{
    "query": "User's question",
    "context_found": bool,
    "context_preview": "First 100 chars...",
    "response": "LLM generated answer",
    "confidence": 0.85,  # Optional
    "sources": ["doc_001", "doc_002"]  # Optional
}
```

### Error Response
```python
{
    "error": True,
    "message": "Description of error",
    "code": "ERROR_CODE",
    "suggestion": "How to fix"
}
```

---

## 🚀 Performance Optimization Tips

1. **Batch Queries**: Process multiple queries at once
2. **Reduce top_k**: Use 2-3 instead of 5+ for speed
3. **Cache Results**: Store frequent queries
4. **GPU Acceleration**: Use GPU for embedding and LLM
5. **Connection Pooling**: Reuse HTTP connections

---

**API Version**: 1.0  
**Last Updated**: May 3, 2026  
**Compatible With**: DeepSeek-R1:7B, ChromaDB 0.3+
