# 🏗️ System Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Integration Points](#integration-points)
6. [Technology Stack](#technology-stack)
7. [Deployment Architecture](#deployment-architecture)

---

## 📋 System Overview

**The AI Vengers** is an integrated platform intelligence system that combines:
- **LLM-powered intelligence** (DeepSeek-R1:7B)
- **Semantic search capabilities** (RAG)
- **Agentic automation** (Agent-driven task execution)
- **Knowledge management** (ChromaDB vector store)

The system processes user queries through a multi-stage pipeline to deliver contextual, actionable responses.

---

## 🎨 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                         │
│                    (Streamlit / Chat Interface)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   User Query Input      │
                    │   (Text Prompt)         │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        │                        ▼
┌──────────────────┐    ┌────────▼────────┐    ┌──────────────────┐
│ Embedding Model  │    │ Query Processing│    │  Agent Router    │
│ (all-MiniLM-L6)  │    │                  │    │  (Agent Setup)   │
└────────┬─────────┘    └────────┬────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       │                       ▼
┌──────────────────┐            │          ┌──────────────────────┐
│  Vector Search   │            │          │ Tool Execution Layer │
│   (ChromaDB)     │            │          │ - Service Restarts   │
│                  │            │          │ - Log Retrieval      │
│ Retrieved Docs   │            │          │ - Health Checks      │
└────────┬─────────┘            │          └──────────┬───────────┘
         │                       │                     │
         └───────────┬───────────┴─────────┬───────────┘
                     │                     │
            ┌────────▼────────┐   ┌────────▼────────┐
            │ RAG Context     │   │ Action Results  │
            │ (Retrieved      │   │ (Execution Info)│
            │  Docs)          │   │                 │
            └────────┬────────┘   └────────┬────────┘
                     │                     │
                     └────────┬────────────┘
                              │
                     ┌────────▼──────────┐
                     │  LLM Backend      │
                     │ (DeepSeek-R1:7B)  │
                     │  via Ollama       │
                     │ ┌────────────────┐│
                     │ │ System Prompt  ││
                     │ │ • Handles RAG  ││
                     │ │ • Fallback to  ││
                     │ │   Foundation   ││
                     │ │   Knowledge    ││
                     │ └────────────────┘│
                     └────────┬──────────┘
                              │
                     ┌────────▼──────────┐
                     │ Response Generator│
                     │ - Format Results  │
                     │ - Ensure Quality  │
                     │ - Error Handling  │
                     └────────┬──────────┘
                              │
                     ┌────────▼──────────┐
                     │ User Response     │
                     │ - LLM Answer      │
                     │ - Recommendations │
                     │ - Action Options  │
                     └───────────────────┘
```

---

## 🔧 Core Components

### 1. **RAG Pipeline**
**Location**: `code/src/rag.py`

**Responsibilities:**
- Document ingestion and embedding
- Semantic search via ChromaDB
- Context retrieval
- LLM response generation

**Key Classes:**
```python
├── LLMBackend (Abstract)
│   ├── HuggingFaceBackend
│   ├── OllamaBackend (Primary)
│   └── GeminiVertexAIBackend
└── RAG Functions
    ├── add_document_to_db()
    ├── retrieve_context()
    ├── generate_response()
    └── load_initial_knowledge()
```

### 2. **Agent System**
**Location**: `code/src/agent_setup.py`

**Responsibilities:**
- Agent initialization and configuration
- Tool binding and execution
- Workflow orchestration

**Key Functions:**
```python
└── Agent Functions
    ├── run_agent()
    ├── Tool registration
    └── Execution management
```

### 3. **Knowledge Base**
**Location**: `code/src/chroma_db/`

**Details:**
- Vector database: ChromaDB
- Collection: `rag_docs`
- Embedding dimension: 384
- Persistence: Disk-based

### 4. **Data Layer**
**Location**: `code/src/data/`

**Supported Formats:**
- PDF documents
- JSON files
- DOCX files
- Excel spreadsheets

---

## 🔄 Data Flow

### Flow 1: Knowledge Base Loading
```
Data Files (PDF/JSON/DOCX/Excel)
    ↓
Text Extraction (format-specific)
    ↓
Text Chunking (500 char chunks)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
ChromaDB Indexing
    ↓
Knowledge Base Ready
```

### Flow 2: Query Processing
```
User Query
    ↓
Embedding (query vector)
    ↓
Vector Search (ChromaDB, top-k=3)
    ↓
Context Retrieval (concatenate results)
    ↓
Check for Empty Context
    ├─ Has Content: Include in prompt
    └─ Empty: Notify LLM to use foundation knowledge
    ↓
LLM Processing (DeepSeek-R1:7B)
    ↓
Response Generation
    ↓
User Output
```

### Flow 3: Agent-Driven Execution
```
User Query → RAG Response + Recommendations
    ↓
Agent Analysis (via LLM)
    ↓
Tool Selection (from available tools)
    ↓
Action Execution (if auto-approved)
    ↓
Result Collection
    ↓
Follow-up LLM Processing
    ↓
Final Response with Execution Results
```

---

## 🔌 Integration Points

### 1. **LLM Integration (Ollama)**
- **Endpoint**: `http://localhost:11434/api/chat`
- **Model**: `deepseek-r1:7b`
- **Protocol**: REST API (JSON)
- **Format**: Message-based chat

### 2. **Vector Database (ChromaDB)**
- **Type**: Persistent local storage
- **Location**: `code/src/chroma_db/`
- **Access**: Python SDK
- **Collection**: `rag_docs`

### 3. **Embedding Service**
- **Model**: `all-MiniLM-L6-v2`
- **Library**: SentenceTransformers
- **Execution**: Local (CPU/GPU)
- **Caching**: Auto-download on first use

### 4. **Document Processing**
- **PDF**: PyMuPDF (fitz)
- **JSON**: Python json
- **DOCX**: python-docx
- **Excel**: pandas

### 5. **Agent Tools** (Extensible)
- Server restart commands
- Service health checks
- Log file retrieval
- Custom executable actions

---

## 💻 Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | DeepSeek-R1:7B | Reasoning and response generation |
| Inference | Ollama | Local LLM serving |
| Vector DB | ChromaDB | Semantic search storage |
| Embeddings | SentenceTransformers | Query and doc embedding |
| Text Split | LangChain | Document chunking |
| Document Parsing | PyMuPDF, pandas, python-docx | Multi-format support |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Interactive web interface |
| Chat Interface | Streamlit Chat | User interaction |
| Visualization | Streamlit Components | Data display |

### Deployment
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Consistent environments |
| Orchestration | Docker Compose | Multi-service management |
| Communication | REST APIs | Service-to-service |

---

## 🚀 Deployment Architecture

### Development Setup
```
┌─────────────────────────────────────────┐
│         Developer Machine               │
├─────────────────────────────────────────┤
│  Python Virtual Environment (venv)      │
│  ├─ RAG System                          │
│  ├─ Agent System                        │
│  ├─ Ollama Local (localhost:11434)      │
│  └─ ChromaDB (code/src/chroma_db/)      │
│                                         │
│  Jupyter/VS Code                        │
│  └─ Test & Development                  │
└─────────────────────────────────────────┘
```

### Production Setup (Docker)
```
┌─────────────────────────────────────────────┐
│           Docker Container                  │
├─────────────────────────────────────────────┤
│  Python Application                         │
│  ├─ RAG Pipeline                           │
│  ├─ Agent System                           │
│  └─ Streamlit Frontend                     │
│                                            │
│  Exposed Ports:                            │
│  ├─ 8501 (Streamlit)                       │
│  └─ 8000 (Optional: FastAPI)               │
├─────────────────────────────────────────────┤
│  External Services (Host)                  │
│  ├─ Ollama Server (localhost:11434)        │
│  └─ ChromaDB Storage (mounted volume)      │
└─────────────────────────────────────────────┘
```

### Multi-Service Deployment (Docker Compose)
```
docker-compose.yml
├─ ai-vengers-app
│  ├─ Streamlit (8501)
│  ├─ RAG System
│  └─ Agent System
│
├─ ollama-service (Optional)
│  └─ DeepSeek-R1:7B (11434)
│
└─ volumes
   └─ chroma_db/ (Persistent storage)
```

---

## 🔐 Security Considerations

### Current Implementation
- Local-only deployment (no external APIs)
- No authentication layer (development mode)
- No encryption (optional for deployment)

### Production Recommendations
1. **API Authentication**: Add API keys for external access
2. **Data Encryption**: Encrypt stored documents and embeddings
3. **Rate Limiting**: Implement request rate limits
4. **Input Validation**: Sanitize all user inputs
5. **Access Control**: Role-based access for sensitive operations

---

## 📊 Performance Characteristics

### Latency
- Query embedding: ~100ms
- Vector search: ~50ms (1000 docs)
- LLM response: 5-30s
- Total E2E: 5-30s

### Throughput
- Sequential processing (single user)
- Can handle multiple concurrent users with threading
- Consider load balancing for high concurrency

### Storage
- ChromaDB: ~1GB per 10,000 documents (384-dim embeddings)
- Model weights: ~4.6GB (DeepSeek-R1:7B)
- Documents: Variable (PDF, JSON, etc.)

---

## 🔄 Scalability Options

### Horizontal Scaling
1. **Multiple RAG instances** behind load balancer
2. **Distributed ChromaDB** (supported by ChromaDB Server)
3. **Load balancing** with nginx/haproxy

### Vertical Scaling
1. **Larger embedding model** for better quality
2. **Larger LLM model** for better reasoning
3. **GPU acceleration** for faster inference

---

## 🛠️ Configuration Management

### Environment Variables
```bash
HUGGINGFACE_API_KEY=     # For HF backend
GCP_PROJECT_ID=          # For Gemini backend
LLM_PROVIDER=2           # 1=HF, 2=Ollama, 3=Gemini
OLLAMA_BASE_URL=http://localhost:11434
```

### Configuration Files
- `code/src/rag.py` - LLM provider selection
- `code/src/chroma_db/` - Vector store location
- `code/src/data/` - Knowledge base documents

---

## 📈 Monitoring & Observability

### Recommended Additions
1. **Logging**: Comprehensive logging for all operations
2. **Metrics**: Response time, accuracy, retrieval quality
3. **Tracing**: End-to-end request tracing
4. **Health Checks**: Service health monitoring
5. **Alerting**: Operational alerts

---

## 🎯 Future Enhancements

1. **Multi-Modal Support**: Images, audio, video
2. **Fine-tuning**: Custom model fine-tuning
3. **Caching**: Response caching for common queries
4. **Streaming**: Real-time response streaming
5. **Analytics**: Usage analytics and insights
6. **Advanced RAG**: Multi-hop retrieval, reranking

---

**Architecture Version**: 1.0  
**Last Updated**: May 3, 2026  
**Maintained By**: AI Vengers Team
