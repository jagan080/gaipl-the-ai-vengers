import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import InferenceClient
import pandas as pd
import fitz  # PyMuPDF
import json
import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from docx import Document
from agent_setup import run_agent 

# ============================================
# CONFIGURATION: Choose LLM Provider (1, 2, or 3)
# ============================================
LLM_PROVIDER = 1  # Change this to switch: 1=HuggingFace, 2=Ollama, 3=Gemini Vertex AI

# ============================================
# LLM Backend Abstraction
# ============================================

class LLMBackend:
    """Base class for LLM backends"""
    def call(self, messages, temperature=0.7, max_tokens=500):
        raise NotImplementedError

class HuggingFaceBackend(LLMBackend):
    """HuggingFace Inference API Backend"""
    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            model="katanemo/Arch-Router-1.5B",
        )
    
    def call(self, messages, temperature=0.7, max_tokens=500):
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"] if response.choices else "Error: No response"
        except Exception as e:
            return f"Error: {str(e)}"

class OllamaBackend(LLMBackend):
    """Ollama Local LLM Backend"""
    def __init__(self):
        self.api_url = "http://localhost:11434/api/chat"
        self.model = "mistral"  # or "neural-chat", "llama2", etc.
    
    def call(self, messages, temperature=0.7, max_tokens=500):
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "Error: No response")
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Ensure Ollama is running on localhost:11434"
        except Exception as e:
            return f"Error: {str(e)}"

class GeminiVertexAIBackend(LLMBackend):
    """Google Vertex AI Gemini Backend"""
    def __init__(self):
        try:
            from google.cloud import aiplatform
            self.aiplatform = aiplatform
            self.aiplatform.init(project=os.getenv("GCP_PROJECT_ID"), location="us-central1")
        except ImportError:
            raise ImportError("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
    
    def call(self, messages, temperature=0.7, max_tokens=500):
        try:
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            
            model = self.aiplatform.GenerativeModel("gemini-1.5-pro")
            full_message = f"{system_message}\n\n{user_message}" if system_message else user_message
            
            response = model.generate_content(
                full_message,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize LLM Backend based on configuration
def get_llm_backend():
    """Factory function to create the appropriate LLM backend"""
    try:
        if LLM_PROVIDER == 1:
            print("Using HuggingFace Inference API")
            return HuggingFaceBackend()
        elif LLM_PROVIDER == 2:
            print("Using Ollama Local LLM")
            return OllamaBackend()
        elif LLM_PROVIDER == 3:
            print("Using Google Vertex AI Gemini")
            return GeminiVertexAIBackend()
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Choose 1, 2, or 3")
    except Exception as e:
        print(f"Error initializing LLM backend: {str(e)}")
        # Fallback to Ollama
        print("Falling back to Ollama")
        return OllamaBackend()

llm_backend = get_llm_backend()


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Lazy-load Embedding Model to avoid network issues at import time
_embedding_model = None

def get_embedding_model():
    """Lazy-load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (this may take a moment on first run)...")
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            print("Make sure you have internet connection or the model is cached locally")
            raise
    return _embedding_model

# Function to add documents to ChromaDB
def add_document_to_db(doc_text_list, doc_id):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(doc_text_list).tolist()
    for i, sentence in enumerate(doc_text_list):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embeddings[i]],
            metadatas=[{"text": sentence}]
        )


# Function to extract text from Excel file
def extract_text_from_excel(uploaded_file):
    """Extract text from Excel file (handles both file paths and file-like objects)"""
    try:
        df = pd.read_excel(uploaded_file)  # Read Excel file
        text_data = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()  # Convert each row to a string
        return text_data if text_data else []
    except Exception as e:
        print(f"Error extracting Excel: {str(e)}")
        return []

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file (handles both file paths and file-like objects)"""
    try:
        text_list = []
        with fitz.open(stream=pdf_path, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    text_list.append(page_text.strip())
        return text_list if text_list else [""]
    except Exception as e:
        print(f"Error extracting PDF: {str(e)}")
        return []


def extract_text_from_json(json_path):
    """Extract text from JSON file"""
    try:
        if isinstance(json_path, str):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Handle file-like objects
            data = json.load(json_path)
        return [json.dumps(data, indent=4)]
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        return []

def extract_text_from_docx(doc_path):
    """Extract text from DOCX file (handles both file paths and file-like objects)"""
    try:
        doc = Document(doc_path)
        text_list = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_list.append(para.text.strip())
        return text_list if text_list else [""]
    except Exception as e:
        print(f"Error extracting DOCX: {str(e)}")
        return []

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".json"):
        return extract_text_from_json(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""  # Skip unsupported files
    
def load_initial_knowledge(folder_path):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))  
        folder_path = os.path.join(base_dir, "data")
        if not os.path.exists(folder_path):
            print(f"Data folder not found: {folder_path}")
            return
        file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        for file_path in file_list:
            if os.path.isfile(file_path):
                print("Processing the document: "+file_path+ " to prepare the KB")
                try:
                    text = extract_text_from_file(file_path)
                    if text:
                        chunks = split_text(text)  
                        add_document_to_db(chunks, file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error loading initial knowledge: {str(e)}")  

# Function to retrieve relevant documents
def retrieve_context(query, top_k=3):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_texts = [doc["text"] for doc in results["metadatas"][0]]
    return " ".join(retrieved_texts)

# Function to generate response using LLM
def generate_response(prompt, context):
    messages = [
        {"role": "system", "content": """You are a highly skilled Platform Engineer AI, responsible for assisting platform support teams. Your primary role is to diagnose and resolve platform-related issues.

When relevant context from the knowledge base is provided, prioritize using that information. However, if no context is available or insufficient, you may also draw upon your own knowledge of platform engineering best practices, common troubleshooting steps, and industry standards to provide helpful solutions.

Always provide practical, actionable guidance for platform engineering challenges including:
- Server restarts and service management
- Checking service health and status
- Debugging deployment issues
- Fetching and analyzing system logs
- General platform engineering best practices"""},
        {"role": "user", "content": f"Context: {context}\n\nUser Query: {prompt}"}
    ]
    
    # Use the configured LLM backend
    return llm_backend.call(messages, temperature=0.7, max_tokens=500)
 
