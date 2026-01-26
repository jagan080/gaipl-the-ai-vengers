import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import InferenceClient
import pandas as pd
import fitz  # PyMuPDF
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import os
from agent_setup import run_agent 

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model="HuggingFaceTB/SmolLM3-3B",
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Function to add documents to ChromaDB
def add_document_to_db(doc_text_list, doc_id):
    embeddings = embedding_model.encode(doc_text_list).tolist()
    for i, sentence in enumerate(doc_text_list):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embeddings[i]],
            metadatas=[{"text": sentence}]
        )

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to extract text from Excel file
def extract_text_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)  # Read Excel file
    text_data = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()  # Convert each row to a string
    return text_data

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def extract_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=4)  # Convert JSON to text

def extract_text_from_docx(doc_path):
    doc = Document(doc_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

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
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    folder_path = os.path.join(base_dir, "data")
    file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    for file_path in file_list:
        print("Processing the document: "+file_path+ " to prepare the KB")
        text = extract_text_from_file(file_path)
        chunks = split_text(text)  
        add_document_to_db(chunks, file_path)  

# Function to retrieve relevant documents
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_texts = [doc["text"] for doc in results["metadatas"][0]]
    return " ".join(retrieved_texts)

# Function to generate response using Hugging Face LLM
def generate_response(prompt, context):
    messages = [
        {"role": "system", "content": """You are a highly skilled Platform Engineer AI, responsible for assisting platform support teams. Your primary role is to diagnose and resolve platform-related issues strictly using the provided knowledge base articles and past incident reports. You must:
Analyze the issue based on user input and match it with relevant past incidents and documentation.
Provide solutions derived strictly from the knowledge base and previous cases—do not generate responses outside this scope."""},
        {"role": "user", "content": f"Context: {context}\n\nUser Query: {prompt}"}
    ]
    response = client.chat_completion(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    
    return response.choices[0].message["content"] if response.choices else "Error: No valid response from Hugging Face"
 
