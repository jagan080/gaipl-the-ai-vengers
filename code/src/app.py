import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from agent_setup import run_agent
from rag import extract_text_from_excel,extract_text_from_pdf, extract_text_from_docx,add_document_to_db, retrieve_context, generate_response, load_initial_knowledge

# Mock action suggestions (can be expanded dynamically)
actions = [
    {"name": "Restart Service"},
    {"name": "Get Service Status"},
    {"name": "Get System Logs"},
    {"name": "Send Email"}
]

# Set the Streamlit page layout
st.set_page_config(page_title="Platform Engineer Chatbot", layout="wide")

st.title("🤖 Platform Engineer Assistant")

# File Upload
uploaded_file = st.file_uploader("Update Knowledge Base", type=["xlsx", "pdf", "docx"])
if uploaded_file:
    st.write("Processing document...")
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    if file_type == "xlsx":
        document_text = extract_text_from_excel(uploaded_file)
    elif file_type == "pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        document_text = extract_text_from_docx(uploaded_file)
    
    add_document_to_db(document_text, uploaded_file.name)
    st.success("Document added to ChromaDB!")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load initial knowledge only once
if "knowledge_loaded" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        load_initial_knowledge("data")
    st.session_state.knowledge_loaded = True

# Sidebar Info
with st.sidebar:
    st.markdown("### 👨‍💻 About Me")
    st.write(
        "I'm a **Platform Engineer Assistant**, here to help with:\n"
        "- Server restarts 🔄\n"
        "- Checking service health ✅\n"
        "- Debugging deployment issues 🛠️\n"
        "- Fetching logs 📜\n"
        "- Ask me anything!"
    )
# Chat UI
st.markdown("### 💬 Chat with the AI")

# User Input Field
user_input = st.chat_input("Ask me anything about platform engineering...")
if user_input and isinstance(user_input, str):
    st.session_state.chat_history.append(HumanMessage(content=user_input))

if user_input:
        context = retrieve_context(user_input)
        print("the context is : "+context)
        if context:
            answer = generate_response(user_input, context)
            st.session_state.chat_history.append(AIMessage(content=answer))
        else:
            st.session_state.chat_history.append(AIMessage(content="No relevant context found!"))

# Display chat history in a conversational format
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown(f"🟢 **Platform Engineer:** {message.content}")
    else:
        st.markdown(f"🤖 **AI Bot:** {message.content}")
        
        
# Display suggested actions
if actions and len(st.session_state.chat_history) > 1:
    st.markdown("### Agentic AI:")
    for action in actions:
        if st.button(action["name"]):
            st.session_state.chat_history.append(HumanMessage(content=action["name"]))
            action_response = run_agent(action["name"])
            st.session_state.chat_history.append(AIMessage(content=action_response.messages[1].content))
            st.rerun()