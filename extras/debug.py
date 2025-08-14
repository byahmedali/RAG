import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
from groq import Groq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from generate_embeddings import load_documents, split_documents
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create or load Chroma database
persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    documents = load_documents()
    chunks = split_documents(documents)
    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    db.persist()
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def get_response(query, context, chat_memory):
    # Prepare conversation history for context
    conversation_context = ""
    if chat_memory and len(chat_memory) > 0:
        conversation_context = "Previous conversation:\n"
        for msg in chat_memory[-5:]:  # Use last 5 exchanges for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"

    prompt = f"""You are a professional virtual assistant for the Government of Pakistan’s Ministry of Information Technology & Telecommunication (MoITT), specializing in the "National Artificial Intelligence Policy".
    Your primary role is to use the provided context to respond with accurate, concise, and helpful information about the policy’s vision, objectives, directives, targets, and related initiatives.
    You should respond to user inquiries in a professional, clear, and neutral manner, ensuring your answers are easy to understand while maintaining policy accuracy.

    If the answer is explicitly present in the retrieved context, quote or paraphrase accurately.  
    When citing, follow this style:  
    - Place citations at the end of the relevant sentence or paragraph.  
    - Use parentheses with the section name and number, e.g., *(Section 3.1 — Vision)* or *(Section 4.1.2 — Center of Excellence in AI)*.  
    - Do not use brackets like 【 】 or repeat the section name twice.  
    - If no exact section number is available, cite the nearest heading in the context.

    Only use the information provided in the context or conversation history to answer the question. **Do not fabricate or assume any details, and do not generate URLs or external references unless they are explicitly included in the context.**
    If the answer cannot be derived from the given information, politely state that the provided excerpts do not contain that information.

    Only answer the specific question asked. Do not include unrelated information or anticipate additional questions.

    Conversation history: {conversation_context}

    Context: {context}

    Question: {query}

    Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-120b",
        temperature=0,
        stream=True,
    )

    return response


def get_relevant_chunks(query: str, k: int = 3):
    if db is None:
        return []
    return db.similarity_search(query, k=k)


def stream_response(user_query: str, retrieved_context: str, chat_memory):
    return get_response(user_query, retrieved_context, chat_memory)


def save_debug_log(debug_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{log_dir}/rag_debug_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== RAG Pipeline Debug Log ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Original Query: {debug_data['original_query']}\n\n")
        f.write("=== Retrieved Chunks ===\n")
        for i, doc in enumerate(debug_data["chunks"], 1):
            f.write(f"\nChunk {i}\n")
            f.write("Content:\n")
            f.write(doc.page_content + "\n")
            f.write("Metadata:\n")
            for key, value in doc.metadata.items():
                f.write(f"{key}: {value}\n")
        f.write("\n=== Final Response ===\n")
        f.write(debug_data["final_response"])

    return filename


st.set_page_config(
    page_title="RAG Debugger", layout="centered", initial_sidebar_state="expanded"
)
st.title("RAG Pipeline Debugger")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {}

with st.sidebar:
    st.subheader("Controls")
    k_value = st.slider("Number of chunks to retrieve", 1, 10, 3)
    show_history = st.checkbox("Include chat history", value=True)
    save_logs = st.checkbox("Save debug log", value=False)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversations = {}
        st.session_state.current_chat = None
        st.session_state.chat_titles = {}
        st.session_state.conversation_memory = {}
        st.rerun()

    if db is None:
        st.warning("Chroma DB not found at ./chroma_db. Run the main app to build it.")

query = st.text_input("Enter your query:")

if query:
    # Retrieval
    with st.spinner("Retrieving relevant chunks..."):
        results = get_relevant_chunks(query, k=k_value)

    # Show table of chunks
    if results:
        rows = []
        for i, doc in enumerate(results, 1):
            rows.append(
                {
                    "Chunk #": i,
                    "Source": doc.metadata.get("source", "N/A"),
                    "Page": doc.metadata.get("page", "N/A"),
                    "Preview": (doc.page_content or "")[:160]
                    + ("..." if len(doc.page_content) > 160 else ""),
                }
            )
        st.dataframe(pd.DataFrame(rows))

        for i, doc in enumerate(results, 1):
            with st.expander(f"Chunk {i} Details"):
                st.markdown("#### Content")
                st.text(doc.page_content)
                st.markdown("#### Metadata")
                for key, value in doc.metadata.items():
                    st.write(f"{key}: {value}")
    else:
        st.info("No chunks retrieved.")

    # Prepare context
    context = "\n".join([doc.page_content for doc in results]) if results else ""

    # Response
    with st.spinner("Generating response..."):
        chat_memory = (
            st.session_state.conversation_memory.get(st.session_state.current_chat, [])
            if show_history
            else []
        )
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in stream_response(query, context, chat_memory):
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
            response_placeholder.markdown(full_response)

    # Save optional log
    if save_logs:
        log_path = save_debug_log(
            {
                "original_query": query,
                "chunks": results,
                "final_response": full_response,
            }
        )
        st.sidebar.success(f"Log saved: {log_path}")

    # Update chat history display/state
    if st.session_state.current_chat is None:
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = (
            query[:30] + "..." if len(query) > 30 else query
        )
        st.session_state.conversation_memory[chat_id] = []

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory.setdefault(
        st.session_state.current_chat, []
    ).append({"role": "user", "content": query})
    st.session_state.conversation_memory.setdefault(
        st.session_state.current_chat, []
    ).append({"role": "assistant", "content": full_response})
    st.session_state.conversations[st.session_state.current_chat] = (
        st.session_state.messages
    )

    if show_history and st.session_state.messages:
        st.markdown("## Chat History")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
