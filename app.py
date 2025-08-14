import streamlit as st
import os
import time
from groq import Groq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from extras.generate_embeddings import load_documents, split_documents
from extras.chat_title import generate_chat_title

load_dotenv()

# Set up Streamlit page
st.set_page_config(
    page_title="AI Policy Assistant",
    page_icon="static/logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize session state for chat history and conversations
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

# Create sidebar for chat history
with st.sidebar:
    # New Chat button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = "Start a new conversation..."
        st.session_state.conversation_memory[chat_id] = []
        st.rerun()

    # Display divider
    st.divider()

    # Display chat history
    st.title("Chat History")
    for chat_id in reversed(list(st.session_state.conversations.keys())):
        chat_title = st.session_state.chat_titles.get(chat_id, "Untitled Chat")

        # Create a button for each chat
        if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.session_state.messages = st.session_state.conversations[chat_id]
            st.rerun()

# Set up the main chat interface
st.image("static/banner.png", use_container_width="auto")
st.title("Pakistan AI Policy Assistant")
st.write(
    "Welcome! I can help you understand Pakistan's National AI Policy. Ask me any questions about the policy consultation draft."
)


# Function to get response from Groq
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
        messages=[{"role": "system", "content": prompt}],
        model="openai/gpt-oss-120b",
        temperature=0,
        stream=True,
    )

    return response


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_query = st.chat_input("Ask about Pakistan's AI Policy:")

if user_query:
    # If no current chat, create one
    if st.session_state.current_chat is None:
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = generate_chat_title(user_query)
        st.session_state.conversation_memory[chat_id] = []

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Update chat title if it's still the default placeholder
    current_title = st.session_state.chat_titles.get(st.session_state.current_chat, "")
    if (
        current_title == "Start a new conversation..."
        or current_title == "Untitled Chat"
    ):
        st.session_state.chat_titles[st.session_state.current_chat] = (
            generate_chat_title(user_query)
        )

    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # Get chat memory for current conversation
    chat_memory = st.session_state.conversation_memory.get(
        st.session_state.current_chat, []
    )

    # Add user query to conversation memory
    st.session_state.conversation_memory.setdefault(
        st.session_state.current_chat, []
    ).append({"role": "user", "content": user_query})

    # Get relevant documents from vector store using the original query
    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    # Get response from Groq
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response
        for chunk in get_response(user_query, context, chat_memory):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)

        response_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory.setdefault(
        st.session_state.current_chat, []
    ).append({"role": "assistant", "content": full_response})

    # Update the conversations dictionary
    st.session_state.conversations[st.session_state.current_chat] = (
        st.session_state.messages
    )
