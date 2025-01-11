import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set page config for a cleaner UI
st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide"
)

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
    .source-docs {
        font-size: 0.8rem;
        color: #616161;
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Custom prompt templates
MEDICAL_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question about medical topics.
If you don't know the answer, just say that you don't have enough information, don't try to make up an answer.
Provide only information that is supported by the context.

Context: {context}
Question: {question}

Start the answer directly. Keep your response clear, accurate, and helpful.
"""

GENERAL_CONVERSATION_TEMPLATE = """
You are a helpful and friendly medical assistant. Please respond to the following input in a conversational manner:

Input: {query}

Respond naturally and helpfully. For medical questions, remind the user that you're providing general information and not medical advice.
"""


@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store with cached resource"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None


@st.cache_resource
def load_llm():
    """Load the LLM with cached resource"""
    try:
        # Get HF_TOKEN from environment or Streamlit secrets
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None and hasattr(st, "secrets"):
            hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.warning("HF_TOKEN not found. Please set it in your environment variables or Streamlit secrets.")
            return None

        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task="text-generation",
            temperature=0.5,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512,
            top_p=0.9,
            repetition_penalty=1.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None


def set_custom_prompt(template):
    """Create a prompt template"""
    if "{context}" in template and "{question}" in template:
        return PromptTemplate(template=template, input_variables=["context", "question"])
    else:
        return PromptTemplate(template=template, input_variables=["query"])


def is_medical_query(query):
    """Determine if a query is medical or general conversation"""
    general_phrases = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon",
                       "good evening", "what's up", "how's it going", "nice to meet you",
                       "thanks", "thank you", "bye", "goodbye", "see you", "talk to you later"]

    query_lower = query.lower()

    # Check if query contains general conversation phrases
    for phrase in general_phrases:
        if phrase in query_lower:
            return False

    # If the query is very short, it's likely general conversation
    if len(query.split()) < 4:
        for phrase in general_phrases:
            if query_lower.startswith(phrase.split()[0]):
                return False

    return True


def handle_medical_query(query, vectorstore, llm):
    """Handle medical queries using RAG"""
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(MEDICAL_PROMPT_TEMPLATE)}
        )
        response = qa_chain.invoke({'query': query})
        return response["result"], response["source_documents"]
    except Exception as e:
        st.error(f"Error in medical query handling: {str(e)}")
        return "I'm having trouble accessing my medical knowledge database. Let me answer based on my general knowledge instead.", []


def handle_general_conversation(query, llm):
    """Handle general conversation directly with LLM"""
    try:
        general_prompt = set_custom_prompt(GENERAL_CONVERSATION_TEMPLATE)
        prompt_value = general_prompt.format(query=query)
        response = llm.invoke(prompt_value)
        return response, []
    except Exception as e:
        st.error(f"Error in general conversation handling: {str(e)}")
        return "I'm having trouble generating a response right now. Please try again later.", []


def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source_documents") and len(message["source_documents"]) > 0:
                with st.expander("View Source Documents"):
                    st.markdown('<div class="source-docs">', unsafe_allow_html=True)
                    for i, doc in enumerate(message["source_documents"]):
                        st.markdown(f"**Source {i + 1}:** {doc.page_content[:200]}...")
                    st.markdown('</div>', unsafe_allow_html=True)


def main():
    # App header
    st.markdown('<h1 class="main-header">Medical Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Ask medical questions or just chat with me</p>', unsafe_allow_html=True)

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load resources
    vectorstore = get_vectorstore()
    llm = load_llm()

    if not llm:
        st.warning("LLM could not be loaded. Please check your HF_TOKEN.")
        return

    # System status indicators
    col1, col2 = st.columns(2)
    with col1:
        if vectorstore:
            st.success("✅ FAISS Vector Store: Loaded")
        else:
            st.error("❌ FAISS Vector Store: Not loaded")

    with col2:
        if llm:
            st.success("✅ LLM: Connected")
        else:
            st.error("❌ LLM: Not connected")

    # Display chat history
    display_chat_history()

    # Chat input
    prompt = st.chat_input("Ask a medical question or just chat...")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Determine if medical or general query
                if is_medical_query(prompt) and vectorstore:
                    result, source_docs = handle_medical_query(prompt, vectorstore, llm)

                    # Display the result
                    st.markdown(result)

                    # Save to session state with source documents
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result,
                        "source_documents": source_docs
                    })

                    # Show source documents in an expander
                    if source_docs:
                        with st.expander("View Source Documents"):
                            st.markdown('<div class="source-docs">', unsafe_allow_html=True)
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i + 1}:** {doc.page_content[:200]}...")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Handle as general conversation
                    result, _ = handle_general_conversation(prompt, llm)

                    # Display the result
                    st.markdown(result)

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result,
                        "source_documents": []
                    })


if __name__ == "__main__":
    main()