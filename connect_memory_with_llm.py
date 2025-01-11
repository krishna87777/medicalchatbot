import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        max_new_tokens=512,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return llm


# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# For general conversation without using the database
GENERAL_CONVERSATION_TEMPLATE = """
You are a helpful and friendly assistant. Please respond to the following input in a conversational manner:

Input: {query}

Respond naturally and helpfully:
"""


def set_general_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["query"])
    return prompt


# Function to determine if the query requires medical knowledge or is general conversation
def is_general_conversation(query):
    general_phrases = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon",
                       "good evening", "what's up", "how's it going", "nice to meet you",
                       "thanks", "thank you", "bye", "goodbye", "see you", "talk to you later"]

    query_lower = query.lower()

    # Check if query contains general conversation phrases
    for phrase in general_phrases:
        if phrase in query_lower:
            return True

    # If the query is very short, it's likely general conversation
    if len(query.split()) < 4:
        for phrase in general_phrases:
            if query_lower.startswith(phrase.split()[0]):
                return True

    return False


# Function to handle general conversation with direct LLM
def handle_general_conversation(llm, query):
    general_prompt = set_general_prompt(GENERAL_CONVERSATION_TEMPLATE)
    prompt_value = general_prompt.format(query=query)
    response = llm.invoke(prompt_value)
    return response


# Load Database
try:
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    db_loaded = True
except Exception as e:
    print(f"Warning: Could not load database: {e}")
    print("Continuing with general conversation only...")
    db_loaded = False


# Main loop
def main():
    llm = load_llm(HUGGINGFACE_REPO_ID)

    while True:
        user_query = input("\nPlease Write Query Here (or type 'exit' to quit): ")

        if user_query.lower() == 'exit':
            print("Thank you for chatting. Goodbye!")
            break

        # Determine if the query is general conversation or requires medical knowledge
        if is_general_conversation(user_query):
            print("\nProcessing as general conversation...")
            response = handle_general_conversation(llm, user_query)
            print("\nRESPONSE: ", response)
        else:
            if db_loaded:
                print("\nSearching medical knowledge base...")
                try:
                    response = qa_chain.invoke({'query': user_query})
                    print("\nRESULT: ", response["result"])
                    print("\nSOURCE DOCUMENTS: ", response["source_documents"])
                except Exception as e:
                    print(f"\nError in retrieval: {e}")
                    print("\nFalling back to general LLM response...")
                    response = handle_general_conversation(llm, user_query)
                    print("\nRESPONSE: ", response)
            else:
                # If database failed to load, use general conversation for all queries
                response = handle_general_conversation(llm, user_query)
                print("\nRESPONSE: ", response)


if __name__ == "__main__":
    main()