# Medical Knowledge Assistant

A Streamlit-based chatbot that uses Retrieval Augmented Generation (RAG) to answer medical questions with precision and accuracy.

## Overview

This Medical Knowledge Assistant is an intelligent chatbot built with Streamlit, LangChain, and Hugging Face's Mistral-7B-Instruct model. It uses a FAISS vector database to retrieve relevant medical information from indexed documents, providing accurate and contextual responses to medical queries while also handling general conversation.

## Features

- **Dual-Mode Operation**: Automatically distinguishes between medical queries (using RAG) and general conversation
- **Interactive Chat Interface**: Clean, professional UI with chat history and expandable source documents
- **Medical Knowledge Base**: Leverages FAISS vector store for efficient retrieval of medical information
- **Error Handling**: Graceful fallbacks and clear error messages
- **Transparency**: Shows source documents for medical responses

## Installation

### Prerequisites

- Python 3.8 or higher
- Hugging Face account with API token
- Sufficient disk space for model and vector store

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/medical-knowledge-assistant.git
   cd medical-knowledge-assistant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   
   # For Windows
   venv\Scripts\activate
   
   # For macOS/Linux
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face API token:
   
   **Option 1**: Create a Streamlit secrets file:
   ```bash
   mkdir -p ~/.streamlit
   echo 'HF_TOKEN = "your_hugging_face_token_here"' > ~/.streamlit/secrets.toml
   ```
   
   **Option 2**: Set as environment variable:
   ```bash
   # For Windows
   set HF_TOKEN=your_hugging_face_token_here
   
   # For macOS/Linux
   export HF_TOKEN=your_hugging_face_token_here
   ```

### Building the Knowledge Base

If you already have the vector store built, ensure it's in the `vectorstore/db_faiss` directory. If not, run the knowledge base creation script:

```bash
python create_knowledge_base.py
```

This will:
1. Load PDF documents from the `data/` directory
2. Split them into manageable chunks
3. Create embeddings using the Hugging Face model
4. Store the embeddings in a FAISS vector database

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at [http://localhost:8501](http://localhost:8501)

3. Start chatting with the assistant:
   - Ask medical questions to get responses based on the vector store
   - Use casual greetings for general conversation

## Project Structure

```
medical-knowledge-assistant/
├── app.py                    # Main Streamlit application
├── create_knowledge_base.py  # Script to create FAISS vector store
├── requirements.txt          # Python dependencies
├── data/                     # Directory containing PDF documents
└── vectorstore/              # Directory containing FAISS vector store
    └── db_faiss/             # FAISS vector store files
```

## Requirements

```
streamlit>=1.30.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
langchain-community>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
```

## Troubleshooting

### Common Issues

1. **"Failed to load LLM" error**:
   - Ensure your Hugging Face token is correct and has proper permissions
   - Check that you have the "Make calls to Inference Providers" permission enabled in your Hugging Face account

2. **"Failed to load vector store" error**:
   - Verify that the `vectorstore/db_faiss` directory exists and contains FAISS index files
   - Run the knowledge base creation script if needed

3. **Slow or no responses**:
   - Check your internet connection
   - The first query might take longer as the model is loaded

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://www.langchain.com/) for the framework
- [Hugging Face](https://huggingface.co/) for the model hosting
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database
