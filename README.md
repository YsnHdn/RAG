# RAG (Retrieval-Augmented Generation) System

A question-answering system that combines document retrieval with Hugging Face models to provide context-aware responses.

## Overview

This project implements a RAG system that:
- Processes and chunks markdown documents
- Creates embeddings using Hugging Face models
- Stores vectors in a Chroma database
- Retrieves relevant context based on user queries
- Generates responses using Hugging Face's falcon-7b-instruct model

## Project Structure
```
rag/
├── data/                  # Directory for source documents
│   └── alice_in_wonderland.md
├── chroma/               # Vector database storage
├── create_dataset.py     # Script for processing documents
├── query_data.py         # Script for querying the system
├── requirements.txt      # Project dependencies
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv ragenv
source ragenv/bin/activate  # On Windows use: ragenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face API key:
```
HUGGINGFACE_API_KEY=your_key_here
```

## Usage

1. Process documents:
```bash
python create_dataset.py
```

2. Query the system:
```bash
python query_data.py "Your question here"
```

## Components

### Document Processing
- Uses LangChain's DirectoryLoader for loading markdown files
- RecursiveCharacterTextSplitter for chunking documents
- Supports UTF-8 encoded text files

### Embeddings
- Uses HuggingFace's all-MiniLM-L6-v2 model
- Local embedding generation for efficiency
- CPU-based processing by default

### Vector Store
- ChromaDB for storing and retrieving document chunks
- Persistent storage in local directory
- Efficient similarity search capabilities

### LLM Integration
- Hugging Face's falcon-7b-instruct model for generating responses
- API-based inference
- Customizable temperature and token limits

## Dependencies

Main packages:
```
# Document Processing
unstructured>=0.10.0
markdown>=3.4.0

# Vector Store & Embeddings
chromadb==0.5.23
sentence-transformers==3.3.1

# LangChain Ecosystem
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.10
langchain-huggingface>=0.0.10
langchain-chroma>=0.0.10

# Machine Learning & LLM
transformers>=4.41.0
torch>=2.0.0
huggingface-hub>=0.20.0

# Utils & Environment
python-dotenv>=0.21.0
pydantic>=2.0.0
tiktoken>=0.5.1
```

## Environment Variables

Required environment variables:
- HUGGINGFACE_API_KEY: Your Hugging Face API key for accessing models

## Implementation Details

### Document Processing
- Chunk size: 300 characters
- Chunk overlap: 100 characters
- Adds start index to each chunk for reference

### Query Processing
- Retrieves top 3 most relevant chunks for context
- Uses similarity scores for ranking
- Maintains document source tracking

### Response Generation
- Temperature: 0.7 for balanced creativity
- Max new tokens: 200 for controlled response length
- Task-specific prompt template for focused answers

## Notes

- Documents are split into chunks of 300 characters with 100 character overlap
- Each query retrieves the 3 most relevant chunks for context
- The system uses local embeddings but remote LLM inference
- All generated embeddings are persisted in the chroma directory
- The system supports multiple document types through the DirectoryLoader

## Future Improvements

Potential enhancements:
- Add support for more document types (PDF, DOCX)
- Implement batch processing for large document sets
- Add result caching for frequent queries
- Integrate alternative embedding models
- Add support for local LLM inference