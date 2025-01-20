from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import HfApi
from langchain_community.document_loaders import TextLoader 
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import shutil
from dotenv import load_dotenv

DATA_PATH = "data"
CHROMA_PATH = "chroma"

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Set Hugging Face API key
os.environ['HUGGINGFACE_API_KEY'] = 'your_huggingface_token_here'
huggingface_api_key = os.environ['HUGGINGFACE_API_KEY']

def main():
    generate_data()

def generate_data():
    documents = load_documents()
    chunks = split_documents(documents=documents)
    save_to_chroma(chunks=chunks)
    
def load_documents():
    
    if not os.path.exists(DATA_PATH):
        print("No path found")   
        return None 
    
    try:
        loader = DirectoryLoader(
            path=DATA_PATH,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}# This will treat MD files as plain text
        )  
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

def split_documents(documents : list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents=documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
        return chunks


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

def save_to_chroma(chunks: list[Document]):
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name="my_documents"
        )
        
        print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}")
        return db
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        raise e
    
if __name__ == "__main__":
    main()