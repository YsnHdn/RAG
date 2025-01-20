from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import argparse

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text to search for")
    args = parser.parse_args()
    
    # Initialize embedding and DB
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name="my_documents"
    )
    
    # Search DB
    results = db.similarity_search_with_relevance_scores(args.query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Initialize model
    model = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",  # Different model
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,
    task="text-generation",
    max_new_tokens=200,
)
    
    # Format prompt and get response
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Answer: """)
    
    prompt = prompt_template.format(context=context_text, question=args.query_text)
    response = model.invoke(prompt)
    return response

if __name__ == "__main__":
    print(main())