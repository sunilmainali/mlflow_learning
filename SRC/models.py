import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

def get_embedding_model(model_name="text-embedding-3-small"):
    """
    Initialize and return the OpenAI Embedding model.

    Args:
        model_name (str): The name of the OpenAI embedding model to use.
        """
    embedding_model= OpenAIEmbeddings(model=model_name)
    return embedding_model

def get_llm_model():
    """
    Initialize and return the ChatOpenAI LLM model.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")
    
    llm_model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=800,
    )
    return llm_model