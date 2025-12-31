from langchain_chroma import Chroma
from models import get_embedding_model
from data_loader import load_data
from embeddings import split_text


from openai import OpenAI


# Load and prepare data
docs = load_data()
if docs:
    text = "\n".join(doc.page_content for doc in docs)
    chunks = split_text(text)
else:
    chunks = []

vector_store = Chroma(
    collection_name="pdf_embeddings",
    embedding_function=get_embedding_model(),
    persist_directory="SRC/chroma_db"
)

# Add chunks to vector store if not empty
if chunks:
    vector_store.add_texts(chunks)

def add_texts_to_vector_store(texts):
    """
    Add texts to the Chroma vector store.

    Args:
        texts (List[str]): List of text chunks to add.
    """
    vector_store.add_texts(texts)
    # Persistence is automatic with persist_directory set

def query_vector_store(query, top_k=5):
    """
    Query the vector store for similar documents.

    Args:
        query (str): The query string.
        top_k (int): Number of top similar documents to retrieve.
        """
    results = vector_store.similarity_search(query, k=top_k)
    return results

if __name__ == "__main__":
    if chunks:
        query = "what is attention mechanism?"
        results = query_vector_store(query)
        for res in results:
            print(res.page_content)
    else:
        print("No data loaded.")
