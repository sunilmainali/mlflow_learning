from models import get_embedding_model
from data_loader import load_data
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = load_data()
text = "\n".join(doc.page_content for doc in docs)

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split the input text into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings(chunks):
    """
    Generate embeddings for the given text chunks.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[List[float]]: List of embeddings.
    """
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings

