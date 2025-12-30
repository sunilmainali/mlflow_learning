from langchain_community.document_loaders import PyPDFLoader

def load_data(filepath="Data/NIPS-2017-attention-is-all-you-need-Paper.pdf"):
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

if __name__ == "__main__":
    docs = load_data()
    print(f"Number of pages: {len(docs)}")
    for doc in docs:
        print(doc.page_content)