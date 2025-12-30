# MLflow Learning Project

A Python project for learning MLflow with RAG (Retrieval-Augmented Generation) implementation using LangChain, OpenAI embeddings, and ChromaDB vector storage.

## Features

- **PDF Data Loading**: Load and extract text from PDF documents
- **Text Chunking**: Split documents into manageable chunks for processing
- **Embeddings**: Generate embeddings using OpenAI's text-embedding-3-small model
- **Vector Storage**: Store and query embeddings with ChromaDB
- **RAG Search**: Perform retrieval-augmented generation for question answering

## Project Structure

```
mlflow_learning/
├── main.py                 # Main entry point
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── Data/                   # Data directory (contains PDF files)
│   └── NIPS-2017-attention-is-all-you-need-Paper.pdf
└── SRC/                    # Source code
    ├── __init__.py
    ├── data_loader.py      # PDF loading functionality
    ├── embeddings.py       # Text splitting and embedding generation
    ├── models.py           # OpenAI model definitions
    ├── vector_stores.py    # ChromaDB vector store management
    └── rag_search.py       # RAG search implementation
```

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd mlflow_learning
   ```

2. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Full Pipeline

Execute the main script to load data, process embeddings, and store in vector database:

```bash
uv run main.py
```

### Individual Components

- **Load PDF data**:
  ```bash
  uv run ./SRC/data_loader.py
  ```

- **Process embeddings**:
  ```bash
  # Note: Requires data loading first
  uv run ./SRC/embeddings.py
  ```

- **Query vector store**:
  ```bash
  uv run ./SRC/vector_stores.py
  ```

- **Perform RAG search**:
  ```bash
  uv run ./SRC/rag_search.py
  ```

## Dependencies

- `langchain-community`: For document loading and LangChain components
- `langchain-openai`: OpenAI integrations
- `langchain-text-splitters`: Text chunking utilities
- `langchain-chroma`: ChromaDB integration
- `pypdf`: PDF processing
- `chromadb`: Vector database
- `python-dotenv`: Environment variable management

## Configuration

- **Python Version**: >= 3.12
- **Embedding Model**: text-embedding-3-small (OpenAI)
- **LLM**: gpt-4o-mini (OpenAI)
- **Chunk Size**: 1000 characters with 200 character overlap
- **Vector Store**: ChromaDB with persistence to `SRC/chroma_db/`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Future Enhancements

- Add MLflow experiment tracking
- Implement evaluation metrics
- Add support for multiple document types
- Create web interface for querying