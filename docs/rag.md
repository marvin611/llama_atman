# RAG Module

This module implements a RAG system that combines document retrieval with llm generation.

## Classes

### `Document`
Base dataclass for documents to be indexed.

#### Attributes
- `id: str` - Unique identifier for the document
- `content: str` - Main text content
- `metadata: Dict[str, Any]` - Additional document metadata
- `source_type: str` - Type of document source
- `timestamp: str` - Document creation/indexing timestamp

### `WebDocument`
Extends `Document` for web-sourced content.

#### Additional Attributes
- `url: str` - Source URL
- `domain: str` - Website domain
- `title: str` - Page title
- `snippet: str` - Brief text excerpt

### `RAGResult`
Dataclass for storing retrieval results.

#### Attributes
- `contexts: List[str]` - Retrieved text contexts
- `sources: List[Dict]` - Source information
- `relevance_scores: List[float]` - Similarity scores

### `RAG`
Main RAG system implementation.

#### Constructor Parameters
- `model_handler` - Optional existing LLM handler
- `model_name: str` - Name of the LLM (default: "TinyLlama-1.1B")
- `device: str` - Computing device (default: "cuda" if available, else "cpu")
- `web_search_enabled: bool` - Enable Bing search (default: False)
- `bing_subscription_key: Optional[str]` - Bing API key
- `collection_name: str` - Vector store collection name (default: "rag_collection")
- `chunk_size: int` - Text chunk size (default: 512)
- `chunk_overlap: int` - Overlap between chunks (default: 50)

## Key Methods

### `index_document(document: Document) -> None`
Indexes a document into the vector store.

Example:
- **Input:**

```python
doc = Document(
    id="doc1",
    content="Sample content",
    metadata={"author": "John Doe"},
    source_type="text",
    timestamp="2024-03-20T10:00:00"
)
await rag.index_document(doc)
```

### `retrieve_context(query: str, top_k: int = 5, web_only: bool = False) -> RAGResult`
Retrieves relevant context for a query.

Example:
- **Input:**

```python
result = rag.retrieve_context(
    query="What is machine learning?",
    top_k=3,
    web_only=True
)
```
- **Output:** RAGResult with contexts, sources, and relevance scores

### `generate_response(query: str, context: Optional[str] = None, seed: Optional[int] = None, temperature: float = 0.7, max_new_tokens: int = 512) -> str`
Generates a response using the language model.

Example:
- **Input:**

```python
response = rag.generate_response(
    query="Explain neural networks",
    context="Neural networks are computational models...",
    temperature=0.8,
    max_new_tokens=256
)
```
- **Output:** Generated text response

## Internal Methods

### `_initialize_collection() -> None`
Sets up the Qdrant vector store collection.

### `_generate_document_id(content: str, source_type: str) -> str`
Generates a UUID for document identification.

### `_chunk_text(text: str) -> List[str]`
Splits text into overlapping chunks for indexing.

Example:
- **Input:** `text = "Long document text..."`
- **Output:** `["chunk1", "chunk2", ...]`

### `_bing_search(search_term: str) -> dict`
Performs web search using Bing API.

### `process_web_search(query: str) -> List[WebDocument]`
Processes web search results into WebDocument objects.

## Dependencies

- `torch`: PyTorch for model operations
- `fastembed`: Text embedding generation
- `qdrant_client`: Vector store operations
- `requests`: HTTP requests for web search
- `llama`: Wrapper class for llama model
- `uuid`
- `asyncio`: Asynchronous operations

## Usage Notes

1. Requires appropriate model setup and dependencies
2. Web search functionality needs valid Bing API key
3. Supports both local and web-based document retrieval
4. Uses cosine similarity for context matching
5. Implements automatic text chunking with overlap

