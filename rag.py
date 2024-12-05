from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
import requests
from urllib.parse import urlparse
import hashlib
from datetime import datetime
from llama import Llama
import uuid
import asyncio

@dataclass
class Document:
    """Base class for documents to be indexed"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_type: str
    timestamp: str

@dataclass
class WebDocument(Document):
    url: str
    domain: str
    title: str
    snippet: str


@dataclass
class RAGResult:
    """Structure for RAG query results"""
    contexts: List[str]
    sources: List[Dict]
    relevance_scores: List[float]

class RAG:
    def __init__(
        self,
        model_handler=None,
        model_name: str = "TinyLlama-1.1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        web_search_enabled: bool = False,
        bing_subscription_key: Optional[str] = None,
        collection_name: str = "rag_collection",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.device = device
        self.web_search_enabled = web_search_enabled
        self.bing_subscription_key = bing_subscription_key
        
        # Use existing model handler if provided
        if model_handler:
            self.llm = model_handler
        else:
            self.llm = Llama(model_name=model_name, device=device)
            if not self.llm.setup():
                raise Exception("Failed to set up model")
        
        # Vector store initialization
        self.collection_name = collection_name
        self.vector_store = QdrantClient(":memory:") # Currently only in-memory; could be changed to a persistent store
        self._initialize_collection()
        
        # Embedding setup
        self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _initialize_collection(self) -> None:
        """
        Initializes or verifies the existence of a Qdrant collection for storing vector embeddings.
        
        This method attempts to retrieve the specified collection from the vector store. If the collection does not exist, it creates a new collection with the specified name and configuration. The configuration includes the embedding dimension size set to 384 and the distance metric set to cosine similarity.
        """
        try:
            self.vector_store.get_collection(self.collection_name)
        except Exception:
            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # Dimensionality of the embedding vectors
                    distance=Distance.COSINE  # Distance metric for similarity calculations
                )
            )

    def _generate_document_id(self, content: str, source_type: str) -> str:
        """Generate unique document ID as UUID"""
        return str(uuid.uuid4())

    def _chunk_text(self, text: str) -> List[str]:
        """
        Splits the input text into smaller chunks with overlap.

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[str]: A list of chunked text strings.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
            else:
                # Find the last period or space before chunk_size
                last_period = text.rfind('.', start, end)
                last_space = text.rfind(' ', start, end)
                split_point = max(last_period, last_space)
                if split_point == -1:
                    split_point = end
                chunks.append(text[start:split_point])
            start = end - self.chunk_overlap
        return chunks

    def _bing_search(self, search_term: str) -> dict:
        """
        Performs a Bing search using the provided search term and returns the search results as a dictionary.

        This method sends a GET request to the Bing Search API with the specified search term and parameters. It requires a valid Bing subscription key to be set in the instance. The search results are returned as a JSON object.

        Args:
            search_term (str): The search term to query Bing with.

        Returns:
            dict: A dictionary containing the search results from Bing.

        Raises:
            ValueError: If the Bing subscription key is not provided.
        """
        if not self.bing_subscription_key:
            raise ValueError("Bing subscription key not provided")
            
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_subscription_key}
        params = {
            "q": search_term,
            "textDecorations": True,
            "textFormat": "HTML",
            "count": 5
        }
        
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    async def index_document(self, document: Document) -> None:
        """
        Indexes a document into the vector store by breaking it down into chunks, embedding each chunk, and storing them with their metadata.

        This method takes a document, splits its content into overlapping chunks, embeds each chunk using the embedding model, and then stores the embedded chunks along with their metadata in the vector store. The metadata includes the document ID, chunk index, source type, metadata, and timestamp.

        Args:
            document (Document): The document to be indexed.

        Returns:
            None
        """
        chunks = self._chunk_text(document.content)
        
        for i, chunk in enumerate(chunks):
            chunk_embedding = list(self.embedding_model.embed([chunk]))[0]
            
            # Create point for vector store with UUID
            point_id = str(uuid.uuid4())  # Generate UUID for each chunk
            point = PointStruct(
                id=point_id,
                vector=chunk_embedding.tolist(),
                payload={
                    "text": chunk,
                    "document_id": document.id,
                    "chunk_index": i,
                    "source_type": document.source_type,
                    "metadata": document.metadata,
                    "timestamp": document.timestamp
                }
            )
            
            self.vector_store.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

    async def process_web_search(self, query: str) -> List[WebDocument]:
        """
        Processes web search results into a list of WebDocument objects.

        This method takes a search query, performs a web search using the Bing API, and then converts the search results into a list of WebDocument objects. Each WebDocument object contains metadata such as the document ID, content, source type, URL, domain, title, snippet, and timestamp.

        Args:
            query (str): The search query to perform.

        Returns:
            List[WebDocument]: A list of WebDocument objects representing the search results.
        """
        search_results = self._bing_search(query)
        documents = []
        
        for result in search_results.get('webPages', {}).get('value', []):
            doc = WebDocument(
                id=self._generate_document_id(result['url'], 'web'),
                content=f"{result.get('name')}\n{result.get('snippet')}",
                metadata={
                    "search_query": query,
                    "rank": len(documents)
                },
                source_type="web",
                url=result['url'],
                domain=urlparse(result['url']).netloc,
                title=result.get('name', ''),
                snippet=result.get('snippet', ''),
                timestamp=datetime.now().isoformat()
            )
            documents.append(doc)
        
        return documents

    def retrieve_context(
        self, 
        query: str,
        top_k: int = 5,
        web_only: bool = False
    ) -> RAGResult:
        """
        Retrieves relevant context based on the query, with options to limit the number of results and specify the source type.

        This method retrieves the most relevant context for a given query. It can perform either a web search or a local search within the vector store, depending on the `web_only` parameter. The number of results is limited by the `top_k` parameter.

        Args:
            query (str): The search query to perform.
            top_k (int, optional): The maximum number of results to retrieve. Defaults to 5.
            web_only (bool, optional): If True, only perform a web search. Defaults to False.

        Returns:
            RAGResult: An object containing the retrieved contexts, their sources, and relevance scores.
        """
        contexts = []
        sources = []
        scores = []

        if self.web_search_enabled and web_only:
            # Perform web search if enabled and web_only is True
            try:
                web_docs = asyncio.run(self.process_web_search(query))
                for doc in web_docs[:top_k]:
                    contexts.append(doc.content)
                    sources.append({
                        "type": "web",
                        "metadata": {
                            "url": doc.url,
                            "title": doc.title,
                            "domain": doc.domain
                        }
                    })
                    scores.append(0.9 - (len(sources) * 0.1))  # Simple decay for ranking
            except Exception as e:
                print(f"Error in web search: {e}")
        else:
            # Perform local search if web search is disabled or web_only is False
            query_embedding = list(self.embedding_model.embed([query]))[0]
            search_results = self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            for hit in search_results:
                contexts.append(hit.payload["text"])
                sources.append({
                    "type": hit.payload["source_type"],
                    "metadata": hit.payload["metadata"]
                })
                scores.append(hit.score)

        return RAGResult(
            contexts=contexts,
            sources=sources,
            relevance_scores=scores
        )

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generates a response using the language model based on the provided query and optional context.

        Parameters:
        - query (str): The question or prompt to generate a response for.
        - context (Optional[str]): Optional context to base the response on. If provided, the response will be generated using this context.
        - seed (Optional[int]): Optional seed for the random number generator to ensure reproducibility.
        - temperature (float): The temperature parameter for the language model to control randomness. Defaults to 0.7.
        - max_new_tokens (int): The maximum number of tokens in the generated response. Defaults to 512.
        - **kwargs: Additional keyword arguments to pass to the language model's generate method.

        Returns:
        - str: The generated response based on the query and context.
        """
        if context:
            formatted_prompt = (
                "Based on the following context, provide a clear and focused answer to the question. "
                "Be concise and specific. Use only information from the context. "
                "If you cannot answer from the context, say so.\n\n"
                f"Context: {context}\n\n"
                f"Question: {query}\n\n"
                "Answer: "
            )
        else:
            formatted_prompt = (
                "Provide a clear and concise answer to the following question. "
                f"Question: {query}\n"
                "Answer: "
            )

        try:
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                
            # Use the Llama class's generate method directly
            response = self.llm.generate(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                **kwargs
            )
            
            # Clean up response if needed
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].strip()
                
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating the response."


