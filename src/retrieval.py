"""
retrieval system combining embedding and vector store
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.embedding import VNLawEmbedder
from src.vector_store import FAISSVectorStore


class RetrievalSystem:
    """retrieval system for semantic search"""
    
    def __init__(
        self,
        embedder: Optional[VNLawEmbedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        model_name: str = "anhtld/VN-Law-Embedding",
        model_cache_dir: Optional[str] = None
    ):
        """init retrieval system"""
        # init embedder
        if embedder is None:
            self.embedder = VNLawEmbedder(
                model_name=model_name,
                model_cache_dir=model_cache_dir
            )
        else:
            self.embedder = embedder
        
        # Initialize vector store
        if vector_store is None:
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedder.get_embedding_dimension()
            )
        else:
            self.vector_store = vector_store
    
    def add_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 32
    ):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys (and optionally other metadata)
            batch_size: Batch size for encoding
        """
        if not documents:
            return
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Create embeddings
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.embedder.encode_batch(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # Prepare metadata (preserve all fields from documents)
        metadata = []
        for doc in documents:
            metadata.append({
                'id': doc.get('id', ''),
                'text': doc.get('text', ''),
                'source': doc.get('source', ''),
                'chunk_index': doc.get('chunk_index', 0),
                'original_id': doc.get('original_id', doc.get('id', ''))
            })
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, metadata)
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            return_scores: Whether to include similarity scores in results
            
        Returns:
            List of dictionaries with document information and optionally scores
        """
        # Encode query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Search
        distances, indices = self.vector_store.search(query_embedding, k=k)
        
        # Get metadata
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            metadata = self.vector_store.metadata[idx]
            result = {
                'rank': i + 1,
                'id': metadata.get('id', ''),
                'text': metadata.get('text', ''),
                'source': metadata.get('source', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'original_id': metadata.get('original_id', '')
            }
            if return_scores:
                result['score'] = float(score)
            results.append(result)
        
        return results
    
    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 10,
        return_scores: bool = True
    ) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of documents to retrieve per query
            return_scores: Whether to include similarity scores
            
        Returns:
            List of result lists, one per query
        """
        # Encode all queries
        query_embeddings = self.embedder.encode_batch(queries)
        
        # Search for all queries
        distances, indices = self.vector_store.search(query_embeddings, k=k)
        
        # Process results
        all_results = []
        for query_idx in range(len(queries)):
            query_results = []
            for rank, (score, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                metadata = self.vector_store.metadata[idx]
                result = {
                    'rank': rank + 1,
                    'id': metadata.get('id', ''),
                    'text': metadata.get('text', ''),
                    'source': metadata.get('source', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'original_id': metadata.get('original_id', '')
                }
                if return_scores:
                    result['score'] = float(score)
                query_results.append(result)
            all_results.append(query_results)
        
        return all_results
    
    def save(self, directory: str):
        """
        Save retrieval system (vector store) to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        self.vector_store.save(directory)
    
    @classmethod
    def load(
        cls,
        directory: str,
        embedder: Optional[VNLawEmbedder] = None,
        model_name: str = "anhtld/VN-Law-Embedding",
        model_cache_dir: Optional[str] = None
    ) -> 'RetrievalSystem':
        """
        Load retrieval system from disk.
        
        Args:
            directory: Directory to load from
            embedder: VNLawEmbedder instance (will create if None)
            model_name: Model name for embedder (if creating new)
            model_cache_dir: Model cache directory (if creating new)
            
        Returns:
            RetrievalSystem instance
        """
        # Load or create embedder
        if embedder is None:
            embedder = VNLawEmbedder(
                model_name=model_name,
                model_cache_dir=model_cache_dir
            )
        
        # Load vector store
        vector_store = FAISSVectorStore.load(directory)
        
        # Create instance
        instance = cls(embedder=embedder, vector_store=vector_store)
        
        return instance
    
    def get_stats(self) -> Dict:
        """Get statistics about the retrieval system."""
        return {
            'num_documents': self.vector_store.get_size(),
            'embedding_dim': self.embedder.get_embedding_dimension(),
            'max_seq_length': self.embedder.get_max_seq_length()
        }

