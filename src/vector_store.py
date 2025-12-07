"""
faiss vector store
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """faiss vector store"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        nlist: int = 100
    ):
        """init faiss vector store"""
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        
        # init index
        if index_type == "flat":
            # flat index for exact search
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "ivf":
            # ivf index for approximate search (faster)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index_type: {index_type}. Use 'flat' or 'ivf'")
        
        # Metadata storage: list of dicts, one per vector
        self.metadata: List[Dict] = []
        
        print(f"Initialized FAISS vector store:")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Index type: {index_type}")
        if index_type == "ivf":
            print(f"  - Number of clusters: {nlist}")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add vectors and metadata to the index.
        
        Args:
            embeddings: Numpy array of shape (n, embedding_dim)
            metadata: List of metadata dicts, one per embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of metadata entries ({len(metadata)})"
            )
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings (required for cosine similarity with Inner Product)
        faiss.normalize_L2(embeddings)
        
        # Train index if using IVF (only needed once, before adding vectors)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} vectors to index. Total vectors: {self.index.ntotal}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query embeddings of shape (n_queries, embedding_dim) or (embedding_dim,)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) where:
            - distances: Array of shape (n_queries, k) with similarity scores
            - indices: Array of shape (n_queries, k) with indices of nearest neighbors
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors before searching.")
        
        # Handle single query
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Ensure float32 and normalize
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        # Search
        distances, indices = self.index.search(query_embeddings, min(k, self.index.ntotal))
        
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict]:
        """
        Get metadata for given indices.
        
        Args:
            indices: Array of indices (can be 1D or 2D)
            
        Returns:
            List of metadata dicts (flattened if indices is 2D)
        """
        indices_flat = indices.flatten()
        return [self.metadata[idx] for idx in indices_flat]
    
    def save(self, directory: str):
        """
        Save index and metadata to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'ntotal': self.index.ntotal
        }
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Saved vector store to {directory}")
        print(f"  - Index: {index_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Total vectors: {self.index.ntotal}")
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load index and metadata from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            FAISSVectorStore instance
        """
        # Load config
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_dim=config['embedding_dim'],
            index_type=config['index_type'],
            nlist=config.get('nlist', 100)
        )
        
        # Load index
        index_path = os.path.join(directory, "index.faiss")
        instance.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            instance.metadata = pickle.load(f)
        
        print(f"Loaded vector store from {directory}")
        print(f"  - Total vectors: {instance.index.ntotal}")
        
        return instance
    
    def get_size(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal

