"""
embedding model wrapper
"""

import os
from typing import List, Union, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class VNLawEmbedder:
    """wrapper for embedding model"""
    
    def __init__(
        self, 
        model_name: str = "anhtld/VN-Law-Embedding",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """init embedding model"""
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir or './embedding_model'
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading embedding model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(
            model_name,
            cache_folder=self.model_cache_dir,
            device=self.device
        )
        
        # Model properties
        self.max_seq_length = self.model.max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Model loaded successfully!")
        print(f"  - Max sequence length: {self.max_seq_length}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings (recommended for cosine similarity)
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy
        )
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts with progress tracking.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
    
    def get_max_seq_length(self) -> int:
        """Get the maximum sequence length."""
        return self.max_seq_length

