"""
api dependencies
"""

from typing import Optional
from functools import lru_cache

from build.rag_chain import VNLawRAGChain
from build.config import RAGConfig
from src.retrieval import RetrievalSystem


# global rag chain instance (singleton)
_rag_chain_instance: Optional[VNLawRAGChain] = None


def get_rag_chain() -> VNLawRAGChain:
    """get or create rag chain instance"""
    global _rag_chain_instance
    
    if _rag_chain_instance is None:
        # validate paths
        RAGConfig.validate_paths()
        
        # load retrieval system
        retrieval_system = RetrievalSystem.load(
            directory=RAGConfig.VECTOR_STORE_DIR,
            model_cache_dir=RAGConfig.EMBEDDING_MODEL_CACHE_DIR,
            model_name=RAGConfig.EMBEDDING_MODEL_NAME
        )
        
        # create rag chain
        _rag_chain_instance = VNLawRAGChain(
            retrieval_system=retrieval_system,
            top_k=RAGConfig.TOP_K,
            temperature=RAGConfig.TEMPERATURE
        )
    
    return _rag_chain_instance


def reset_rag_chain():
    """reset rag chain instance (for testing)"""
    global _rag_chain_instance
    _rag_chain_instance = None

