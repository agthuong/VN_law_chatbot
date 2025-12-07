"""
Pydantic schemas for API requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    
    question: str = Field(..., description="User question about Vietnamese law")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Người sử dụng lao động có phải thiết lập cơ chế đối thoại với người lao động không?",
                "top_k": 5,
                "temperature": 0.7
            }
        }


class SourceDocument(BaseModel):
    """Source document schema"""
    
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    source: str = Field(..., description="Document source")
    score: Optional[float] = Field(default=None, description="Similarity score")
    rank: Optional[int] = Field(default=None, description="Rank in retrieval results")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index if document was chunked")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Retrieved source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Có, theo quy định của Bộ luật Lao động 2019...",
                "sources": [
                    {
                        "id": "doc_123",
                        "text": "Nội dung văn bản pháp luật...",
                        "source": "vn-law-corpus",
                        "score": 0.85,
                        "rank": 1
                    }
                ],
                "metadata": {
                    "retrieved_count": 5,
                    "top_k": 5,
                    "temperature": 0.7
                }
            }
        }


class StreamChunk(BaseModel):
    """Stream chunk schema"""
    
    type: str = Field(..., description="Chunk type: 'sources', 'answer', or 'error'")
    content: Any = Field(..., description="Chunk content")
    done: Optional[bool] = Field(default=False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    ollama: Dict[str, Any] = Field(..., description="Ollama health status")
    vector_store: Dict[str, Any] = Field(..., description="Vector store health status")
    retrieval: Optional[Dict[str, Any]] = Field(default=None, description="Retrieval system stats")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ollama": {
                    "status": "healthy",
                    "ollama_running": True,
                    "model_available": True,
                    "model_name": "qwen2.5:30b"
                },
                "vector_store": {
                    "status": "healthy",
                    "num_documents": 15000,
                    "embedding_dim": 768
                }
            }
        }


class VectorStoreHealthResponse(BaseModel):
    """Vector store health check response"""
    
    status: str = Field(..., description="Status: 'healthy' or 'unhealthy'")
    num_documents: int = Field(..., description="Number of documents in vector store")
    embedding_dim: int = Field(..., description="Embedding dimension")
    max_seq_length: int = Field(..., description="Maximum sequence length")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "num_documents": 15000,
                "embedding_dim": 768,
                "max_seq_length": 512
            }
        }

