"""
health check endpoints
"""

from fastapi import APIRouter, HTTPException
from api.models.schemas import HealthResponse, VectorStoreHealthResponse
from api.dependencies import get_rag_chain

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """health check endpoint"""
    try:
        rag_chain = get_rag_chain()
        stats = rag_chain.get_stats()
        
        # check ollama
        ollama_status = stats.get("ollama", {})
        ollama_healthy = ollama_status.get("status") == "healthy"
        
        # check vector store
        retrieval_stats = stats.get("retrieval", {})
        vector_store_healthy = retrieval_stats.get("num_documents", 0) > 0
        
        # overall status
        overall_healthy = ollama_healthy and vector_store_healthy
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            ollama=ollama_status,
            vector_store={
                "status": "healthy" if vector_store_healthy else "unhealthy",
                "num_documents": retrieval_stats.get("num_documents", 0),
                "embedding_dim": retrieval_stats.get("embedding_dim", 0)
            },
            retrieval=retrieval_stats
        )
    except Exception as e:
        # return unhealthy status instead of raising exception
        return HealthResponse(
            status="unhealthy",
            ollama={"status": "error", "error": str(e)},
            vector_store={
                "status": "error",
                "num_documents": 0,
                "embedding_dim": 0
            },
            retrieval=None
        )


@router.get("/vector-store", response_model=VectorStoreHealthResponse)
async def vector_store_health():
    """vector store health check"""
    try:
        rag_chain = get_rag_chain()
        stats = rag_chain.get_stats()
        retrieval_stats = stats.get("retrieval", {})
        
        num_docs = retrieval_stats.get("num_documents", 0)
        is_healthy = num_docs > 0
        
        return VectorStoreHealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            num_documents=num_docs,
            embedding_dim=retrieval_stats.get("embedding_dim", 0),
            max_seq_length=retrieval_stats.get("max_seq_length", 512)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store health check failed: {str(e)}"
        )

