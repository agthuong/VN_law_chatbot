"""
chat endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import json

from api.models.schemas import ChatRequest, ChatResponse, StreamChunk, SourceDocument
from api.dependencies import get_rag_chain

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """chat endpoint with rag"""
    try:
        rag_chain = get_rag_chain()
        
        # call rag chain
        result = rag_chain.invoke(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # format sources
        sources = [
            SourceDocument(
                id=doc.get("id", ""),
                text=doc.get("text", ""),
                source=doc.get("source", ""),
                score=doc.get("score"),
                rank=doc.get("rank"),
                chunk_index=doc.get("chunk_index")
            )
            for doc in result.get("sources", [])
        ]
        
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=sources,
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """streaming chat endpoint"""
    try:
        rag_chain = get_rag_chain()
        
        async def generate_stream() -> AsyncIterator[str]:
            """generate streaming response"""
            try:
                for chunk in rag_chain.stream(
                    question=request.question,
                    top_k=request.top_k,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    # format chunk as json
                    chunk_data = StreamChunk(
                        type=chunk.get("type", "answer"),
                        content=chunk.get("content", ""),
                        done=chunk.get("done", False),
                        metadata=chunk.get("metadata")
                    )
                    
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                    
                    if chunk.get("done", False):
                        break
                        
            except Exception as e:
                error_chunk = StreamChunk(
                    type="error",
                    content=f"Error: {str(e)}",
                    done=True
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming chat request: {str(e)}"
        )

