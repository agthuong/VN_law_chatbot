"""
fastapi main app
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import chat, health
from build.config import RAGConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    """startup/shutdown stuff"""
    # startup
    print("=" * 60)
    print("Starting VN Law RAG API")
    print("=" * 60)
    print(f"Ollama URL: {RAGConfig.OLLAMA_BASE_URL}")
    print(f"Ollama Model: {RAGConfig.OLLAMA_MODEL}")
    print(f"Vector Store: {RAGConfig.VECTOR_STORE_DIR}")
    print("=" * 60)
    
    # check config
    try:
        RAGConfig.validate_paths()
        print("Vector store validated")
    except Exception as e:
        print(f"Warning: {e}")
    
    # check ollama
    try:
        from api.dependencies import get_rag_chain
        rag_chain = get_rag_chain()
        ollama_health = rag_chain.ollama_client.check_health()
        if ollama_health.get("status") == "healthy":
            print("Ollama is healthy")
        else:
            print(f"Ollama health check: {ollama_health.get('status')}")
    except Exception as e:
        print(f"Could not check Ollama: {e}")
    
    print("=" * 60)
    print("API started successfully!")
    print("=" * 60)
    
    yield
    
    # shutdown
    print("\nShutting down API...")


# create app
app = FastAPI(
    title="VN Law RAG API",
    description="REST API for Vietnamese Law RAG system using Ollama and LangChain",
    version="0.1.0",
    lifespan=lifespan
)

# cors stuff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# add routes
app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def root():
    """root endpoint"""
    return {
        "message": "VN Law RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=RAGConfig.API_HOST,
        port=RAGConfig.API_PORT,
        reload=RAGConfig.API_RELOAD
    )

