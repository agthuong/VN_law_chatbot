"""
Script to build FAISS index from Vietnamese law datasets
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DatasetLoader
from src.retrieval import RetrievalSystem


def main():
    """Build FAISS index from datasets"""
    parser = argparse.ArgumentParser(description='Build FAISS index from Vietnamese law datasets')
    parser.add_argument(
        '--index-type',
        type=str,
        default='flat',
        choices=['flat', 'ivf'],
        help='Type of FAISS index (flat for exact search, ivf for approximate search)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding embeddings'
    )
    parser.add_argument(
        '--max-chunk-length',
        type=int,
        default=512,
        help='Maximum chunk length (approximate characters)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./vector_db',
        help='Output directory for vector store'
    )
    parser.add_argument(
        '--model-cache-dir',
        type=str,
        default='./embedding_model',
        help='Directory to cache embedding model'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building FAISS Index for Vietnamese Law Documents")
    print("=" * 60)
    print(f"Index type: {args.index_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max chunk length: {args.max_chunk_length}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Load datasets
    print("Step 1: Loading datasets...")
    loader = DatasetLoader(cache_dir='./dataset')
    documents = loader.prepare_corpus_for_indexing(
        max_chunk_length=args.max_chunk_length,
        chunk_overlap=50
    )
    
    if not documents:
        print("Error: No documents loaded. Please run download_datasets.py first.")
        return
    
    print(f"Loaded {len(documents)} documents")
    print()
    
    # init retrieval system
    print("Step 2: Initializing retrieval system...")
    retrieval = RetrievalSystem(
        model_name="anhtld/VN-Law-Embedding",
        model_cache_dir=args.model_cache_dir
    )
    
    # recreate vector store if needed
    if args.index_type == 'ivf':
        embedding_dim = retrieval.embedder.get_embedding_dimension()
        from src.vector_store import FAISSVectorStore
        retrieval.vector_store = FAISSVectorStore(
            embedding_dim=embedding_dim,
            index_type='ivf',
            nlist=100
        )
    
    print("Retrieval system initialized")
    print()
    
    # Step 3: Add documents to vector store
    print("Step 3: Creating embeddings and building index...")
    retrieval.add_documents(documents, batch_size=args.batch_size)
    print()
    
    # Step 4: Save vector store
    print("Step 4: Saving vector store...")
    os.makedirs(args.output_dir, exist_ok=True)
    retrieval.save(args.output_dir)
    print()
    
    # Step 5: Display statistics
    print("Step 5: Statistics")
    stats = retrieval.get_stats()
    print(f"  Total documents: {stats['num_documents']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Max sequence length: {stats['max_seq_length']}")
    print()
    
    print("=" * 60)
    print("Index building complete!")
    print("=" * 60)
    print(f"\nVector store saved to: {args.output_dir}")



if __name__ == "__main__":
    main()

