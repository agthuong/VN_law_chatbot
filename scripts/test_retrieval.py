"""
test retrieval system
"""

import sys
import os
import argparse

# add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import RetrievalSystem


def main():
    """test retrieval"""
    parser = argparse.ArgumentParser(description='Test retrieval system')
    parser.add_argument(
        '--vector-db-dir',
        type=str,
        default='./vector_db',
        help='Directory containing vector store'
    )
    parser.add_argument(
        '--model-cache-dir',
        type=str,
        default='./embedding_model',
        help='Directory containing embedding model'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of documents to retrieve'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Query text to test (if not provided, uses sample queries)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Retrieval System")
    print("=" * 60)
    print(f"Vector DB directory: {args.vector_db_dir}")
    print(f"Top-K: {args.k}")
    print()
    
    # load retrieval system
    print("Loading retrieval system...")
    try:
        retrieval = RetrievalSystem.load(
            directory=args.vector_db_dir,
            model_cache_dir=args.model_cache_dir
        )
        print("Retrieval system loaded")
    except Exception as e:
        print(f"Error loading retrieval system: {e}")
        print("\nPlease make sure you have:")
        print("  1. Run download_datasets.py to download datasets")
        print("  2. Run build_index.py to build the vector index")
        return
    
    stats = retrieval.get_stats()
    print(f"  Total documents: {stats['num_documents']}")
    print()
    
    # Sample queries
    sample_queries = [
        "Người sử dụng lao động có phải thiết lập cơ chế đối thoại với người lao động không?",
        "Trọng tài viên lao động cần đào tạo như thế nào?",
        "Quy định về thời gian làm việc của người lao động",
        "Xử lý vi phạm trong hợp đồng lao động",
        "Quyền và nghĩa vụ của người lao động"
    ]
    
    # Use provided query or sample queries
    if args.query:
        queries = [args.query]
    else:
        queries = sample_queries
        print("Using sample queries...")
        print()
    
    # Test retrieval
    print("=" * 60)
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        
        results = retrieval.retrieve(query, k=args.k, return_scores=True)
        
        if not results:
            print("No results found.")
            continue
        
        for result in results:
            print(f"\n[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"ID: {result['id']}")
            print(f"Source: {result['source']}")
            if result.get('chunk_index', 0) > 0:
                print(f"Chunk: {result['chunk_index']}")
            print(f"Text preview: {result['text'][:200]}...")
        
        print()
    
    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

