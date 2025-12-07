"""
download datasets from huggingface
"""

import sys
import os

# add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DatasetLoader


def main():
    """download datasets"""
    print("=" * 60)
    print("Downloading Vietnamese Law Datasets")
    print("=" * 60)
    
    # Initialize loader
    loader = DatasetLoader(cache_dir='./dataset')
    
    # download qa dataset
    print("\n1. Downloading vn-law-questions-and-corpus...")
    try:
        qa_dataset = loader.load_questions_and_corpus()
        print(f"   Successfully loaded Q&A dataset")
        print(f"   Available splits: {list(qa_dataset.keys())}")
        if 'train' in qa_dataset:
            print(f"   Train size: {len(qa_dataset['train'])}")
    except Exception as e:
        print(f"   Error loading Q&A dataset: {e}")
    
    # download corpus dataset
    print("\n2. Downloading vn-law-corpus...")
    try:
        corpus_dataset = loader.load_corpus()
        print(f"   Successfully loaded corpus dataset")
        print(f"   Available splits: {list(corpus_dataset.keys())}")
        if 'train' in corpus_dataset:
            print(f"   Train size: {len(corpus_dataset['train'])}")
    except Exception as e:
        print(f"   Error loading corpus dataset: {e}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nDatasets are cached in ./dataset/")
    print("You can now run build_index.py to create the vector database.")


if __name__ == "__main__":
    main()

