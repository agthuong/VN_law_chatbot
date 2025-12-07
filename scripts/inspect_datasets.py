"""
inspect dataset structure
"""

import sys
import os

# add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from src.data_loader import DatasetLoader


def inspect_dataset_structure(dataset_name: str, cache_dir: str = './dataset'):
    """inspect dataset structure"""
    print("=" * 80)
    print(f"Inspecting dataset: {dataset_name}")
    print("=" * 80)
    
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        print(f"\nDataset type: {type(dataset)}")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Inspect each split
        for split_name in dataset.keys():
            print(f"\n{'='*80}")
            print(f"Split: {split_name}")
            print(f"{'='*80}")
            
            split_data = dataset[split_name]
            print(f"Number of examples: {len(split_data)}")
            print(f"Features: {split_data.features}")
            print(f"Column names: {split_data.column_names}")
            
            # Show first few examples
            if len(split_data) > 0:
                print(f"\nFirst example:")
                first_example = split_data[0]
                for key, value in first_example.items():
                    if isinstance(value, str):
                        preview = value[:200] + "..." if len(value) > 200 else value
                        print(f"  {key}: {preview}")
                    else:
                        print(f"  {key}: {value}")
                
                # Show data types
                print(f"\nData types:")
                for key, value in first_example.items():
                    print(f"  {key}: {type(value).__name__}")
                
                # Show a few more examples to check consistency
                if len(split_data) > 1:
                    print(f"\nChecking consistency across examples...")
                    all_keys = set(first_example.keys())
                    for i in range(1, min(5, len(split_data))):
                        example = split_data[i]
                        example_keys = set(example.keys())
                        if example_keys != all_keys:
                            print(f"  Warning: Example {i} has different keys!")
                            print(f"    Expected: {all_keys}")
                            print(f"    Got: {example_keys}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_extraction():
    """Test if our data extraction methods work correctly"""
    print("\n" + "=" * 80)
    print("Testing Data Extraction Methods")
    print("=" * 80)
    
    loader = DatasetLoader(cache_dir='./dataset')
    
    # test corpus extraction
    print("\n1. Testing corpus extraction...")
    try:
        corpus_dataset = loader.load_corpus()
        corpus_texts = loader.extract_corpus_texts(corpus_dataset)
        print(f"   Extracted {len(corpus_texts)} texts from corpus")
        if corpus_texts:
            print(f"   Sample text (first 200 chars): {corpus_texts[0]['text'][:200]}...")
            print(f"   Sample ID: {corpus_texts[0]['id']}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # test qa extraction
    print("\n2. Testing Q&A extraction...")
    try:
        qa_dataset = loader.load_questions_and_corpus()
        qa_pairs = loader.extract_qa_pairs(qa_dataset)
        print(f"   Extracted {len(qa_pairs)} Q&A pairs")
        if qa_pairs:
            print(f"   Sample question: {qa_pairs[0]['question'][:100]}...")
            print(f"   Sample answer (first 200 chars): {qa_pairs[0]['answer'][:200]}...")
            print(f"   Sample ID: {qa_pairs[0]['id']}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # test full prep
    print("\n3. Testing full corpus preparation...")
    try:
        documents = loader.prepare_corpus_for_indexing(max_chunk_length=512)
        print(f"   Prepared {len(documents)} documents for indexing")
        if documents:
            print(f"   Sample document:")
            print(f"     ID: {documents[0]['id']}")
            print(f"     Source: {documents[0]['source']}")
            print(f"     Text (first 200 chars): {documents[0]['text'][:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main inspection function"""
    print("Vietnamese Law Datasets Inspection Tool")
    print("=" * 80)
    
    # Inspect vn-law-corpus
    print("\n")
    corpus_dataset = inspect_dataset_structure("truro7/vn-law-corpus")
    
    # Inspect vn-law-questions-and-corpus
    print("\n")
    qa_dataset = inspect_dataset_structure("truro7/vn-law-questions-and-corpus")
    
    # Test our extraction methods
    test_data_extraction()
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

