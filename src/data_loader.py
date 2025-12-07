"""
load datasets from huggingface
"""

import os
from typing import List, Dict, Optional, Union
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

from src.utils import normalize_text, clean_text, chunk_text


class DatasetLoader:
    """load vn law datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """init loader, cache dir defaults to ./dataset"""
        self.cache_dir = cache_dir or './dataset'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_questions_and_corpus(self) -> Dataset:
        """load qa dataset"""
        print("Loading vn-law-questions-and-corpus dataset...")
        dataset = load_dataset(
            "truro7/vn-law-questions-and-corpus",
            cache_dir=self.cache_dir
        )
        return dataset
    
    def load_corpus(self) -> Dataset:
        """load corpus dataset"""
        print("Loading vn-law-corpus dataset...")
        dataset = load_dataset(
            "truro7/vn-law-corpus",
            cache_dir=self.cache_dir
        )
        return dataset
    
    def extract_corpus_texts(self, dataset: Dataset, split: str = "train") -> List[Dict[str, str]]:
        """extract texts from dataset"""
        texts = []
        
        # Handle dataset structure
        if hasattr(dataset, 'keys') and split in dataset:
            data = dataset[split]
        elif hasattr(dataset, '__iter__'):
            # If it's already a DatasetDict, try to get the split
            if split in dataset:
                data = dataset[split]
            else:
                # Use the first available split
                available_splits = list(dataset.keys()) if hasattr(dataset, 'keys') else []
                if available_splits:
                    print(f"Warning: Split '{split}' not found. Using '{available_splits[0]}' instead.")
                    data = dataset[available_splits[0]]
                else:
                    data = dataset
        else:
            data = dataset
        
        print(f"Extracting texts from dataset (split: {split})...")
        print(f"Dataset features: {data.features if hasattr(data, 'features') else 'N/A'}")
        print(f"Column names: {data.column_names if hasattr(data, 'column_names') else 'N/A'}")
        
        # Get first example to understand structure
        if len(data) > 0:
            first_example = data[0]
            print(f"Available fields in first example: {list(first_example.keys())}")
        
        # Handle different dataset structures
        empty_count = 0
        for idx, item in enumerate(tqdm(data, desc="Extracting texts")):
            text = None
            doc_id = None
            
            # try different field names for text
            text_fields = ['text', 'content', 'document', 'corpus', 'body', 'passage', 'paragraph']
            for field in text_fields:
                if field in item and item[field] is not None:
                    text = item[field]
                    break
            
            # if no text found, try any string field that's not an id
            if not text:
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 50 and key not in ['id', 'doc_id', 'document_id', 'corpus_id']:
                        text = value
                        print(f"Warning: Using field '{key}' as text for item {idx}")
                        break
            
            # get doc id
            id_fields = ['id', 'doc_id', 'document_id', 'corpus_id', '_id']
            for field in id_fields:
                if field in item and item[field] is not None:
                    doc_id = item[field]
                    break
            
            if not doc_id:
                doc_id = f"doc_{idx}"
            
            if text:
                # clean text
                text = clean_text(str(text))
                if text and len(text.strip()) > 10:
                    texts.append({
                        'id': str(doc_id),
                        'text': text,
                        'source': 'vn-law-corpus'
                    })
                else:
                    empty_count += 1
            else:
                empty_count += 1
        
        if empty_count > 0:
            print(f"Warning: {empty_count} items had no extractable text")
        
        print(f"Extracted {len(texts)} texts from corpus")
        return texts
    
    def extract_qa_pairs(self, dataset: Dataset, split: str = "train") -> List[Dict[str, str]]:
        """extract qa pairs from dataset"""
        qa_pairs = []
        
        # Handle dataset structure
        if hasattr(dataset, 'keys') and split in dataset:
            data = dataset[split]
        elif hasattr(dataset, '__iter__'):
            if split in dataset:
                data = dataset[split]
            else:
                available_splits = list(dataset.keys()) if hasattr(dataset, 'keys') else []
                if available_splits:
                    print(f"Warning: Split '{split}' not found. Using '{available_splits[0]}' instead.")
                    data = dataset[available_splits[0]]
                else:
                    data = dataset
        else:
            data = dataset
        
        print(f"Extracting Q&A pairs from dataset (split: {split})...")
        print(f"Dataset features: {data.features if hasattr(data, 'features') else 'N/A'}")
        print(f"Column names: {data.column_names if hasattr(data, 'column_names') else 'N/A'}")
        
        # Get first example to understand structure
        if len(data) > 0:
            first_example = data[0]
            print(f"Available fields in first example: {list(first_example.keys())}")
        
        missing_question = 0
        missing_answer = 0
        
        for idx, item in enumerate(tqdm(data, desc="Extracting Q&A pairs")):
            question = None
            answer = None
            doc_id = None
            
            # try different field names for question
            question_fields = ['question', 'q', 'query', 'questions', 'prompt']
            for field in question_fields:
                if field in item and item[field] is not None:
                    question = item[field]
                    break
            
            # try different field names for answer
            answer_fields = ['answer', 'a', 'answers', 'response', 'corpus', 'text', 'content']
            for field in answer_fields:
                if field in item and item[field] is not None:
                    answer = item[field]
                    break
            
            # get doc id
            id_fields = ['id', 'doc_id', 'document_id', 'question_id', 'qa_id', '_id']
            for field in id_fields:
                if field in item and item[field] is not None:
                    doc_id = item[field]
                    break
            
            if not doc_id:
                doc_id = f"qa_{idx}"
            
            if question and answer:
                question = clean_text(str(question))
                answer = clean_text(str(answer))
                
                if question and answer and len(question.strip()) > 5 and len(answer.strip()) > 10:
                    qa_pairs.append({
                        'id': str(doc_id),
                        'question': question,
                        'answer': answer,
                        'source': 'vn-law-questions-and-corpus'
                    })
                else:
                    if not question or len(question.strip()) <= 5:
                        missing_question += 1
                    if not answer or len(answer.strip()) <= 10:
                        missing_answer += 1
            else:
                if not question:
                    missing_question += 1
                if not answer:
                    missing_answer += 1
        
        if missing_question > 0 or missing_answer > 0:
            print(f"Warning: {missing_question} items missing question, {missing_answer} items missing answer")
        
        print(f"Extracted {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def prepare_corpus_for_indexing(
        self, 
        max_chunk_length: int = 512,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Union[str, int]]]:
        """prepare corpus for indexing, load both datasets and combine"""
        all_documents = []
        
        # load corpus dataset
        print("\n" + "="*60)
        print("Processing vn-law-corpus dataset")
        print("="*60)
        try:
            corpus_dataset = self.load_corpus()
            corpus_texts = self.extract_corpus_texts(corpus_dataset)
            
            if not corpus_texts:
                print("Warning: No texts extracted from corpus dataset")
            else:
                # process texts, chunk if needed
                chunked_count = 0
                for doc in corpus_texts:
                    text = doc['text']
                    doc_id = doc['id']
                    
                    # check if needs chunking (roughly 1 token = 4 chars)
                    char_limit = max_chunk_length * 4
                    if len(text) > char_limit:
                        chunks = chunk_text(text, char_limit, chunk_overlap * 4)
                        chunked_count += 1
                        for chunk_idx, chunk in enumerate(chunks):
                            all_documents.append({
                                'id': f"{doc_id}_chunk_{chunk_idx}",
                                'text': chunk,
                                'source': doc['source'],
                                'chunk_index': chunk_idx,
                                'original_id': doc_id
                            })
                    else:
                        all_documents.append({
                            'id': doc_id,
                            'text': text,
                            'source': doc['source'],
                            'chunk_index': 0,
                            'original_id': doc_id
                        })
                
                print(f"Processed {len(corpus_texts)} corpus documents")
                if chunked_count > 0:
                    print(f"  {chunked_count} documents were chunked")
        except Exception as e:
            print(f"Error loading corpus dataset: {e}")
            import traceback
            traceback.print_exc()
        
        # load qa dataset, use answers as corpus
        print("\n" + "="*60)
        print("Processing vn-law-questions-and-corpus dataset")
        print("="*60)
        try:
            qa_dataset = self.load_questions_and_corpus()
            qa_pairs = self.extract_qa_pairs(qa_dataset)
            
            if not qa_pairs:
                print("Warning: No Q&A pairs extracted from Q&A dataset")
            else:
                # add answers to corpus
                chunked_count = 0
                char_limit = max_chunk_length * 4
                for qa in qa_pairs:
                    answer = qa['answer']
                    doc_id = qa['id']
                    
                    if len(answer) > char_limit:
                        chunks = chunk_text(answer, char_limit, chunk_overlap * 4)
                        chunked_count += 1
                        for chunk_idx, chunk in enumerate(chunks):
                            all_documents.append({
                                'id': f"{doc_id}_answer_chunk_{chunk_idx}",
                                'text': chunk,
                                'source': 'vn-law-questions-and-corpus',
                                'chunk_index': chunk_idx,
                                'original_id': doc_id
                            })
                    else:
                        all_documents.append({
                            'id': f"{doc_id}_answer",
                            'text': answer,
                            'source': 'vn-law-questions-and-corpus',
                            'chunk_index': 0,
                            'original_id': doc_id
                        })
                
                print(f"Processed {len(qa_pairs)} Q&A pairs (using answers as corpus)")
                if chunked_count > 0:
                    print(f"  {chunked_count} answers were chunked")
        except Exception as e:
            print(f"Error loading Q&A dataset: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print(f"Total documents prepared for indexing: {len(all_documents)}")
        print("="*60)
        
        if len(all_documents) == 0:
            print("\nWARNING: No documents were prepared for indexing!")
            print("Please check:")
            print("  1. Datasets are downloaded (run download_datasets.py)")
            print("  2. Dataset structure matches expected format")
            print("  3. Run inspect_datasets.py to check dataset structure")
        
        return all_documents

