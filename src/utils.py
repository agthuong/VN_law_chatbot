"""
Utility functions for VN Law RAG system
"""

import re
from typing import List, Optional


def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text by removing extra whitespace and special characters.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Split long text into chunks with overlap.
    
    Args:
        text: Input text to chunk
        max_length: Maximum length of each chunk (in characters, approximate)
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the end
            for i in range(end, max(start + max_length - 100, start), -1):
                if text[i] in ['.', '!', '?', '\n']:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters while preserving Vietnamese characters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = normalize_text(text)
    
    return text

