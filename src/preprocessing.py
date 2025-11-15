"""
Preprocessing Module for Document Intelligence System
Text cleaning and normalization utilities
"""

import re
import string
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\:\;\!\?\-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text to lowercase and remove extra spaces.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_lines(text: str) -> List[str]:
    """
    Split text into lines and clean each line.
    
    Args:
        text: Input text
        
    Returns:
        List of cleaned lines
    """
    lines = text.split('\n')
    cleaned_lines = [clean_text(line) for line in lines if clean_text(line)]
    return cleaned_lines


def extract_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = [clean_text(p) for p in paragraphs if clean_text(p)]
    return cleaned_paragraphs


def remove_stopwords_custom(text: str, custom_stopwords: List[str] = None) -> str:
    """
    Remove custom stopwords from text.
    
    Args:
        text: Input text
        custom_stopwords: List of stopwords to remove
        
    Returns:
        Text with stopwords removed
    """
    if custom_stopwords is None:
        custom_stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
    
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in custom_stopwords]
    return ' '.join(filtered_words)


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    # Pattern for currency, decimals, integers
    pattern = r'[\$€£]?\s*\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # Remove currency symbols and convert
            num_str = re.sub(r'[\$€£\s]', '', match)
            numbers.append(float(num_str))
        except ValueError:
            continue
    
    return numbers


def extract_dates(text: str) -> List[str]:
    """
    Extract date-like patterns from text.
    
    Args:
        text: Input text
        
    Returns:
        List of date strings
    """
    # Common date patterns
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
        r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}', # January 1, 2024
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Input text
        
    Returns:
        List of email addresses
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails


def extract_phones(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of phone numbers
    """
    # Various phone number patterns
    patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Simple format
    ]
    
    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    return phones


if __name__ == "__main__":
    # Test preprocessing functions
    sample_text = """
    Invoice #12345
    Date: 01/15/2024
    Contact: john.doe@example.com
    Phone: (555) 123-4567
    Total: $1,234.56
    """
    
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
    print("Numbers:", extract_numbers(sample_text))
    print("Dates:", extract_dates(sample_text))
    print("Emails:", extract_emails(sample_text))
    print("Phones:", extract_phones(sample_text))

