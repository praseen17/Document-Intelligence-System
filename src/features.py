"""
Feature Extraction Module for Document Intelligence System
Extracts features from text for classification and analysis
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import re
import string


def extract_statistical_features(text: str) -> Dict[str, float]:
    """
    Extract statistical features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of statistical features
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'digit_count': 0,
            'uppercase_ratio': 0,
            'punctuation_count': 0,
            'line_count': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lines = text.split('\n')
    
    features = {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'digit_count': sum(1 for c in text if c.isdigit()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'punctuation_count': sum(1 for c in text if c in string.punctuation),
        'line_count': len([l for l in lines if l.strip()])
    }
    
    return features


def extract_keyword_features(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Count occurrences of specific keywords.
    
    Args:
        text: Input text
        keywords: List of keywords to search for
        
    Returns:
        Dictionary of keyword counts
    """
    text_lower = text.lower()
    features = {}
    
    for keyword in keywords:
        features[f'has_{keyword}'] = 1 if keyword.lower() in text_lower else 0
        features[f'count_{keyword}'] = text_lower.count(keyword.lower())
    
    return features


def extract_tfidf_features(texts: List[str], max_features: int = 1000) -> np.ndarray:
    """
    Extract TF-IDF features from a list of texts.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features
        
    Returns:
        TF-IDF feature matrix
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer


def extract_document_type_keywords(text: str) -> Dict[str, int]:
    """
    Extract keywords that indicate document type.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of document type indicators
    """
    text_lower = text.lower()
    
    invoice_keywords = ['invoice', 'bill', 'total', 'amount due', 'payment', 'invoice number']
    receipt_keywords = ['receipt', 'thank you', 'purchase', 'transaction', 'date']
    resume_keywords = ['resume', 'cv', 'experience', 'education', 'skills', 'objective']
    contract_keywords = ['contract', 'agreement', 'terms', 'party', 'signature', 'effective date']
    
    features = {
        'invoice_score': sum(1 for kw in invoice_keywords if kw in text_lower),
        'receipt_score': sum(1 for kw in receipt_keywords if kw in text_lower),
        'resume_score': sum(1 for kw in resume_keywords if kw in text_lower),
        'contract_score': sum(1 for kw in contract_keywords if kw in text_lower),
    }
    
    return features


def extract_structured_features(text: str) -> Dict[str, any]:
    """
    Extract structured information from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of structured features
    """
    from .preprocessing import extract_numbers, extract_dates, extract_emails, extract_phones
    
    numbers = extract_numbers(text)
    dates = extract_dates(text)
    emails = extract_emails(text)
    phones = extract_phones(text)
    
    features = {
        'has_numbers': 1 if numbers else 0,
        'number_count': len(numbers),
        'max_number': max(numbers) if numbers else 0,
        'min_number': min(numbers) if numbers else 0,
        'has_dates': 1 if dates else 0,
        'date_count': len(dates),
        'has_emails': 1 if emails else 0,
        'email_count': len(emails),
        'has_phones': 1 if phones else 0,
        'phone_count': len(phones),
    }
    
    return features


def combine_features(statistical: Dict, keywords: Dict, structured: Dict) -> np.ndarray:
    """
    Combine all feature dictionaries into a single feature vector.
    
    Args:
        statistical: Statistical features
        keywords: Keyword features
        structured: Structured features
        
    Returns:
        Combined feature vector as numpy array
    """
    all_features = {**statistical, **keywords, **structured}
    feature_vector = np.array(list(all_features.values()))
    return feature_vector


if __name__ == "__main__":
    # Test feature extraction
    sample_text = """
    Invoice #12345
    Date: 01/15/2024
    Contact: john.doe@example.com
    Phone: (555) 123-4567
    Total: $1,234.56
    """
    
    stat_features = extract_statistical_features(sample_text)
    doc_features = extract_document_type_keywords(sample_text)
    struct_features = extract_structured_features(sample_text)
    
    print("Statistical features:", stat_features)
    print("Document type features:", doc_features)
    print("Structured features:", struct_features)

