"""
Named Entity Recognition and Key-Value Extraction Module
Extracts structured information from documents using spaCy
"""

import os
import spacy
from typing import List, Dict, Tuple
import re


class NERExtractor:
    """
    Named Entity Recognition extractor using spaCy.
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize NER extractor.
        
        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: {model_name} not found. Install with: python -m spacy download {model_name}")
            print("Falling back to basic regex extraction.")
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if self.nlp is None:
            return self._extract_entities_regex(text)
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(ent.text)
        
        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    
    def _extract_entities_regex(self, text: str) -> Dict[str, List[str]]:
        """
        Fallback regex-based entity extraction.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'MONEY': [],
            'EMAIL': [],
            'PHONE': []
        }
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities['EMAIL'] = emails
        
        # Extract phone numbers
        phones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        entities['PHONE'] = phones
        
        # Extract dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', text)
        entities['DATE'] = dates
        
        # Extract money
        money = re.findall(r'[\$€£]\s*\d+\.?\d*', text)
        entities['MONEY'] = money
        
        return {k: v for k, v in entities.items() if v}


class KeyValueExtractor:
    """
    Key-Value pair extractor for structured documents.
    """
    
    def __init__(self):
        """Initialize key-value extractor."""
        self.patterns = {
            'invoice_number': [r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)', r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)'],
            'date': [r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'dated?\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'],
            'total': [r'total\s*:?\s*[\$€£]?\s*(\d+\.?\d*)', r'amount\s*:?\s*[\$€£]?\s*(\d+\.?\d*)'],
            'due_date': [r'due\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'],
            'vendor': [r'vendor\s*:?\s*([A-Za-z\s&]+)', r'from\s*:?\s*([A-Za-z\s&]+)'],
            'customer': [r'customer\s*:?\s*([A-Za-z\s&]+)', r'to\s*:?\s*([A-Za-z\s&]+)'],
            'email': [r'[\w\.-]+@[\w\.-]+\.\w+'],
            'phone': [r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'],
        }
    
    def extract_key_values(self, text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of key-value pairs
        """
        text_lower = text.lower()
        kv_pairs = {}
        
        for key, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    kv_pairs[key] = matches[0].strip()
                    break
        
        return kv_pairs
    
    def extract_invoice_fields(self, text: str) -> Dict[str, any]:
        """
        Extract common invoice fields.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of invoice fields
        """
        fields = self.extract_key_values(text)
        
        # Extract line items (basic pattern)
        line_items = []
        lines = text.split('\n')
        for line in lines:
            # Look for patterns like "Item $XX.XX" or "Description Amount"
            if re.search(r'[\$€£]\s*\d+\.?\d*', line, re.IGNORECASE):
                line_items.append(line.strip())
        
        if line_items:
            fields['line_items'] = line_items[:10]  # Limit to 10 items
        
        return fields
    
    def extract_resume_fields(self, text: str) -> Dict[str, any]:
        """
        Extract common resume fields.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of resume fields
        """
        fields = {}
        
        # Extract name (usually first line or after "Name:")
        lines = text.split('\n')
        if lines:
            fields['name'] = lines[0].strip()
        
        # Extract email
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            fields['email'] = emails[0]
        
        # Extract phone
        phones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phones:
            fields['phone'] = phones[0]
        
        # Extract sections
        sections = {
            'experience': [],
            'education': [],
            'skills': []
        }
        
        current_section = None
        for line in lines:
            line_lower = line.lower().strip()
            if 'experience' in line_lower or 'work' in line_lower:
                current_section = 'experience'
            elif 'education' in line_lower:
                current_section = 'education'
            elif 'skill' in line_lower:
                current_section = 'skills'
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        
        fields.update(sections)
        
        return fields


if __name__ == "__main__":
    # Test NER and KV extraction
    sample_text = """
    Invoice #INV-12345
    Date: 01/15/2024
    Vendor: ABC Company
    Customer: XYZ Corp
    Email: contact@abc.com
    Phone: (555) 123-4567
    Total: $1,234.56
    Due Date: 02/15/2024
    """
    
    ner = NERExtractor()
    entities = ner.extract_entities(sample_text)
    print("Entities:", entities)
    
    kv = KeyValueExtractor()
    kv_pairs = kv.extract_invoice_fields(sample_text)
    print("Key-Value pairs:", kv_pairs)

