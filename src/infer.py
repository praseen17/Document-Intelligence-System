"""
Inference Script for Document Intelligence System
Classifies documents and extracts information
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import DocumentClassifier
from src.ocr import extract_text
from src.ner_kv import NERExtractor, KeyValueExtractor
from src.preprocessing import clean_text


def process_document(file_path: str, classifier: DocumentClassifier,
                    ner_extractor: NERExtractor, kv_extractor: KeyValueExtractor) -> Dict:
    """
    Process a single document.
    
    Args:
        file_path: Path to document file
        classifier: Trained classifier
        ner_extractor: NER extractor
        kv_extractor: Key-value extractor
        
    Returns:
        Dictionary with results
    """
    result = {
        'filename': os.path.basename(file_path),
        'filepath': file_path
    }
    
    # Extract text
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        text = extract_text(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            result['error'] = str(e)
            return result
    
    result['text'] = clean_text(text)
    result['text_length'] = len(text)
    
    # Classify document
    if classifier.model is not None:
        predictions = classifier.predict([text])
        probabilities = classifier.predict_proba([text])
        
        result['predicted_class'] = predictions[0]
        result['confidence'] = float(max(probabilities[0]))
        
        # Get probabilities for all classes
        if classifier.classes_ is not None:
            result['class_probabilities'] = {
                cls: float(prob) for cls, prob in zip(classifier.classes_, probabilities[0])
            }
    
    # Extract entities
    entities = ner_extractor.extract_entities(text)
    result['entities'] = entities
    
    # Extract key-value pairs
    kv_pairs = kv_extractor.extract_key_values(text)
    result['key_values'] = kv_pairs
    
    # Extract document-specific fields
    if result.get('predicted_class') == 'invoice':
        result['invoice_fields'] = kv_extractor.extract_invoice_fields(text)
    elif result.get('predicted_class') == 'resume':
        result['resume_fields'] = kv_extractor.extract_resume_fields(text)
    
    return result


def process_directory(directory: str, classifier: DocumentClassifier,
                     ner_extractor: NERExtractor, kv_extractor: KeyValueExtractor,
                     output_file: str = None) -> List[Dict]:
    """
    Process all documents in a directory.
    
    Args:
        directory: Directory path
        classifier: Trained classifier
        ner_extractor: NER extractor
        kv_extractor: Key-value extractor
        output_file: Optional JSON output file path
        
    Returns:
        List of results
    """
    results = []
    
    # Supported file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    text_extensions = ('.txt', '.pdf')  # PDF would need additional library
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.lower().endswith(image_extensions + text_extensions):
                print(f"Processing: {file_path}")
                result = process_document(file_path, classifier, ner_extractor, kv_extractor)
                results.append(result)
    
    # Save results to JSON if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Infer Document Types and Extract Information')
    parser.add_argument('input', type=str, help='Input file or directory')
    parser.add_argument('--model', type=str, default='models/document_classifier.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results')
    
    args = parser.parse_args()
    
    # Load classifier
    classifier = DocumentClassifier(model_path=args.model)
    if classifier.model is None:
        print("Warning: No model loaded. Classification will be skipped.")
        print("Train a model first using: python src/train.py")
    
    # Initialize extractors
    ner_extractor = NERExtractor()
    kv_extractor = KeyValueExtractor()
    
    # Process input
    if os.path.isfile(args.input):
        print(f"Processing file: {args.input}")
        result = process_document(args.input, classifier, ner_extractor, kv_extractor)
        results = [result]
    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        results = process_directory(args.input, classifier, ner_extractor, kv_extractor, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)
    
    # Print results
    if args.verbose:
        for result in results:
            print("\n" + "="*60)
            print(f"File: {result['filename']}")
            if 'predicted_class' in result:
                print(f"Predicted Class: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
            if result.get('entities'):
                print(f"Entities: {result['entities']}")
            if result.get('key_values'):
                print(f"Key-Value Pairs: {result['key_values']}")
            print("="*60)
    else:
        # Summary
        print("\n" + "="*60)
        print("Summary:")
        print(f"Total documents processed: {len(results)}")
        if any('predicted_class' in r for r in results):
            classes = [r.get('predicted_class', 'unknown') for r in results]
            from collections import Counter
            class_counts = Counter(classes)
            print("Predicted classes:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")
        print("="*60)


if __name__ == "__main__":
    main()

