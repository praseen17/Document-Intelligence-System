"""
Training Script for Document Intelligence System
Trains the document classifier on labeled data
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import DocumentClassifier
from src.ocr import extract_text
from src.visualize import plot_confusion_matrix, plot_class_distribution
from src.preprocessing import clean_text


def load_data(csv_path: str, examples_dir: str = None):
    """
    Load training data from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: filename, label
        examples_dir: Directory containing example images/documents
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        filename = row.get('filename', row.get('file', ''))
        label = row.get('label', row.get('type', ''))
        
        if not filename or not label:
            continue
        
        # If examples_dir is provided, try to read the file
        if examples_dir:
            file_path = os.path.join(examples_dir, filename)
            if os.path.exists(file_path):
                # Try OCR if it's an image
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    text = extract_text(file_path)
                else:
                    # Read as text file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
            else:
                # Use filename or other columns as text
                text = row.get('text', filename)
        else:
            # Use text column if available
            text = row.get('text', filename)
        
        if text:
            texts.append(clean_text(text))
            labels.append(label)
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description='Train Document Intelligence Classifier')
    parser.add_argument('--data', type=str, default='data/labels/train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--examples', type=str, default='data/examples',
                       help='Path to examples directory')
    parser.add_argument('--model', type=str, default='models/document_classifier.pkl',
                       help='Path to save the model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.model) if os.path.dirname(args.model) else 'models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("Loading training data...")
    texts, labels = load_data(args.data, args.examples)
    
    if len(texts) == 0:
        print("Error: No training data found!")
        sys.exit(1)
    
    print(f"Loaded {len(texts)} training samples")
    print(f"Classes: {set(labels)}")
    
    # Plot class distribution
    plot_class_distribution(labels)
    
    # Initialize and train classifier
    print("\nTraining classifier...")
    classifier = DocumentClassifier()
    train_acc, test_acc = classifier.train(texts, labels, test_size=args.test_size)
    
    # Save model
    print(f"\nSaving model to {args.model}...")
    classifier.save(args.model)
    
    # Generate predictions for confusion matrix
    predictions = classifier.predict(texts)
    plot_confusion_matrix(labels, predictions)
    
    print("\nTraining complete!")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

