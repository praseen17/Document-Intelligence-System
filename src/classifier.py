"""
Classification Module for Document Intelligence System
Document type classification using TF-IDF and RandomForest
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .features import extract_tfidf_features, extract_statistical_features, extract_document_type_keywords, extract_structured_features, combine_features
from .preprocessing import clean_text
from typing import List, Tuple


class DocumentClassifier:
    """
    Classifier for document type classification.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
        self.vectorizer = None
        self.model_path = model_path
        self.classes_ = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def extract_features(self, texts: List[str], fit_vectorizer: bool = True):
        """
        Extract features from texts.
        
        Args:
            texts: List of text documents
            fit_vectorizer: Whether to fit the vectorizer (True for training)
            
        Returns:
            Feature matrix
        """
        # Clean texts
        cleaned_texts = [clean_text(text) for text in texts]
        
        # TF-IDF features
        if fit_vectorizer or self.vectorizer is None:
            tfidf_features, self.vectorizer = extract_tfidf_features(cleaned_texts)
        else:
            tfidf_features = self.vectorizer.transform(cleaned_texts).toarray()
        
        # Additional features
        additional_features = []
        for text in cleaned_texts:
            stat = extract_statistical_features(text)
            keywords = extract_document_type_keywords(text)
            structured = extract_structured_features(text)
            
            # Combine into single vector
            combined = np.concatenate([
                np.array(list(stat.values())),
                np.array(list(keywords.values())),
                np.array(list(structured.values()))
            ])
            additional_features.append(combined)
        
        additional_features = np.array(additional_features)
        
        # Combine TF-IDF and additional features
        all_features = np.hstack([tfidf_features, additional_features])
        
        return all_features
    
    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2):
        """
        Train the classifier.
        
        Args:
            texts: List of training texts
            labels: List of corresponding labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Training accuracy and test accuracy
        """
        # Extract features
        X = self.extract_features(texts, fit_vectorizer=True)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        return train_acc, test_acc
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict document types for given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted labels
        """
        X = self.extract_features(texts, fit_vectorizer=False)
        predictions = self.model.predict(X)
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of prediction probabilities
        """
        X = self.extract_features(texts, fit_vectorizer=False)
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def save(self, model_path: str):
        """
        Save the model and vectorizer.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load(self, model_path: str):
        """
        Load a pre-trained model and vectorizer.
        
        Args:
            model_path: Path to the model file
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
        # Load vectorizer
        vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
        
        if hasattr(self.model, 'classes_'):
            self.classes_ = self.model.classes_


if __name__ == "__main__":
    # Example usage
    texts = [
        "Invoice #12345 Date: 01/15/2024 Total: $100.00",
        "Receipt Thank you for your purchase Date: 01/15/2024",
        "John Doe Resume Experience: Software Engineer Education: BS Computer Science",
        "Contract Agreement between Party A and Party B Effective Date: 01/01/2024"
    ]
    labels = ["invoice", "receipt", "resume", "contract"]
    
    classifier = DocumentClassifier()
    classifier.train(texts, labels)
    predictions = classifier.predict(texts)
    print("Predictions:", predictions)

