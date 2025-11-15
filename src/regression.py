"""
Regression Module for Document Intelligence System
Predicts numerical values from documents (e.g., invoice amounts, dates)
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .features import extract_statistical_features, extract_document_type_keywords, extract_structured_features
from .preprocessing import extract_numbers, clean_text
from typing import List, Tuple


class DocumentRegressor:
    """
    Regressor for predicting numerical values from documents.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the regressor.
        
        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20)
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Feature matrix
        """
        features = []
        
        for text in texts:
            cleaned = clean_text(text)
            
            # Statistical features
            stat = extract_statistical_features(cleaned)
            
            # Document type keywords
            keywords = extract_document_type_keywords(cleaned)
            
            # Structured features
            structured = extract_structured_features(cleaned)
            
            # Combine features
            feature_vector = np.concatenate([
                np.array(list(stat.values())),
                np.array(list(keywords.values())),
                np.array(list(structured.values()))
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, texts: List[str], targets: List[float], test_size: float = 0.2):
        """
        Train the regressor.
        
        Args:
            texts: List of training texts
            targets: List of target values
            test_size: Proportion of data to use for testing
            
        Returns:
            Training and test metrics
        """
        # Extract features
        X = self.extract_features(texts)
        y = np.array(targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict target values for given texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of predictions
        """
        X = self.extract_features(texts)
        predictions = self.model.predict(X)
        return predictions
    
    def save(self, model_path: str):
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path


if __name__ == "__main__":
    # Example usage for predicting invoice amounts
    texts = [
        "Invoice #12345 Date: 01/15/2024 Total: $100.00",
        "Invoice #12346 Date: 01/16/2024 Total: $250.50",
        "Invoice #12347 Date: 01/17/2024 Total: $75.25",
        "Invoice #12348 Date: 01/18/2024 Total: $500.00"
    ]
    targets = [100.00, 250.50, 75.25, 500.00]
    
    regressor = DocumentRegressor()
    metrics = regressor.train(texts, targets)
    predictions = regressor.predict(texts)
    print("Predictions:", predictions)

