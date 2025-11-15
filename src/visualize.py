"""
Visualization Module for Document Intelligence System
Creates plots and visualizations for analysis
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Optional


def ensure_plots_dir():
    """Ensure plots directory exists."""
    os.makedirs('plots', exist_ok=True)


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], 
                         class_names: Optional[List[str]] = None,
                         save_path: str = 'plots/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    if class_names is None:
        class_names = sorted(list(set(y_true + y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_distribution(labels: List[str], 
                           save_path: str = 'plots/class_distribution.png'):
    """
    Plot distribution of document classes.
    
    Args:
        labels: List of labels
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    label_counts = pd.Series(labels).value_counts()
    
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='steelblue')
    plt.title('Document Class Distribution')
    plt.xlabel('Document Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           top_n: int = 20,
                           save_path: str = 'plots/feature_importance.png'):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def plot_training_history(train_scores: List[float], test_scores: List[float],
                         metric_name: str = 'Accuracy',
                         save_path: str = 'plots/training_history.png'):
    """
    Plot training history.
    
    Args:
        train_scores: Training scores over epochs/iterations
        test_scores: Test scores over epochs/iterations
        metric_name: Name of the metric
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, test_scores, 'r-', label=f'Test {metric_name}')
    plt.title(f'Training History - {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_prediction_distribution(predictions: np.ndarray,
                                save_path: str = 'plots/prediction_distribution.png'):
    """
    Plot distribution of predictions.
    
    Args:
        predictions: Array of predictions
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction distribution plot saved to {save_path}")


def plot_entity_extraction_results(entities: Dict[str, List[str]],
                                  save_path: str = 'plots/entity_counts.png'):
    """
    Plot counts of extracted entities.
    
    Args:
        entities: Dictionary mapping entity types to lists of entities
        save_path: Path to save the plot
    """
    ensure_plots_dir()
    
    entity_counts = {k: len(v) for k, v in entities.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counts.keys(), entity_counts.values(), color='coral')
    plt.title('Extracted Entity Counts')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Entity counts plot saved to {save_path}")


if __name__ == "__main__":
    # Test visualization functions
    labels = ['invoice', 'receipt', 'invoice', 'resume', 'contract', 'receipt']
    plot_class_distribution(labels)
    
    y_true = ['invoice', 'receipt', 'invoice', 'resume']
    y_pred = ['invoice', 'receipt', 'invoice', 'resume']
    plot_confusion_matrix(y_true, y_pred)

