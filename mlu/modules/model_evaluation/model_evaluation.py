import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred):
    """
    Calculate and print various performance metrics.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        
        logging.info(f'Accuracy: {accuracy:.4f}')
        logging.info(f'Precision: {precision:.4f}')
        logging.info(f'Recall: {recall:.4f}')
    except Exception as e:
        logging.error("Error calculating performance metrics", exc_info=True)

def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot ROC curve and calculate AUC.
    
    Parameters:
    - y_true: Actual labels
    - y_scores: Predicted scores (not labels)
    - save_path: Optional path to save the plot image
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info(f"ROC curve plotted successfully.{' Saved to: ' + save_path if save_path else ''}")
    except Exception as e:
        logging.error("Error plotting ROC curve", exc_info=True)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels
    - save_path: Optional path to save the plot image
    """
    try:
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info(f"Confusion Matrix plotted successfully.{' Saved to: ' + save_path if save_path else ''}")
    except Exception as e:
        logging.error("Error plotting confusion matrix", exc_info=True)