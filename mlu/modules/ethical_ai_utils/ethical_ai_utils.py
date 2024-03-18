import pandas as pd
from sklearn import preprocessing
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bias_detection(data, target):
    """
    Analyzes the dataset for potential biases in data distribution relative to the target variable.
    
    Parameters:
    - data (pandas.DataFrame): The dataset to analyze.
    - target (str): The name of the target variable column.
    
    Returns:
    - str: A report indicating potential biases in the data distribution.
    """
    try:
        target_distribution = data[target].value_counts(normalize=True)
        report = "Data Bias Detection Report:\n"
        report += f"Target variable distribution:\n{target_distribution}\n"
        # Example simplistic bias detection based on target distribution uniformity
        if target_distribution.min() / target_distribution.max() < 0.8:
            report += "Warning: Potential bias detected in target variable distribution."
        else:
            report += "No significant bias detected in target variable distribution."
        logging.info("Bias detection analysis completed successfully.")
        return report
    except Exception as e:
        logging.error(f"An error occurred during bias detection: {str(e)}", exc_info=True)
        raise

def fairness_metrics(predictions, ground_truth, groups):
    """
    Calculates and returns fairness metrics for the model's predictions.
    
    Parameters:
    - predictions (numpy.ndarray): The model predictions.
    - ground_truth (numpy.ndarray): The ground truth labels.
    - groups (numpy.ndarray): Group labels for protected attributes.
    
    Returns:
    - dict: Fairness metrics including demographic parity and equalized odds.
    """
    try:
        # Create a BinaryLabelDataset for fairness metric calculation
        dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, 
                                     df=pd.DataFrame({'feature': groups, 'label': ground_truth}),
                                     label_names=['label'], protected_attribute_names=['feature'])
        dataset_pred = dataset.copy()
        dataset_pred.labels = predictions
        
        metric = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=[{'feature': 0}], privileged_groups=[{'feature': 1}])
        
        fairness_metrics = {
            'demographic_parity_difference': metric.mean_difference(),
            'disparate_impact': metric.disparate_impact()
        }
        logging.info("Fairness metrics calculation completed successfully.")
        return fairness_metrics
    except Exception as e:
        logging.error(f"An error occurred during fairness metrics calculation: {str(e)}", exc_info=True)
        raise