from mlu.modules.ethical_ai_utils import bias_detection, fairness_metrics

import pandas as pd
import numpy as np

# Sample dataset and predictions
data = pd.DataFrame({
    'Age': [25, 35, 45, 20],
    'Income': [50000, 60000, 80000, 40000],
    'Gender': [0, 1, 0, 1]  # 0: Female, 1: Male
})

predictions = np.array([0, 1, 1, 0])
ground_truth = np.array([0, 1, 0, 1])
groups = data['Gender'].values  # Protected attribute

# Bias detection
try:
    bias_report = bias_detection(data, 'Gender')
    print("Bias Detection Report:\n", bias_report)
except Exception as e:
    print(f"An error occurred during bias detection: {e}")

# Fairness metrics
try:
    fairness_report = fairness_metrics(predictions, ground_truth, groups)
    print("\nFairness Metrics:\n", fairness_report)
except Exception as e:
    print(f"An error occurred during fairness metrics calculation: {e}")