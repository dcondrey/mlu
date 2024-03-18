from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def isolation_forest(data, contamination=0.01):
    """
    Applies the Isolation Forest algorithm to detect outliers.

    Parameters:
    - data (numpy.ndarray): The input dataset.
    - contamination (float): The proportion of outliers in the data.

    Returns:
    - numpy.ndarray: Indices of detected anomalies.
    """
    try:
        clf = IsolationForest(contamination=contamination)
        preds = clf.fit_predict(data)
        # Isolation Forest marks inliers as 1 and outliers as -1
        anomaly_indices = np.where(preds == -1)[0]
        logging.info("Isolation Forest anomaly detection completed successfully.")
        return anomaly_indices
    except Exception as e:
        logging.error("An error occurred during Isolation Forest anomaly detection: %s", e, exc_info=True)
        raise

def local_outlier_factor(data, neighbors=20):
    """
    Applies the Local Outlier Factor algorithm to identify anomalies.

    Parameters:
    - data (numpy.ndarray): The input dataset.
    - neighbors (int): Number of neighbors to consider for measuring the local density.

    Returns:
    - numpy.ndarray: Indices of detected anomalies.
    """
    try:
        clf = LocalOutlierFactor(n_neighbors=neighbors, novelty=True)
        clf.fit(data)
        # LOF does not directly give predictions, so we need to fit first, 
        # then use negative_outlier_factor_ to find anomalies. Here's a simplified approach:
        # For simplicity, thresholding based on the quantile might not be perfect for every case.
        outlier_factor_threshold = np.percentile(clf.negative_outlier_factor_, 95)  # assuming the top 5% as outliers
        anomaly_indices = np.where(clf.negative_outlier_factor_ > outlier_factor_threshold)[0]
        logging.info("Local Outlier Factor anomaly detection completed successfully.")
        return anomaly_indices
    except Exception as e:
        logging.error("An error occurred during Local Outlier Factor anomaly detection: %s", e, exc_info=True)
        raise