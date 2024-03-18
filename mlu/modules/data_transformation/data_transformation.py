import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_missing_values(data, strategy='mean'):
    try:
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'mode':
            # mode() returns a DataFrame. Use iloc to get the first mode if exists.
            return data.fillna(data.mode().iloc[0])
        elif strategy == 'constant':
            return data.fillna(0)  # Assuming 0 as the constant value for simplicity.
        else:
            raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', or 'constant'.")
    except Exception as e:
        logging.error(f"An error occurred in handle_missing_values: {e}", exc_info=True)
        raise

def normalize(array):
    try:
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)
    except Exception as e:
        logging.error(f"An error occurred in normalize: {e}", exc_info=True)
        raise

def encode_categorical(data, encoding_type='onehot'):
    try:
        if encoding_type == 'onehot':
            return pd.get_dummies(data)
        elif encoding_type == 'label':
            return data.astype('category').cat.codes
        else:
            raise ValueError("Unsupported encoding type. Choose from 'onehot' or 'label'.")
    except Exception as e:
        logging.error(f"An error occurred in encode_categorical: {e}", exc_info=True)
        raise

def remove_low_variance_features(data, threshold=0.0):
    """
    Removes features with low variance.

    Parameters:
    - data (pandas.DataFrame): The input dataset.
    - threshold (float): Features with a variance lower than this threshold will be removed.

    Returns:
    - pandas.DataFrame: The transformed dataset with low-variance features removed.
    """
    try:
        selector = VarianceThreshold(threshold=threshold)
        data_transformed = selector.fit_transform(data)
        return pd.DataFrame(data_transformed, columns=data.columns[selector.get_support()])
    except Exception as e:
        logging.error(f"An error occurred in remove_low_variance_features: {e}", exc_info=True)
        raise

def apply_pca(data, n_components=None):
    """
    Applies PCA for dimensionality reduction.

    Parameters:
    - data (pandas.DataFrame): The input dataset.
    - n_components (int, optional): Number of components to keep.

    Returns:
    - pandas.DataFrame: The transformed dataset after applying PCA.
    """
    try:
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data)
        components_col = [f"PCA_Component_{i}" for i in range(1, n_components + 1)]
        return pd.DataFrame(data_transformed, columns=components_col)
    except Exception as e:
        logging.error(f"An error occurred in apply_pca: {e}", exc_info=True)
        raise

def generate_polynomial_features(data, degree=2, include_bias=False):
    """
    Generates polynomial features.

    Parameters:
    - data (pandas.DataFrame or numpy.ndarray): The input dataset.
    - degree (int): The degree of the polynomial features.
    - include_bias (bool): If True, include a bias column (the feature in which all polynomial powers are zero).

    Returns:
    - pandas.DataFrame or numpy.ndarray: The transformed dataset with polynomial features.
    """
    try:
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        data_transformed = poly.fit_transform(data)
        return data_transformed
    except Exception as e:
        logging.error(f"An error occurred in generate_polynomial_features: {e}", exc_info=True)
        raise