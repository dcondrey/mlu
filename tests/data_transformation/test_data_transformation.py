import numpy as np
import pandas as pd
import sys
import os

# Adjust system path to include the directory above this one, to find the modules package
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.data_transformation.data_transformation import (
    handle_missing_values,
    normalize,
    encode_categorical,
    remove_low_variance_features,
    apply_pca,
    generate_polynomial_features
)

def test_handle_missing_values():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    result = handle_missing_values(df, 'mean')
    assert result.isnull().sum().sum() == 0, "Failed to handle missing values."

def test_normalize():
    array = np.array([1, 2, 3, 4, 5])
    result = normalize(array)
    expected = np.array([0., 0.25, 0.5, 0.75, 1.])
    np.testing.assert_array_equal(result, expected), "Normalization failed."

def test_encode_categorical():
    df = pd.DataFrame({'A': ['cat', 'dog', 'cat']})
    result = encode_categorical(df, 'label')
    assert result.tolist() == [0, 1, 0], "Categorical encoding failed."

def test_remove_low_variance_features():
    df = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3]})
    result = remove_low_variance_features(df)
    assert 'A' not in result.columns, "Failed to remove low variance features."

def test_apply_pca():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = apply_pca(df, n_components=2)
    assert result.shape[1] == 2, "PCA application failed."

def test_generate_polynomial_features():
    data = np.array([[1, 2], [3, 4]])
    result = generate_polynomial_features(data, degree=2)
    assert result.shape == (2, 6), "Polynomial feature generation failed."  # Includes bias and original features