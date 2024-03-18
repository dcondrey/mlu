def detect_data_leakage(X_train, X_test):
    """
    Detects potential data leakage between training and testing datasets.
    
    Parameters:
    - X_train (pandas.DataFrame): Training features dataset.
    - X_test (pandas.DataFrame): Testing features dataset.
    
    Returns:
    - bool: True if data leakage is detected, False otherwise.
    """
    try:
        common_elements = set(X_train.columns) & set(X_test.columns)
        if len(common_elements) != len(X_train.columns) or len(common_elements) != len(X_test.columns):
            print("Data Leakage Detected: True")
            return True
        print("Data Leakage Detected: False")
        return False
    except Exception as e:
        print(f"An error occurred during detecting data leakage: {str(e)}")
        raise