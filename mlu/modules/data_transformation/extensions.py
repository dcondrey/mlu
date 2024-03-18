"""
This module is designed for extending the data_transformation functionality of the mlu application.
Developers can add custom data transformation functions here or extend the existing ones.

Instructions:
- Define your custom function.
- Ensure it follows the input-output consistency with the core module functions.
- Import and use core module functionalities if needed to avoid code duplication.
- Export your functions by adding them to the __all__ list.

Example:
def custom_normalizer(array):
    # Custom normalization logic
    normalized_array = array / array.max()
    return normalized_array

__all__ = ['custom_normalizer']
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_normalizer(array):
    try:
        # Custom normalization logic
        normalized_array = array / array.max()
        logging.info("Custom normalization completed successfully.")
        return normalized_array
    except Exception as e:
        logging.error("An error occurred in custom_normalizer: %s", e, exc_info=True)
        raise

__all__ = ['custom_normalizer']  # Add the names of your custom functions here to make them available for import.