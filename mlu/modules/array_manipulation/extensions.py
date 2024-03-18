"""
This module is designed for extending the array_manipulation functionality of the mlu application.
Developers can add custom array manipulation functions here or extend the existing ones.

Instructions:
- Define your custom function.
- Ensure it follows the input-output consistency with the core module functions.
- Import and use core module functionalities if needed to avoid code duplication.
- Export your functions by adding them to the __all__ list.

Example:
def custom_filter(array, condition):
    # Custom filtering logic
    try:
        filtered_array = array[array > condition]
        return filtered_array
    except Exception as e:
        logging.error(f"An error occurred in custom_filter: {e}", exc_info=True)
        raise

__all__ = ['custom_filter']
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_filter(array, condition):
    """
    Custom filter function for array manipulation.

    Parameters:
    - array (numpy.ndarray): The input array.
    - condition (callable): A function that returns a boolean value.

    Returns:
    - numpy.ndarray: Filtered array based on the condition.
    """
    try:
        result = array[condition(array)]
        logging.info("Array filtered successfully using custom_filter.")
        return result
    except Exception as e:
        logging.error(f"An error occurred in custom_filter function: {e}", exc_info=True)
        raise

__all__ = ['custom_filter']  # Add the names of your custom functions here to make them available for import.