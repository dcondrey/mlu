"""
This module is designed for extending the chaining mechanism functionality of the mlu application.
Developers can add custom chainable functions here or extend the existing ones.

Instructions:
- Define your custom chainable function using decorators to extend the Chain class functionality.
- Ensure it maintains chainability by returning the Chain instance.
- Import and use core module functionalities if needed to avoid code duplication.
- Export your functions by adding them to the __all__ list.

Example:
@extend_chain
def custom_map(chain_instance, function):
    # Custom map logic
    chain_instance.data = np.array(list(map(function, chain_instance.data)))
    return chain_instance

__all__ = ['custom_map']
"""

import numpy as np
import logging
from functools import wraps
from .chain import Chain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extend_chain(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            raise
    setattr(Chain, func.__name__, wrapper)
    return wrapper

@extend_chain
def custom_map(chain_instance, function):
    chain_instance.data = np.array(list(map(function, chain_instance.data)))
    logging.info("Custom map operation completed successfully.")
    return chain_instance

__all__ = ['custom_map']  # Add the names of your custom chainable functions here to make them available for import.