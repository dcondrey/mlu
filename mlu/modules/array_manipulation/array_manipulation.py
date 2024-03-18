import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter(array, condition):
    """
    Filters elements in an array based on a condition function.
    
    Parameters:
    - array (numpy.ndarray): The input array.
    - condition (callable): A function that returns a boolean value.
    
    Returns:
    - numpy.ndarray: Filtered array.
    """
    try:
        result = array[condition(array)]
        logging.info("Array filtered successfully.")
        return result
    except Exception as e:
        logging.error(f"An error occurred in filter function: {str(e)}", exc_info=True)
        raise

def aggregate(array, operation):
    """
    Applies an aggregation operation on an array.
    
    Parameters:
    - array (numpy.ndarray): The input array.
    - operation (str): The aggregation operation ('sum', 'mean', 'max', 'min').
    
    Returns:
    - float: The result of the aggregation operation.
    """
    try:
        if operation == 'sum':
            result = np.sum(array)
        elif operation == 'mean':
            result = np.mean(array)
        elif operation == 'max':
            result = np.max(array)
        elif operation == 'min':
            result = np.min(array)
        else:
            raise ValueError("Unsupported operation. Choose from 'sum', 'mean', 'max', 'min'.")
        logging.info(f"Aggregate operation '{operation}' completed successfully.")
        return result
    except Exception as e:
        logging.error(f"An error occurred in aggregate function: {str(e)}", exc_info=True)
        raise

def summary(array):
    """
    Generates a statistical summary of the numerical data in the array.
    
    Parameters:
    - array (numpy.ndarray): The input array.
    
    Returns:
    - dict: A dictionary containing count, mean, std, min, max, 25%, 50%, 75% percentiles.
    """
    try:
        summary_stats = {
            "count": array.size,
            "mean": np.mean(array),
            "std": np.std(array),
            "min": np.min(array),
            "max": np.max(array),
            "25%": np.percentile(array, 25),
            "50%": np.median(array),
            "75%": np.percentile(array, 75)
        }
        logging.info("Summary statistics generated successfully.")
        return summary_stats
    except Exception as e:
        logging.error(f"An error occurred in summary function: {str(e)}", exc_info=True)
        raise