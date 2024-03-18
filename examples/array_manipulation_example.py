from mlu.modules.array_manipulation import filter, aggregate, summary

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Sample array
        array = np.array([1, -2, 3, -4, 5])

        # Filtering array based on a condition
        filtered_array = filter(array, lambda x: x > 0)
        logging.info(f"Filtered array (positive values): {filtered_array}")

        # Aggregating array
        sum_value = aggregate(array, 'sum')
        logging.info(f"Sum of array: {sum_value}")

        # Generating summary statistics
        summary_stats = summary(array)
        logging.info(f"Summary statistics: {summary_stats}")
    except Exception as e:
        logging.error("An error occurred in array manipulation example", exc_info=True)

if __name__ == '__main__':
    main()