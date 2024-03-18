from mlu.modules.data_leakage_detector import detect_data_leakage
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting data leakage detection example.")
    X_train = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    X_test = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    try:
        leakage_detected = detect_data_leakage(X_train, X_test)
        logging.info(f"Data Leakage Detected: {leakage_detected}")
    except Exception as e:
        logging.error(f"An error occurred during the data leakage detection process: {e}", exc_info=True)

if __name__ == "__main__":
    main()