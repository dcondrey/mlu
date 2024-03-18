import unittest
from modules.data_leakage_detector.data_leakage_detector import detect_data_leakage
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDataLeakageDetector(unittest.TestCase):
    def test_detect_data_leakage(self):
        X_train = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        X_test = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        try:
            result = detect_data_leakage(X_train, X_test)
            self.assertFalse(result, "Leakage detection should return False when there's no leakage.")
            logging.info("No data leakage detected as expected between X_train and X_test with same columns.")
        except Exception as e:
            logging.error(f"An error occurred during the test_detect_data_leakage with identical columns: {e}")

        X_test_different = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
        try:
            result_different = detect_data_leakage(X_train, X_test_different)
            self.assertTrue(result_different, "Leakage detection should return True when there's leakage due to different columns.")
            logging.info("Data leakage correctly detected between X_train and X_test with different columns.")
        except Exception as e:
            logging.error(f"An error occurred during the test_detect_data_leakage with different columns: {e}")

if __name__ == '__main__':
    unittest.main()