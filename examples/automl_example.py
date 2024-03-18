from mlu.modules.automl import AutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing AutoML example.")
    
    # Load sample dataset
    logging.info("Loading Iris dataset.")
    X, y = load_iris(return_X_y=True)
    
    # Splitting dataset into training and testing
    logging.info("Splitting dataset into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    automl = AutoML()
    
    # Demonstrating model selection
    logging.info("Demonstrating model selection.")
    models = {
        'LogisticRegression': LogisticRegression(max_iter=200),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
    }
    best_model_name, best_model = automl.model_selection(X_train, y_train, models, scoring='accuracy')
    logging.info(f"Selected best model: {best_model_name}")

    # Demonstrating hyperparameter tuning
    logging.info("Demonstrating hyperparameter tuning.")
    if best_model_name == 'LogisticRegression':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [100, 200, 300]
        }
    elif best_model_name == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 20, None]
        }
    tuned_model = automl.hyperparameter_tuning(best_model, param_grid, X_train, y_train, scoring='accuracy')
    logging.info(f"Tuned model with best parameters: {tuned_model}")
    
    logging.info("AutoML example demonstration completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during the AutoML example execution.", exc_info=True)