from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump, load
from modules.model_evaluation.model_evaluation import calculate_metrics
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_pipeline(preprocessing_steps, model):
    """
    Creates a machine learning pipeline.
    
    Parameters:
    - preprocessing_steps (list of tuples): Each tuple contains the name and the transformer object.
    - model (tuple): A tuple containing the name and the model object.
    
    Returns:
    - Pipeline: The constructed pipeline.
    """
    try:
        steps = preprocessing_steps + [model]
        pipeline = Pipeline(steps)
        logging.info("Pipeline created successfully.")
        return pipeline
    except Exception as e:
        logging.error("Failed to create pipeline.", exc_info=True)
        raise

def train_pipeline(pipeline, X_train, y_train):
    """
    Trains the given pipeline on the provided training data.
    
    Parameters:
    - pipeline (Pipeline): The pipeline to train.
    - X_train: Training features.
    - y_train: Training labels.
    
    Returns:
    - Pipeline: The trained pipeline.
    """
    try:
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline trained successfully.")
        return pipeline
    except Exception as e:
        logging.error("Failed to train pipeline.", exc_info=True)
        raise

def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Evaluates the trained pipeline on the test data using model_evaluation module.
    
    Parameters:
    - pipeline (Pipeline): The trained pipeline.
    - X_test: Test features.
    - y_test: Test labels.
    """
    try:
        predictions = pipeline.predict(X_test)
        calculate_metrics(y_test, predictions)
    except Exception as e:
        logging.error("Failed to evaluate pipeline.", exc_info=True)
        raise

def save_pipeline(pipeline, filename):
    """
    Saves the trained pipeline to a file.
    
    Parameters:
    - pipeline (Pipeline): The trained pipeline.
    - filename (str): The filename to save the pipeline.
    """
    try:
        dump(pipeline, filename)
        logging.info(f"Pipeline saved to {filename}.")
    except Exception as e:
        logging.error("Failed to save pipeline.", exc_info=True)
        raise

def load_pipeline(filename):
    """
    Loads a pipeline from a file.
    
    Parameters:
    - filename (str): The filename to load the pipeline from.
    
    Returns:
    - Pipeline: The loaded pipeline.
    """
    try:
        pipeline = load(filename)
        logging.info(f"Pipeline loaded from {filename}.")
        return pipeline
    except Exception as e:
        logging.error("Failed to load pipeline.", exc_info=True)
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Define X, y with your data
        X, y = None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define actual preprocessing steps and model
        preprocessing_steps = [('preprocessor', None)] 
        model = ('model', None)
        
        pipeline = create_pipeline(preprocessing_steps, model)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        evaluate_pipeline(trained_pipeline, X_test, y_test)
        
        # Save the trained pipeline
        save_pipeline(trained_pipeline, 'trained_pipeline.joblib')
        
        # Load the pipeline (for demonstration)
        loaded_pipeline = load_pipeline('trained_pipeline.joblib')
    except Exception as e:
        logging.error("An error occurred in the main execution block.", exc_info=True)