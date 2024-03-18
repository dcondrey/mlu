import numpy as np
from modules.data_transformation import handle_missing_values, normalize, encode_categorical, remove_low_variance_features, apply_pca, generate_polynomial_features
from modules.array_manipulation import filter, aggregate, summary
from modules.model_building import initialize_decision_tree, initialize_neural_network, optimize_hyperparameters
from modules.model_evaluation import calculate_metrics, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Chain:
    def __init__(self, data):
        self.data = np.array(data)
        logging.info("Chain initialized with data.")

    def handle_missing_values(self, strategy='mean'):
        try:
            self.data = handle_missing_values(self.data, strategy)
            logging.info(f"Missing values handled using {strategy} strategy.")
        except Exception as e:
            logging.error(f"An error occurred in handle_missing_values: {str(e)}", exc_info=True)
        return self

    def normalize(self):
        try:
            self.data = normalize(self.data)
            logging.info("Data normalized.")
        except Exception as e:
            logging.error(f"An error occurred in normalize: {str(e)}", exc_info=True)
        return self

    def encode_categorical(self, encoding_type='onehot'):
        try:
            self.data = encode_categorical(self.data, encoding_type)
            logging.info(f"Categorical data encoded using {encoding_type} encoding.")
        except Exception as e:
            logging.error(f"An error occurred in encode_categorical: {str(e)}", exc_info=True)
        return self

    def filter(self, condition):
        try:
            self.data = filter(self.data, condition)
            logging.info("Data filtered.")
        except Exception as e:
            logging.error(f"An error occurred in filter: {str(e)}", exc_info=True)
        return self

    def aggregate(self, operation):
        try:
            result = aggregate(self.data, operation)
            self.data = np.array([result])  # Wrap the result in an array to keep the data consistent
            logging.info(f"Data aggregated using {operation} operation.")
        except Exception as e:
            logging.error(f"An error occurred in aggregate: {str(e)}", exc_info=True)
        return self

    def map(self, function):
        try:
            self.data = np.array(list(map(function, self.data)))
            logging.info("Data mapped.")
        except Exception as e:
            logging.error(f"An error occurred in map: {str(e)}", exc_info=True)
        return self

    def summary(self):
        try:
            result = summary(self.data)
            logging.info("Summary statistics generated.")
            for key, value in result.items():
                logging.info(f"{key}: {value}")
            self.data = np.array([result])  # Wrap the summary in an array to maintain consistency and allow chaining
        except Exception as e:
            logging.error(f"An error occurred in summary: {str(e)}", exc_info=True)
        return self

    def value(self):
        logging.info("Returning data.")
        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[:, :-1], self.data[:, -1], test_size=test_size, random_state=random_state)
            logging.info("Data split into training and testing sets.")
        except Exception as e:
            logging.error(f"An error occurred in data splitting: {str(e)}", exc_info=True)
        return self

    def select_model(self, model_type, **kwargs):
        try:
            if model_type == 'decision_tree':
                self.model = initialize_decision_tree(**kwargs)
            elif model_type == 'neural_network':
                self.model = initialize_neural_network(**kwargs)
            else:
                raise ValueError("Unsupported model type. Choose 'decision_tree' or 'neural_network'.")
            logging.info(f"{model_type} model selected.")
        except Exception as e:
            logging.error(f"An error occurred in model selection: {str(e)}", exc_info=True)
        return self

    def train_model(self):
        try:
            if self.model is None:
                raise ValueError("Model not selected. Use the select_model method first.")
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model trained successfully.")
        except Exception as e:
            logging.error(f"An error occurred in model training: {str(e)}", exc_info=True)
        return self

    def evaluate_model(self):
        try:
            if self.model is None:
                raise ValueError("Model not trained. Use the train_model method first.")
            predictions = self.model.predict(self.X_test)
            calculate_metrics(self.y_test, predictions)
        except Exception as e:
            logging.error(f"An error occurred in model evaluation: {str(e)}", exc_info=True)
        return self