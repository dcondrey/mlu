from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_decision_tree(depth=None, random_state=None):
    """
    Initializes a decision tree model.

    Parameters:
    - depth (int, optional): The maximum depth of the tree.
    - random_state (int, optional): Random state for reproducibility.

    Returns:
    - DecisionTreeClassifier: An uninitialized decision tree classifier.
    """
    try:
        model = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        logging.info("Decision tree model initialized successfully.")
        return model
    except Exception as e:
        logging.error("An error occurred while initializing the decision tree model: %s", e, exc_info=True)
        raise

def initialize_neural_network(input_shape, layers, activation='relu', output_units=1, output_activation='sigmoid'):
    """
    Initializes a neural network model.

    Parameters:
    - input_shape (tuple): Shape of the input data.
    - layers (list of int): Number of units in each hidden layer.
    - activation (str, optional): Activation function for the hidden layers.
    - output_units (int, optional): Number of units in the output layer.
    - output_activation (str, optional): Activation function for the output layer.

    Returns:
    - Sequential: An uninitialized neural network model.
    """
    try:
        model = Sequential()
        model.add(Dense(layers[0], input_shape=input_shape, activation=activation))
        for units in layers[1:]:
            model.add(Dense(units, activation=activation))
        model.add(Dense(output_units, activation=output_activation))
        logging.info("Neural network model initialized successfully.")
        return model
    except Exception as e:
        logging.error("An error occurred while initializing the neural network model: %s", e, exc_info=True)
        raise

def optimize_hyperparameters(model, param_grid, X, y, cv=5, n_iter=None, scoring=None, search_type='grid'):
    """
    Perform hyperparameter optimization for a given model using Grid Search or Randomized Search.

    Parameters:
    - model: The model instance to optimize.
    - param_grid (dict): The parameter grid to search over.
    - X: Features dataset.
    - y: Target dataset.
    - cv (int, optional): Number of cross-validation folds. Defaults to 5.
    - n_iter (int, optional): Number of parameter settings sampled for RandomizedSearchCV. Use only if search_type='random'.
    - scoring (str, optional): A single string to evaluate the predictions on the test set. For example, 'accuracy'.
    - search_type (str, optional): Type of search to perform. Can be 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.

    Returns:
    - The best estimator model after hyperparameter optimization.
    - The best parameter set found during the search.
    """
    if search_type == 'random' and n_iter is None:
        raise ValueError("n_iter must be specified when using RandomizedSearchCV.")
    
    try:
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring)
        else:
            raise ValueError("Invalid search_type. Choose either 'grid' or 'random'.")

        search.fit(X, y)
        logging.info(f"Best parameters found: {search.best_params_}")
        logging.info(f"Best score achieved: {search.best_score_}")

        return search.best_estimator_, search.best_params_
    except Exception as e:
        logging.error("An error occurred during hyperparameter optimization: %s", e, exc_info=True)
        raise