from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutoML:
    def __init__(self):
        pass

    def model_selection(self, X, y, models, scoring):
        """
        Selects the best model from a list of models based on cross-validation scores.
        Parameters:
        - X: Features dataset.
        - y: Target dataset.
        - models: Dictionary of models to evaluate {model_name: model_instance}.
        - scoring: Scoring metric.
        """
        best_score = -np.inf
        best_model_name = None
        best_model = None
        for name, model in models.items():
            score = np.mean(cross_val_score(model, X, y, scoring=scoring))
            logging.info(f"Model: {name}, Score: {score}")
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = model
        logging.info(f"Best Model: {best_model_name}, Score: {best_score}")
        return best_model

    def hyperparameter_tuning(self, model, param_grid, X, y, scoring):
        """
        Tunes hyperparameters for the given model using GridSearchCV.
        Parameters:
        - model: The model instance to optimize.
        - param_grid: The parameter grid to search over.
        - X: Features dataset.
        - y: Target dataset.
        - scoring: Scoring metric.
        """
        grid_search = GridSearchCV(model, param_grid, scoring=scoring)
        grid_search.fit(X, y)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def parallel_grid_search(self, model, param_grid, X, y, cv=5, n_jobs=-1, scoring=None, early_stopping_rounds=None):
        """
        Performs grid search with parallel processing and optional early stopping.
        Parameters:
        - model: The model instance to optimize.
        - param_grid: The parameter grid to search over.
        - X: Features dataset.
        - y: Target dataset.
        - cv (int): Number of cross-validation folds.
        - n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        - scoring: Scoring metric.
        - early_stopping_rounds (int, optional): Activates early stopping if set. Grid search will stop if the score doesn't improve after the specified number of rounds.
        """
        if early_stopping_rounds:
            logging.warning("Early stopping is not directly supported by GridSearchCV. Implementing model-specific early stopping logic.")

            # Implement early stopping in model training
            best_score = None
            best_estimator = None
            for g in ParameterGrid(param_grid):
                model.set_params(**g)
                scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
                if best_score is None or np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_estimator = model
                if len(scores) > early_stopping_rounds and np.std(scores[-early_stopping_rounds:]) < 0.01:
                    break
            logging.info(f"Early stopping activated. Best score: {best_score}")
            return best_estimator
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, scoring=scoring)
        grid_search.fit(X, y)
        logging.info(f"Parallel grid search best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_