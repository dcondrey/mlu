import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_feature_importance(model, feature_names, X=None, y=None, n_repeats=10, random_state=42, n_jobs=2, save_path=None):
    """
    Plots the feature importance of a fitted model.
    
    Parameters:
    - model: The fitted model instance.
    - feature_names (list): A list of feature names.
    - X (numpy.ndarray, optional): Feature dataset for permutation importance.
    - y (numpy.ndarray, optional): Target dataset for permutation importance.
    - n_repeats (int, optional): Number of times to permute a feature.
    - random_state (int, optional): Seed for the random number generator.
    - n_jobs (int, optional): Number of jobs to run in parallel.
    - save_path (str, optional): Path to save the plot image.
    """
    try:
        importance_vals = model.feature_importances_
        sns.barplot(x=importance_vals, y=feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info("Feature importance plot created.")
    except AttributeError as e:
        if X is None or y is None:
            logging.error("X and y need to be provided for permutation importance calculation.", exc_info=True)
            return
        logging.error("Error in plot_feature_importance: The model does not have an attribute 'feature_importances_'. Trying permutation importance.", exc_info=True)
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        sorted_idx = result.importances_mean.argsort()
        plt.barh([feature_names[i] for i in sorted_idx], result.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info("Permutation importance plot created.")

def plot_decision_boundaries(model, X, y, save_path=None):
    """
    Plots the decision boundaries for a model. Works for models with two features.
    
    Parameters:
    - model: The fitted model instance.
    - X (numpy.ndarray): Feature dataset (should be two-dimensional).
    - y (numpy.ndarray): Target dataset.
    - save_path (str, optional): Path to save the plot image.
    """
    try:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title('Decision Boundaries')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info("Decision boundaries plot created.")
    except Exception as e:
        logging.error("Error plotting decision boundaries.", exc_info=True)

def plot_model_predictions(model, X, y, save_path=None):
    """
    Plots the model predictions against the actual labels.
    
    Parameters:
    - model: The fitted model instance.
    - X (numpy.ndarray): Feature dataset.
    - y (numpy.ndarray): Actual labels.
    - save_path (str, optional): Path to save the plot image.
    """
    try:
        predictions = model.predict(X)
        plt.figure(figsize=(10, 5))
        plt.plot(predictions, label='Predictions')
        plt.plot(y, label='Actual', alpha=0.7)
        plt.legend()
        plt.title('Model Predictions vs Actual')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        logging.info("Model predictions plot created.")
    except Exception as e:
        logging.error("Error plotting model predictions.", exc_info=True)