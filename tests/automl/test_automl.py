from modules.automl.automl import AutoML
import unittest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TestAutoML(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier()
        }
        self.param_grid = {
            'LogisticRegression': {'C': [0.1, 1.0, 10.0]},
            'RandomForest': {'n_estimators': [10, 50, 100]}
        }
        self.automl = AutoML()

    def test_model_selection(self):
        best_model = self.automl.model_selection(self.X, self.y, self.models, scoring='accuracy')
        self.assertIn(type(best_model).__name__, ['LogisticRegression', 'RandomForest'], "The best model selected is not among the expected types.")

    def test_hyperparameter_tuning(self):
        model = LogisticRegression(max_iter=1000)
        tuned_model = self.automl.hyperparameter_tuning(model, self.param_grid['LogisticRegression'], self.X, self.y, scoring='accuracy')
        self.assertIsInstance(tuned_model, LogisticRegression, "The tuned model is not an instance of LogisticRegression.")
        self.assertIn(tuned_model.C, [0.1, 1.0, 10.0], "The tuned parameter C for LogisticRegression is not within the expected range.")

if __name__ == '__main__':
    unittest.main()