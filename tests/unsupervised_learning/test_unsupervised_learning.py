import unittest
from modules.unsupervised_learning.unsupervised_learning import cluster_data
import numpy as np

class TestUnsupervisedLearning(unittest.TestCase):
    def test_cluster_data(self):
        # Test with evenly spaced clusters
        data_even = np.array([[1, 2], [1, 4], [1, 0],
                              [10, 2], [10, 4], [10, 0]])
        labels_even = cluster_data(data_even, n_clusters=2)
        self.assertEqual(len(set(labels_even)), 2, "Unexpected number of clusters returned for evenly spaced data.")
        
        # Test with randomly generated data
        data_random = np.random.rand(100, 2) * 100
        labels_random = cluster_data(data_random, n_clusters=3)
        self.assertEqual(len(set(labels_random)), 3, "Unexpected number of clusters returned for randomly generated data.")
        
        # Test with single cluster
        data_single = np.array([[1, 2], [1, 2], [1, 2]])
        labels_single = cluster_data(data_single, n_clusters=1)
        self.assertTrue(all(label == 0 for label in labels_single), "All data points should belong to a single cluster.")

if __name__ == '__main__':
    unittest.main()