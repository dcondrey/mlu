from mlu.modules.unsupervised_learning import cluster_data
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        data = np.array([[1, 2], [2, 2], [1, -1], [-1, -2], [-2, -1], [-2, 2]])
        n_clusters = 2  # INPUT_REQUIRED {Specify the desired number of clusters}
        labels = cluster_data(data, n_clusters=n_clusters)
        logging.info("Cluster Labels: {}".format(labels))
    except Exception as e:
        logging.error("An error occurred during the Unsupervised Learning example execution: %s", e, exc_info=True)

if __name__ == "__main__":
    main()