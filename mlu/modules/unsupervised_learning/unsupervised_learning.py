from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cluster_data(data, n_clusters=3):
    """
    Clusters the data into specified number of clusters using KMeans.
    
    Parameters:
    - data (numpy.ndarray): The input dataset.
    - n_clusters (int): The number of clusters to form.
    
    Returns:
    - numpy.ndarray: Cluster labels for each data point.
    """
    try:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        logging.info(f"Data successfully clustered into {n_clusters} clusters.")
        return kmeans.labels_
    except Exception as e:
        logging.error("An error occurred during clustering.", exc_info=True)
        raise