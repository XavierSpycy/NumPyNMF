import numpy as np
from typing import Union
from collections import Counter
from sklearn.cluster import BisectingKMeans

def labeling(X: np.ndarray, Y: np.ndarray, random_state: Union[int, np.random.RandomState, None]=None) -> np.ndarray:
    """
    Label data based on clusters obtained from KMeans clustering, 
    by assigning the most frequent label in each cluster.
    
    Parameters:
    - X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
    - Y (numpy.ndarray): True labels corresponding to each sample in X of shape (n_samples,).

    Returns:
    - Y_pred (numpy.ndarray): Predicted labels for each sample based on the clustering results.

    Note:
    This function works best when the input data is somewhat separated into distinct 
    clusters that align with the true labels.
    """

    cluster = BisectingKMeans(len(set(Y)), random_state=random_state).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(cluster.labels_):
        ind = cluster.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred