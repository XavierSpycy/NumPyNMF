import numpy as np
from typing import Union

def PCA(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Implementation of Principal Component Analysis (PCA) algorithm to reduce the 
    dimensionality of the dataset while preserving as much variance as possible.

    Parameters:
        X (numpy.ndarray): Input dataset of shape (n_samples, n_features) where 
                           n_samples is the number of samples and n_features is 
                           the number of features in the dataset.
        n_components (int): The number of principal components to retain.

    Return:
        transformed_data (numpy.ndarray): Transformed dataset of shape 
                                          (n_samples, n_components).
    """

    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # Calculate the covariance matrix
    cov_mat = np.cov(X_centered, rowvar=False)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Calculate the principal components
    principal_components = np.dot(X_centered, eigenvectors)
    # Calculate the projection matrix
    projection_matrix = eigenvectors[:, :n_components]
    # Calculate the transformed data
    transformed_data = np.dot(X_centered, projection_matrix)
    return transformed_data

def FastICA(X: np.ndarray, max_iter: int=200, random_state: Union[int, np.random.RandomState, None]=None) -> np.ndarray:
    """
    Implementation of FastICA algorithm to separate the independent sources 
    from mixed signals in the input data.
    
    Parameters:
        X (numpy.ndarray): Input dataset of shape (n_samples, n_features) where 
                           n_samples is the number of independent sources and 
                           n_features is the number of observations.
        max_iter (int, optional): The maximum number of iterations for the convergence 
                                  of the estimation. Default is 200.
                                  
    Return:
        S (numpy.ndarray): Matrix of shape (n_samples, n_features) representing 
                           the estimated independent sources.
    """

    # Set the random state
    rng = np.random.RandomState(random_state)
    # Center the data by removing the mean
    X -= X.mean(axis=1, keepdims=True)
    n = X.shape[0]
    # Compute the independent components iteratively
    W = np.zeros((n, n))
    for i in range(n):
        w = rng.rand(n)
        for j in range(max_iter):  # max iterations for convergence
            w_new = (X * np.dot(w, X)).mean(axis=1) - 2 * w
            w_new /= np.sqrt((w_new ** 2).sum())
            # Convergence check based on the weight vector's direction
            if np.abs(np.abs((w_new * w).sum()) - 1) < 1e-04:
                break
            w = w_new
        W[i, :] = w
        X -= np.outer(w, np.dot(w, X))
    # Compute the estimated independent sources
    S = np.dot(W, X)
    return S