import numpy as np
from scipy.linalg import pinv
from typing import Union
from algorithm.decomposition import PCA, FastICA

def NICA(X: np.ndarray, r: int, random_state: Union[int, np.random.RandomState, None]=None) -> (np.ndarray, np.ndarray):
    """
    Implementation of a non-negative Independent Component Analysis (NICA). 
    The process involves obtaining a non-negative basic matrix and a 
    non-negative coefficient matrix from the input data.

    Parameters:
    - X (numpy.ndarray): The input data matrix of shape (n_samples, n_features) 
                         where n_samples is the number of samples, and n_features 
                         is the number of features.
    - r (int): The number of components to be retained after applying PCA.

    Returns:
    - W_0 (numpy.ndarray): The non-negative basic matrix.
    - H_0 (numpy.ndarray): The non-negative coefficient matrix.
    """

    # Set A as a pseudoinverse of X
    A = pinv(X)

    # Apply PCA on the matrix A to generate the basic matrix W
    W = PCA(A.T, n_components=r)

    # Whiten the basic matrix W obtained above by using the eigenvalue decomposition of the covariance matrix of W.
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(W, rowvar=False))
    
    # Preallocate memory for whitened matrix
    W_whitened = np.empty_like(W)
    np.dot(W, eigenvectors, out=W_whitened)
    W_whitened /= np.sqrt(eigenvalues + 1e-5)

    # Implement ICA algorithm on the whitened matrix W and obtain the independent basic matrix W_0
    # Assuming FastICA() returns the transformed matrix
    W_0 = FastICA(W_whitened, random_state=random_state)

    # Preallocate memory for H_0 and calculate it
    H_0 = np.empty((W_0.shape[1], X.shape[1]))
    np.dot(W_0.T, X, out=H_0)

    # Take the absolute value in-place
    np.abs(W_0, out=W_0)
    np.abs(H_0, out=H_0)

    return W_0, H_0