import numpy as np
from sklearn.metrics import mean_squared_error,  accuracy_score, normalized_mutual_info_score

def RMSE(X: np.ndarray, D: np.ndarray, R: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between the original matrix X and its approximation using D and R.
    
    Parameters:
    - X (numpy.ndarray): Original data matrix of shape (n_samples, n_features).
    - D (numpy.ndarray): Basis matrix obtained from matrix factorization.
    - R (numpy.ndarray): Coefficient matrix obtained from matrix factorization.

    Returns:
    - float: The computed RMSE value.

    Note:
    Lower RMSE indicates a closer match between X and the approximated matrix using D and R.
    """

    return np.sqrt(mean_squared_error(X, D.dot(R)))

def Acc(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy between the true labels and the predicted labels.
    
    Parameters:
    - Y_true (numpy.ndarray): Ground truth label array of shape (n_samples,).
    - Y_pred (numpy.ndarray): Predicted label array of shape (n_samples,).

    Returns:
    - float: The accuracy score ranging from 0.0 (worst) to 1.0 (best).
    """

    return accuracy_score(Y_true, Y_pred)

def NMI(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculate the Normalized Mutual Information (NMI) between the true labels and the predicted labels.
    
    Parameters:
    - Y_true (numpy.ndarray): Ground truth label array of shape (n_samples,).
    - Y_pred (numpy.ndarray): Predicted label array of shape (n_samples,).

    Returns:
    - float: The NMI score ranging from 0.0 (no mutual information) to 1.0 (perfect correlation).

    Note:
    NMI is especially useful for evaluating clustering algorithms, particularly when true cluster labels are known.
    """
    
    return normalized_mutual_info_score(Y_true, Y_pred)