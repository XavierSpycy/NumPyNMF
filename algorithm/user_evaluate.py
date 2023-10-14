import numpy as np
from typing import Union
from algorithm.label import labeling
from algorithm.evaluations import RMSE, Acc, NMI
from algorithm.visualize import origin_versus_dictrep

def evaluate(X: np.ndarray, D: np.ndarray, R: np.ndarray, Y: np.ndarray, X_noise: np.ndarray, image_size: tuple, reduce: int, idx=2, 
            random_state: Union[int, np.random.RandomState, None]=None, imshow: bool=False) -> None:
    
    """Evaluate the performance of NMF algorithms.
    
    Parameters
    - X (numpy.ndarray): The original data matrix, shape (n_samples, n_features).
    - D (numpy.ndarray): The dictionary matrix, shape (n_features, n_components).
    - R (numpy.ndarray): The representation matrix, shape (n_components, n_samples).
    - Y (numpy.ndarray): The label matrix, shape (n_samples, n_classes).
    - X_noise (numpy.ndarray): The noisy data matrix, shape (n_samples, n_features).
    - image_size (tuple): The size of images.
    - reduce (int): The reduction ratio of images.
    - idx (int): The index of the image to be visualized.
    - random_state (int): The random state.
    """

    # Start to evaluate
    print('==> Evaluating...')
    # Calculate RMSE
    rmse = RMSE(X, D, R)
    # Labeling using KMeans or its variants
    Y_label = labeling(R.T, Y, random_state=random_state)
    # Calculate Accuracy and NMI
    acc = Acc(Y, Y_label)   # Accuracy
    nmi = NMI(Y, Y_label)   # Normalized Mutual Information
    # Visualize
    print('RMSE = {:.4f}'.format(rmse))
    print('Accuracy = {:.4f}'.format(acc))
    print('NMI = {:.4f}'.format(nmi))
    if imshow:
        origin_versus_dictrep(X, D, R, X_noise, image_size=image_size, reduce=reduce, idx=idx)