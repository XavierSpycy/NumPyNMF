import numpy as np
from typing import Tuple
from algorithm.nmf import BasicNMF
from algorithm.visualize import origin_versus_dictrep

def evaluate(nmf: BasicNMF, metrics: Tuple, X: np.ndarray, X_noise: np.ndarray, 
             image_size: tuple, reduce: int, idx=2, imshow: bool=False) -> None:
    
    """Evaluate the performance of NMF algorithms.
    
    Parameters
    - nmf (BasicNMF): The NMF algorithm.
    - metrics (tuple): The evaluation metrics, (rmse, acc, nmi).
    - X (numpy.ndarray): The original data matrix, shape (n_samples, n_features).
    - X_noise (numpy.ndarray): The noisy data matrix, shape (n_samples, n_features).
    - image_size (tuple): The size of images.
    - reduce (int): The reduction ratio of images.
    - idx (int): The index of the image to be visualized.
    - random_state (int): The random state.
    """
    # Start to evaluate
    print('Evaluating...')
    rmse, acc, nmi = metrics
    # Visualize
    print('RMSE = {:.4f}'.format(rmse))
    print('Accuracy = {:.4f}'.format(acc))
    print('NMI = {:.4f}'.format(nmi))
    if imshow:
        origin_versus_dictrep(X, nmf.D, nmf.R, X_noise, image_size=image_size, reduce=reduce, idx=idx)