from typing import Union

import numpy as np

def random_sample(X: np.ndarray, Y: np.ndarray, fraction: float=0.90, random_state: Union[int, np.random.RandomState, None]=None) -> np.ndarray:
    """
    Randomly sample a fraction of the data.

    Parameters:
    - X (numpy.ndarray): The input data matrix of shape (n_features, n_samples)
                            where n_samples is the number of samples, and n_features
                            is the number of features.
    - Y (numpy.ndarray): The output data matrix of shape (n_samples, )
    - fraction (float): The fraction of the data to be sampled.
    - random_state (int): The seed for the random number generator.

    Returns:
    - X_sample (numpy.ndarray): The sampled data matrix of shape (n_features, n_samples)
                                where n_samples is the number of samples, and n_features

    """
    
    # Create a random number generator
    rng = np.random.default_rng(random_state)

    # Compute the number of samples to be drawn
    n_samples = X.shape[1]
    sample_size = int(fraction * n_samples)

    # Randomly sample the indices
    sampled_indices = rng.choice(n_samples, sample_size, replace=False)

    # Use the sampled indices to extract columns from the original data
    X_sample = X[:, sampled_indices]
    Y_sample = Y[sampled_indices]

    return X_sample, Y_sample