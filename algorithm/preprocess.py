import numpy as np
from typing import Union

class MinMaxScaler:
    """
    This class scales and transforms features to [0, 1].
    """
    def fit(self, X: np.ndarray) -> None:
        """
        Compute the minimum and the range of the data for later scaling.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            The data used to compute the minimum and range used for later scaling.
        """
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale the data using the values computed during the fit method.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            Input data that needs to be scaled.
        
        Returns:
        - numpy array, shape (n_samples, n_features)
            Transformed data.
        """
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to the data and then transform it.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            Input data that needs to be scaled and transformed.
        
        Returns:
        - numpy array, shape (n_samples, n_features)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

class StandardScaler:
    """
    This class standardizes features by removing the mean and scaling to unit variance.
    """
    def fit(self, X: np.ndarray) -> None:
        """
        Compute the mean and standard deviation of the data for later standardization.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            The data used to compute the mean and standard deviation used for later standardization.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the data using the values computed during the fit method.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            Input data that needs to be standardized.
        
        Returns:
        - numpy array, shape (n_samples, n_features)
            Transformed data.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to the data and then transform it.
        
        Parameters:
        - X: numpy array-like, shape (n_samples, n_features)
            Input data that needs to be standardized and transformed.
        
        Returns:
        - numpy array, shape (n_samples, n_features)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

class NoiseAdder:
    """
        This class adds noise to data.
    """
    def __init__(self, random_state: Union[int, np.random.RandomState, None]=None) -> None:
        """
        Initializes the NoiseAdder with a random state and noise parameters.

        Parameters:
        - random_state (int or RandomState instance or None): Controls the randomness. If int, is used as seed for RandomState.
        - noise_params (dict): Additional noise parameters.
        """
        self.rng = np.random.RandomState(random_state)
    
    def add_uniform_noise(self, X_hat: np.ndarray, noise_level: int=0.1) -> (np.ndarray, np.ndarray):
        """
        Add uniform random noise to data.

        Parameters:
        - X_hat (numpy array): Original data.

        Returns:
        - Numpy array of uniform noise.
        - Numpy array with added uniform noise.
        """
        a, b = 0, 1
        # Generate noise
        X_noise = self.rng.uniform(a, b, size=X_hat.shape) * noise_level * (np.max(X_hat) - np.min(X_hat))
        return X_noise, X_hat + X_noise

    def add_gaussian_noise(self, X_hat, noise_level=0.1):
        """
        Add Gaussian noise to data.

        Parameters:
        - X_hat (numpy array): Original data.
        - mean (float): Mean of the Gaussian distribution.
        - std (float): Standard deviation of the Gaussian distribution.

        Returns:
        - Numpy array of Gaussian noise.
        - Numpy array with added Gaussian noise.
        """
        mean, std = 0, 1
        # Generate noise
        X_noise = self.rng.normal(mean, std, size=X_hat.shape) * noise_level * (np.max(X_hat) - np.min(X_hat))
        return X_noise, X_hat + X_noise

    def add_laplacian_noise(self, X_hat, noise_level=0.1):
        """
        Add Laplacian noise to data.

        Parameters:
        - X_hat (numpy array): Original data.
        - mu (float): Location parameter for the Laplacian distribution.
        - lambd (float): Scale (diversity) parameter for the Laplacian distribution.

        Returns:
        - Numpy array of Laplacian noise.
        - Numpy array with added Laplacian noise.
        """
        # Initialize parameters
        mu, lambd = 0, 1
        # Generate noise
        X_noise = self.rng.laplace(mu, lambd, size=X_hat.shape) * noise_level * np.max(X_hat)
        return X_noise, X_hat + X_noise

    def add_block_noise(self, X_hat: np.ndarray, img_width: int, block_size: int=10) -> (np.ndarray, np.ndarray):
        """
        Add block noise to multiple flattened image samples.

        Parameters:
        - X (numpy array): Array of shape (m, n) where m is flattened image length and n is number of samples
        - img_width (int): width of the original image
        - block_size (int): size of the block to occlude
        
        Returns:
        - Numpy array of noise added to each sample
        - Numpy array with added block noise for all samples
        """
        # Initalize parameters
        X = X_hat.copy()
        m, n_samples = X.shape
        X_noise = np.zeros((m, n_samples), dtype=np.uint8)
        # For each sample in X
        for i in range(n_samples):
            sample = X[:, i]
            # Reshape the flattened array to 2D
            img_2d = sample.reshape(-1, img_width)
            height, width = img_2d.shape
            # Ensure the block size isn't larger than the image dimensions
            block_size = min(block_size, width, height)
            # Generate a random starting point for the block
            x_start = self.rng.randint(0, width - block_size)
            y_start = self.rng.randint(0, height - block_size)
            # Add block noise
            img_2d[y_start:y_start+block_size, x_start:x_start+block_size] = 255
            # Store the noise block to noise array
            noise_2d = np.zeros((height, width), dtype=np.uint8)
            noise_2d[y_start:y_start+block_size, x_start:x_start+block_size] = 255
            X_noise[:, i] = noise_2d.ravel()
            # Flatten the array back to 1D and store back in X
            X[:, i] = img_2d.ravel()
        return X_noise, X
    
    def add_salt_and_pepper_noise(self, X_hat, noise_level=0.02, salt_ratio=0.5) -> (np.ndarray, np.ndarray):
        """
        Add "salt and pepper" noise to data.

        Parameters:
        - X_hat (numpy array): Original data.
        - amount (float): Proportion of image pixels to be replaced.
        - salt_ratio (float): Proportion of replaced pixels that are "salt".

        Returns:
        - Numpy array of salt and pepper noise.
        - Numpy array with added salt and pepper noise.
        """
        # Initialize parameters
        X = X_hat.copy()
        X_noise = np.zeros_like(X)
        # Get the total number of pixels that should be replaced by noise
        total_pixels = X.size
        num_noise_pixels = int(total_pixels * noise_level)
        # Separate the number of salt and pepper pixels based on the salt_ratio
        num_salt = int(num_noise_pixels * salt_ratio)
        num_pepper = num_noise_pixels - num_salt
        # Directly generate the noise coordinates without overlap
        noise_coords = self.rng.choice(total_pixels, num_noise_pixels, replace=False)
        salt_coords = noise_coords[:num_salt]
        pepper_coords = noise_coords[num_salt:]
        # Convert the 1D noise coordinates back to tuple of N-dim coordinates
        salt_coords = np.unravel_index(salt_coords, X.shape)
        pepper_coords = np.unravel_index(pepper_coords, X.shape)
        # Set salt and pepper pixels in the image
        max_pixel_val = np.max(X)
        X_noise[salt_coords] = max_pixel_val
        X_noise[pepper_coords] = 0
        X[salt_coords] = max_pixel_val
        X[pepper_coords] = 0
        return X_noise, X