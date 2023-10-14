# nmf, random_state, noise_type, max_iter
import numpy as np
from algorithm.datasets import load_data, get_image_size
from algorithm.preprocess import NoiseAdder, MinMaxScaler, StandardScaler
from algorithm.sample import random_sample
from algorithm.NMF import L2NormNMF, L1NormNMF, KLdivergenceNMF, ISdivergenceNMF, RobustNMF, HypersurfaceNMF, L1NormRegularizedNMF, CappedNormNMF, CauchyNMF
from algorithm.user_evaluate import evaluate

class Pipeline:
    def __init__(self, nmf: str, dataset: str='ORL', reduce: int=1, noise_type: str='uniform', noise_level: float=0.02, random_state: int=3407, scaler: str='MinMax') -> None:
        """
        Initialize the pipeline.

        Parameters:
        - nmf (str): Name of the NMF algorithm to use.
        - dataset (str): Name of the dataset to use.
        - reduce (int): Factor by which the image size is reduced for visualization.
        - noise_type (str): Type of noise to add to the data.
        - noise_level (float): Level of noise to add to the data.
        - random_state (int): Random state to use for the NMF algorithm.
        - scaler (str): Name of the scaler to use for scaling the data.

        Returns:
        None. The function will initialize the pipeline.
        """
        # Store NMF algorithms in a dictionary
        self.nmf_dict = {
                'L2NormNMF': L2NormNMF,
                'L1NormNMF': L1NormNMF,
                'KLdivergenceNMF': KLdivergenceNMF,
                'ISdivergenceNMF': ISdivergenceNMF,
                'RobustNMF': RobustNMF,
                'HypersurfaceNMF': HypersurfaceNMF,
                'L1NormRegularizedNMF': L1NormRegularizedNMF,
                'CappedNormNMF': CappedNormNMF,
                'CauchyNMF': CauchyNMF
        }

        dataset_dict = {
                'ORL': 'data/ORL',
                'YaleB': 'data/CroppedYaleB'
        }

        scaler_dict = {
                'MinMax': MinMaxScaler(),
                'Standard': StandardScaler()
        }

        folder = dataset_dict.get(dataset, 'data/ORL')
        # Load ORL dataset
        X_hat, self.__Y_hat = load_data(folder, reduce=reduce)
        # Randomly sample 90% of the data
        X_hat, self.__Y_hat = random_sample(X_hat, self.__Y_hat, 0.9, random_state=random_state)
        # Get the size of images
        self.img_size = get_image_size(folder)
        # set random state and noise adder
        noise_adder = NoiseAdder(random_state=random_state)

        # Scale the data
        scaler = scaler_dict.get(scaler, 'MinMax')
        self.__X_hat_scaled = scaler.fit_transform(X_hat)
        # Add noise
        noise_dict = {
                'uniform': (noise_adder.add_uniform_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'gaussian': (noise_adder.add_gaussian_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'laplacian': (noise_adder.add_laplacian_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'salt_and_pepper': (noise_adder.add_salt_and_pepper_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'block': (noise_adder.add_block_noise, {'X_hat': X_hat, 'block_size': noise_level, 'img_width': self.img_size[0]//reduce})
        }
        noise_func, args = noise_dict.get(noise_type, (noise_adder.add_uniform_noise, {'X': X_hat, 'noise_level': noise_level}))
        _, X_noise = noise_func(**args)
        X_noise_scaled = scaler.transform(X_noise)
        self.__X_noise_scaled = X_noise_scaled + np.abs(np.min(X_noise_scaled)) * np.abs(np.min(X_noise_scaled)) * int(np.min(X_noise_scaled) < 0)
        self.nmf = nmf
        self.reduce = reduce
        self.random_state = random_state

    def run(self, max_iter, convergence_trend=False, matrix_size=False, verbose=False) -> None:
        """
        Run the pipeline.

        Parameters:
        - max_iter (int): Maximum number of iterations to run the NMF algorithm.
        - convergence_trend (bool): Whether to display the convergence trend of the NMF algorithm.
        - matrix_size (bool): Whether to display the size of the basis and coefficient matrices.
        - verbose (bool): Whether to display the verbose output of the NMF algorithm.

        Returns:
        None. The function will run the pipeline.
        """
        # Choose an NMF
        nmf = self.nmf_dict.get(self.nmf, 'L1NormRegularizedNMF')
        # Run NMF
        self.D, self.R = nmf(self.__X_noise_scaled, len(set(self.__Y_hat)), max_iter=max_iter, random_state=self.random_state, imshow=convergence_trend, verbose=verbose)
        if matrix_size:
            print('D.shape={}, R.shape={}'.format(self.D.shape, self.R.shape))

    def evaluate(self, idx=2, imshow=False):
        evaluate(self.__X_hat_scaled, self.D, self.R, self.__Y_hat, self.__X_noise_scaled, image_size=self.img_size, reduce=self.reduce, random_state=self.random_state, idx=idx, imshow=imshow)