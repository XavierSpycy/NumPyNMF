import os
import csv
import logging
from typing import Union, List

import numpy as np
import pandas as pd

from algorithm.datasets import load_data, get_image_size
from algorithm.preprocess import NoiseAdder, MinMaxScaler, StandardScaler
from algorithm.sample import random_sample
from algorithm.nmf import BasicNMF, L2NormNMF, KLDivergenceNMF, ISDivergenceNMF, L21NormNMF, HSCostNMF, L1NormRegularizedNMF, CappedNormNMF, CauchyNMF
from algorithm.user_evaluate import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def summary(log_file_name: str) -> pd.DataFrame:
    df = pd.read_csv(log_file_name)
    result = df.groupby(by=['dataset', 'noise_type', 'noise_level'])[['rmse', 'nmi', 'acc']].mean()
    return result

class BasicBlock(object):
    def basic_info(self, nmf, dataset, scaler):
        nmf_dict = {
                'L2NormNMF': L2NormNMF,
                'KLDivergenceNMF': KLDivergenceNMF,
                'ISDivergenceNMF': ISDivergenceNMF,
                'L21NormNMF': L21NormNMF,
                'HSCostNMF': HSCostNMF,
                'L1NormRegularizedNMF': L1NormRegularizedNMF,
                'CappedNormNMF': CappedNormNMF,
                'CauchyNMF': CauchyNMF
        }
        # Store NMF algorithms in a dictionary
        dataset_dict = {
                'ORL': 'data/ORL',
                'YaleB': 'data/CroppedYaleB'
        }

        scaler_dict = {
                'MinMax': MinMaxScaler(),
                'Standard': StandardScaler()
        }
        folder = dataset_dict.get(dataset, 'data/ORL')
        # Scale the data
        scaler = scaler_dict.get(scaler, 'MinMax')
        if isinstance(nmf, BasicNMF):
            nmf = nmf
        else:
             # Choose an NMF algorithm
            nmf = nmf_dict.get(nmf, L1NormRegularizedNMF)()
        return folder, scaler, nmf
    
    def load_data(self, folder, reduce=1, random_state=None):
        # Load ORL dataset
        X_hat, Y_hat = load_data(folder, reduce=reduce)
        # Randomly sample 90% of the data
        X_hat, Y_hat = random_sample(X_hat, Y_hat, 0.9, random_state=random_state)
        # Get the size of images
        img_size = get_image_size(folder)
        return X_hat, Y_hat, img_size
    
    def add_noise(self, X_hat, noise_type, noise_level, random_state, reduce=1):
        # set random state and noise adder
        noise_adder = NoiseAdder(random_state=random_state)
        noise_dict = {
                'uniform': (noise_adder.add_uniform_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'gaussian': (noise_adder.add_gaussian_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'laplacian': (noise_adder.add_laplacian_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'salt_and_pepper': (noise_adder.add_salt_and_pepper_noise, {'X_hat': X_hat, 'noise_level': noise_level}),
                'block': (noise_adder.add_block_noise, {'X_hat': X_hat, 'block_size': noise_level, 'img_width': self.img_size[0]//reduce})
        }
        noise_func, args = noise_dict.get(noise_type, (noise_adder.add_uniform_noise, {'X_hat': X_hat, 'noise_level': noise_level}))
        _, X_noise = noise_func(**args)
        return X_noise
    
    def scale(self, X_hat, X_noise, scaler):
        # Scale the data
        X_hat_scaled = scaler.fit_transform(X_hat)
        X_noise_scaled = scaler.transform(X_noise)
        X_noise_scaled += np.abs(np.min(X_noise_scaled)) * np.abs(np.min(X_noise_scaled)) * int(np.min(X_noise_scaled) < 0)
        return X_hat_scaled, X_noise_scaled

class Pipeline(BasicBlock):
    def __init__(self, nmf: Union[str, BasicNMF], dataset: str='ORL', reduce: int=1, noise_type: str='uniform', 
                 noise_level: float=0.02, random_state: int=3407, scaler: str='MinMax') -> None:
        """
        Initialize the pipeline.

        Parameters:
        - nmf (str or BasicNMF): Name of the NMF algorithm to use.
        - dataset (str): Name of the dataset to use.
        - reduce (int): Factor by which the image size is reduced for visualization.
        - noise_type (str): Type of noise to add to the data.
        - noise_level (float): Level of noise to add to the data.
        - random_state (int): Random state to use for the NMF algorithm.
        - scaler (str): Name of the scaler to use for scaling the data.

        Returns:
        None. The function will initialize the pipeline.
        """
        folder, scaler, self.nmf = self.basic_info(nmf, dataset, scaler)
        X_hat, self.__Y_hat, self.img_size = self.load_data(folder, reduce=reduce, random_state=random_state)
        X_noise = self.add_noise(X_hat, noise_type, noise_level, random_state, reduce)
        self.__X_hat_scaled, self.__X_noise_scaled = self.scale(X_hat, X_noise, scaler)
        self.reduce = reduce
        self.random_state = random_state
        del X_hat, X_noise, folder, scaler, noise_type, noise_level, random_state, dataset, reduce, nmf

    def execute(self, max_iter, convergence_trend=False, matrix_size=False, verbose=False) -> None:
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
        # Run NMF
        self.nmf.fit(self.__X_noise_scaled, len(set(self.__Y_hat)), max_iter=max_iter, 
                     random_state=self.random_state, imshow=convergence_trend, verbose=verbose)
        # Get the dictionary and representation matrices
        self.D, self.R = self.nmf.D, self.nmf.R
        if matrix_size:
            print('D.shape={}, R.shape={}'.format(self.D.shape, self.R.shape))
        self.metrics = self.nmf.evaluate(self.__X_hat_scaled, self.__Y_hat, random_state=self.random_state)
        return self.metrics

    def evaluate(self, idx=2, imshow=False):
        evaluate(self.nmf, self.metrics, self.__X_hat_scaled, self.__X_noise_scaled, 
                self.img_size, self.reduce, idx, imshow)

    def visualization(self, idx=2):
        DR = np.dot(self.D, self.R).reshape(self.__X_hat_scaled.shape[0], self.__X_hat_scaled.shape[1])
        # Calculate reduced image size based on the 'reduce' factor
        img_size = [i//self.reduce for i in self.img_size]
        
        # Retrieve the specified image from the data
        X_i = self.__X_hat_scaled[:,idx].reshape(img_size[1],img_size[0])
        X_noise_i = self.__X_noise_scaled[:,idx].reshape(img_size[1],img_size[0])
        DR_i = DR[:,idx].reshape(img_size[1],img_size[0])
        return X_i, X_noise_i, DR_i
    
    def cleanup(self):
        """
        Cleanup method to release resources and delete instances.
        """
        # Delete attributes that might occupy significant memory
        if hasattr(self, 'nmf'):
            del self.nmf, self.__X_hat_scaled, self.__X_noise_scaled, self.D, self.R, self.metrics


def task_params_generator():
            noises = {
                'uniform': [0.1, 0.3],
                'gaussian': [0.05, 0.08],
                'laplacian': [0.04, 0.06],
                'salt_and_pepper': [0.02, 0.1],
                'block': [10, 15]
                }
            for noise_type in noises:
                for noise_level in noises[noise_type]:
                    yield noise_type, noise_level

class Experiment:
    data_dirs = ['data/ORL', 'data/CroppedYaleB']
    data_container = [[], []]
    noises = {
        'uniform': [0.1, 0.3],
        'gaussian': [0.05, 0.08],
        'laplacian': [0.04, 0.06],
        'salt_and_pepper': [0.02, 0.1],
        'block': [10, 15],}
    
    nmf_dict = {
        'L2NormNMF': L2NormNMF,
        'KLDivergenceNMF': KLDivergenceNMF,
        'ISDivergenceNMF': ISDivergenceNMF,
        'L21NormNMF': L21NormNMF,
        'HSCostNMF': HSCostNMF,
        'L1NormRegularizedNMF': L1NormRegularizedNMF,
        'CappedNormNMF': CappedNormNMF,
        'CauchyNMF': CauchyNMF,}
    
    def __init__(self, 
                 seeds: List[int]=None):
        self.seeds = [0, 42, 99, 512, 3407] if seeds is None else seeds

    def choose(self, nmf: Union[str, BasicNMF]):
        if isinstance(nmf, BasicNMF):
            self.nmf = nmf
        else:
             # Choose an NMF algorithm
            self.nmf = self.nmf_dict.get(nmf, L1NormRegularizedNMF)()

    def data_loader(self):
        scaler = MinMaxScaler()
        for data_file in self.data_dirs:
            reduce = 1 if data_file.endswith('ORL') else 3
            image_size = get_image_size(data_file)
            X_hat_, Y_hat_ = load_data(root=data_file, reduce=reduce)
            for seed in self.seeds:
                noise_adder = NoiseAdder(random_state=seed)
                X_hat, Y_hat = random_sample(X_hat_, Y_hat_, 0.9, random_state=seed)
                X_hat_scaled = scaler.fit_transform(X_hat)
                for noise_type in self.noises:
                    add_noise_ = getattr(noise_adder, f'add_{noise_type}_noise')
                    for noise_level in self.noises[noise_type]:
                        _, X_noise = add_noise_(X_hat, noise_level=noise_level) if noise_type != 'block' else add_noise_(X_hat, image_size[0]//reduce, noise_level)
                        X_noise_scaled = scaler.transform(X_noise)
                        X_noise_scaled += np.abs(np.min(X_noise_scaled)) * np.abs(np.min(X_noise_scaled)) * int(np.min(X_noise_scaled) < 0)
                        yield data_file.split("/")[-1], seed, X_hat_scaled, Y_hat, X_noise_scaled, noise_type, noise_level
    
    def sync_fit(self, dataset, seed, X_hat_scaled, Y_hat, X_noise_scaled, noise_type, noise_level):
        self.nmf.fit(X_noise_scaled, len(set(Y_hat)), random_state=seed, verbose=False)
        logging.info(f'Dataset: {dataset} Random seed: {seed} - Test on {noise_type} with {noise_level} ended.')
        return dataset, noise_type, noise_level, seed, *self.nmf.evaluate(X_hat_scaled, Y_hat, random_state=seed)
    
    def execute(self):
        import multiprocessing
        results = []

        with multiprocessing.Pool(10) as pool:
            for result in pool.starmap(self.sync_fit, self.data_loader()):
                results.append(result)

        if not os.path.exists(f'{self.nmf.name}_log.csv'):
            mode = 'w'
        else:
            mode = 'a'
        with open(f'{self.nmf.name}_log.csv', mode) as f:
            writer = csv.writer(f)
            if mode == 'w': 
                writer.writerow(['dataset', 'noise_type', 'noise_level', 'seed', 'rmse', 'acc', 'nmi'])
            for result in results:
                writer.writerow(result)