from typing import Union

import numpy as np
import pandas as pd

from algorithm.datasets import load_data, get_image_size
from algorithm.preprocess import NoiseAdder, MinMaxScaler, StandardScaler
from algorithm.sample import random_sample
from algorithm.nmf import BasicNMF, L2NormNMF, KLDivergenceNMF, ISDivergenceNMF, L21NormNMF, HSCostNMF, L1NormRegularizedNMF, CappedNormNMF, CauchyNMF
from algorithm.user_evaluate import evaluate

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

class Experiment(BasicBlock):
    def __init__(self, seeds=[0, 42, 99, 512, 3407]):
        self.seeds = seeds

    def one_type_one_level(self, nmf, dataset, reduce, noise_type, noise_level, scaler_='MinMax', max_iter=500, summary_only=True):
        df = pd.DataFrame(columns=['dataset', 'noise type', 'noise level', 'seed', 'rmse', 'acc', 'nmi'])
        for seed in self.seeds:
            pipeline = Pipeline(nmf, dataset, reduce, noise_type, noise_level, seed, scaler_)
            rmse, acc, nmi = pipeline.execute(max_iter=max_iter)
            pipeline.cleanup()
            del pipeline
            result = pd.DataFrame({'dataset': dataset, 'noise type': noise_type, 'noise level': noise_level, 
                                   'seed': seed, 'rmse': rmse, 'acc': acc, 'nmi': nmi}, index=[0])
            df = pd.concat([df, result], ignore_index=True)
        avg = pd.DataFrame({'dataset': dataset, 'noise type': noise_type, 'noise level': noise_level,
                            'seed': 'avg', 'rmse': df.rmse.mean(), 'acc': df.acc.mean(), 'nmi': df.nmi.mean()}, index=[0])
        std = pd.DataFrame({'dataset': dataset, 'noise type': noise_type, 'noise level': noise_level,
                            'seed': 'std', 'rmse': df.rmse.std(), 'acc': df.acc.std(), 'nmi': df.nmi.std()}, index=[0])
        if summary_only:
            df = pd.concat([avg, std], ignore_index=True)
        else:
            df = pd.concat([df, avg, std], ignore_index=True)
        df[['rmse', 'acc', 'nmi']] = df[['rmse', 'acc', 'nmi']].round(4)
        return df
    
    def one_type_multi_levels(self, nmf, dataset, reduce, noise_type, scaler_='MinMax', max_iter=500, summary_only=True):
        df = pd.DataFrame(columns=['dataset', 'noise type', 'noise level', 'seed', 'rmse', 'acc', 'nmi'])
        if noise_type == 'uniform':
            noise_levels = [0.1, 0.3]
        elif noise_type == 'gaussian':
            noise_levels = [0.05, 0.08]
        elif noise_type == 'laplacian':
            noise_levels = [0.04, 0.06]
        elif noise_type == 'salt_and_pepper':
            noise_levels = [0.02, 0.10]
        elif noise_type == 'block':
            noise_levels = [10, 15]
        for noise_level in noise_levels:
            print(f'Running with {noise_level} level...')
            result = self.one_type_one_level(nmf, dataset, reduce, noise_type, noise_level, scaler_, max_iter, summary_only)
            df = pd.concat([df, result], ignore_index=True)
        return df
    
    def multi_types_multi_levels(self, nmf, dataset, reduce, scaler_='MinMax', max_iter=500, summary_only=True):
        df = pd.DataFrame(columns=['dataset', 'noise type', 'noise level', 'seed', 'rmse', 'acc', 'nmi'])
        noise_types = ['uniform', 'gaussian', 'laplacian', 'salt_and_pepper', 'block']
        for noise_type in noise_types:
            print(f'{noise_type} noise:')
            result = self.one_type_multi_levels(nmf, dataset, reduce, noise_type, scaler_, max_iter, summary_only)
            df = pd.concat([df, result], ignore_index=True)
        return df
    
    def multi_datasets(self, nmf, scaler_='MinMax', max_iter=500, summary_only=True):
        df = pd.DataFrame(columns=['dataset', 'noise type', 'noise level', 'seed', 'rmse', 'acc', 'nmi'])
        datasets = ['ORL', 'YaleB']
        for dataset in datasets:
            print(f'{dataset} dataset:')
            if dataset == 'ORL':
                reduce = 1
            elif dataset == 'YaleB':
                reduce = 3
            result = self.multi_types_multi_levels(nmf, dataset, reduce, scaler_, max_iter, summary_only)
            df = pd.concat([df, result], ignore_index=True)
        print('Done!')
        return df