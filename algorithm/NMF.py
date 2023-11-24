import time
import numpy as np
from scipy.linalg import pinv
from sklearn.cluster import KMeans, BisectingKMeans
from collections import Counter
from sklearn.metrics import mean_squared_error,  accuracy_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple
from tqdm import tqdm

class BasicNMF(object):
    """
    A basic framework for Non-negative Matrix Factorization (NMF) algorithms.
    """
    def __init__(self) -> None:
        """
        Initialize the basic NMF algorithm.
        """
        self.loss_list = []

    def __PCA(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """
        Principal Component Analysis (PCA) for dimensionality reduction.

        Parameters:
            X (numpy.ndarray): Input dataset of shape (n_samples, n_features).
            n_components (int): Number of principal components to retain.

        Returns:
            transformed_data (numpy.ndarray): Dataset transformed into principal component space.
        """
        if n_components > X.shape[1]:
            raise ValueError("n_components must be less than or equal to the number of features")

        # Center the data
        X_centered = X - np.mean(X, axis=0)
        # Calculate the covariance matrix and its eigenvalues and eigenvectors
        cov_mat = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Projection matrix using the first n_components eigenvectors
        projection_matrix = eigenvectors[:, :n_components]
        # Project the data onto the new feature space
        transformed_data = np.dot(X_centered, projection_matrix)
        return transformed_data

    def __FastICA(self, X: np.ndarray, max_iter: int=200, random_state: Union[int, np.random.RandomState, None]=None) -> np.ndarray:
        """
        Implementation of FastICA algorithm to separate the independent sources 
        from mixed signals in the input data.
        
        Parameters:
        X (numpy.ndarray): Input dataset of shape (n_samples, n_features).
        max_iter (int, optional): The maximum number of iterations for the convergence of the estimation. Default is 200.
                                    
        Return:
        S (numpy.ndarray): Matrix of shape (n_samples, n_features) representing the estimated independent sources.
        """
        # Set the random state
        rng = np.random.RandomState(random_state)
        # Center the data by removing the mean
        X = X - np.mean(X, axis=1, keepdims=True)
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
    
    def __NICA(self, X: np.ndarray, r: int, random_state: Union[int, np.random.RandomState, None]=None) -> (np.ndarray, np.ndarray):
        """
        Implementation of a non-negative Independent Component Analysis (NICA). 
        The process involves obtaining a non-negative basic matrix and a 
        non-negative coefficient matrix from the input data.

        Parameters:
        - X (numpy.ndarray): The input data matrix of shape (n_features, n_samples) 
                            where n_samples is the number of samples, and n_features 
                            is the number of features.
        - r (int): The number of components to be retained after applying PCA.

        Returns:
        - W_0 (numpy.ndarray): The non-negative dictionary matrix.
        - H_0 (numpy.ndarray): The non-negative representation matrix.
        """
        # Set A as a pseudoinverse of X
        A = pinv(X.T)
        # Apply PCA on the matrix A to generate the basic matrix W
        W = self.__PCA(A, n_components=r)
        # Whiten the basic matrix W obtained above by using the eigenvalue decomposition of the covariance matrix of W.
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(W, rowvar=False))
        # Preallocate memory for whitened matrix
        W_whitened = np.empty_like(W)
        np.dot(W, eigenvectors, out=W_whitened)
        W_whitened /= np.sqrt(eigenvalues + 1e-5)
        # Implement ICA algorithm on the whitened matrix W and obtain the independent basic matrix W_0
        # Assuming FastICA() returns the transformed matrix
        W_0 = self.__FastICA(W_whitened, random_state=random_state)
        # Preallocate memory for H_0 and calculate it
        H_0 = np.empty((W_0.shape[1], X.shape[1]))
        np.dot(W_0.T, X, out=H_0)
        # Take the absolute value in-place
        np.abs(W_0, out=W_0)
        np.abs(H_0, out=H_0)
        return W_0, H_0
    
    def Kmeans(self, X: np.ndarray, n_components: int, random_state: Union[int, np.random.RandomState, None]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize D and R matrices using K-means algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - n_components (int): The number of components for matrix factorization.
        - random_state (int, np.random.RandomState, None): Random state for reproducibility.
        """
        # Intialize
        kmeans = KMeans(n_clusters=n_components, n_init='auto', random_state=random_state)
        kmeans.fit(X.T)
        D = kmeans.cluster_centers_.T
        labels = kmeans.labels_
        G = np.zeros(((len(labels)), n_components))
        for i, label in enumerate(labels):
            G[i, label] = 1
        G = G / np.sqrt(np.sum(G, axis=0, keepdims=True))
        G += 0.2
        R = G.T
        return D, R
    
    def matrix_init(self, X: np.ndarray, n_components: int, 
                     random_state: Union[int, np.random.RandomState, None]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize D and R matrices using NICA algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - n_components (int): The number of components for matrix factorization.
        - random_state (int, np.random.RandomState, None): Random state for reproducibility.
        
        Returns:
        - D (numpy.ndarray): The non-negative dictionary matrix.
        - R (numpy.ndarray): The non-negative representation matrix.
        """
        # Intialize
        D, R = self.__NICA(X, n_components, random_state=random_state)
        return D, R
        
    def fit(self, X: np.ndarray, n_components: int, max_iter: int=500, 
            random_state: Union[int, np.random.RandomState, None]=None, 
            verbose: bool=True, imshow: bool=False, **kwargs) -> None:
        """
        Non-negative Matrix Factorization (NMF) algorithm using L2-norm for convergence criterion.
        
        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - n_components (int): The number of components for matrix factorization.
        - max_iter (int, optional): Maximum number of iterations. Default is 5000.
        - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
        - imshow (bool, optional): Whether to plot convergence trend. Default is False.
        """
        # Record start time
        start_time = time.time()
        # Initialize D and R matrices using NICA algorithm
        self.D, self.R = self.matrix_init(X, n_components, random_state)
        # Compute initialization time
        init_time = time.time() - start_time
        # Copy D and R matrices for convergence check
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        if verbose:
            print(f'Initialization done. Time elapsed: {init_time:.2f} seconds.')
        # Iteratively update D and R matrices until convergence
        for _ in self.conditional_tqdm(range(max_iter), verbose=verbose):
            # Update D and R matrices
            flag = self.update(X, kwargs)
            # Check convergence
            if flag:
                if verbose:
                    print('Converged at iteration', _)
                break
        if imshow:
            self.plot()

    def update(self, X: np.ndarray, kwargs: Dict[str, float]) -> None:
        threshold = kwargs.get('threshold', 1e-6)
        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((self.D - self.D_prev) ** 2, axis=(0, 1))) / self.D.size
        e_R = np.sqrt(np.sum((self.R - self.R_prev) ** 2, axis=(0, 1))) / self.R.size
        return (e_D < threshold and e_R < threshold)

    def plot(self) -> None:
        plt.plot(self.loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()
    
    def conditional_tqdm(self, iterable, verbose: bool=True) -> int:
        """
        Determine whether to use tqdm or not based on the verbose flag.

        Parameters:
        - iterable (range): Range of values to iterate over.
        - verbose (bool, optional): Whether to print progress bar. Default is True.

        Returns:
        - item (int): Current iteration.
        """
        if verbose:
            for item in tqdm(iterable):
                yield item
        else:
            for item in iterable:
                yield item
    
    def normalize(self, epsilon: float=1e-7):
        """
        Normalize columns of D and rows of R.
        """
        # Normalize columns of D and rows of R
        norms = np.sqrt(np.sum(self.D**2, axis=0))
        self.D /= norms[np.newaxis, :] + epsilon
        self.R *= norms[:, np.newaxis]
    
    def evaluate(self, X_clean, Y_true, random_state=None):
        Y_label = self.__labeling(self.R.T, Y_true, random_state=random_state)
        rmse = np.sqrt(mean_squared_error(X_clean, np.dot(self.D, self.R)))
        acc = accuracy_score(Y_true, Y_label)
        nmi = normalized_mutual_info_score(Y_true, Y_label)
        return rmse, acc, nmi

    def __labeling(self, X: np.ndarray, Y: np.ndarray, random_state: Union[int, np.random.RandomState, None]=None) -> np.ndarray:
        """
        Label data based on clusters obtained from KMeans clustering, 
        by assigning the most frequent label in each cluster.
        
        Parameters:
        - X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
        - Y (numpy.ndarray): True labels corresponding to each sample in X of shape (n_samples,).

        Returns:
        - Y_pred (numpy.ndarray): Predicted labels for each sample based on the clustering results.

        Note:
        This function works best when the input data is somewhat separated into distinct 
        clusters that align with the true labels.
        """
        cluster = BisectingKMeans(len(set(Y)), random_state=random_state).fit(X)
        Y_pred = np.zeros(Y.shape)
        for i in set(cluster.labels_):
            ind = cluster.labels_ == i
            Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
        return Y_pred
    
    def vectorized_armijo_rule(self, f, grad_f, X, alpha, c=1e-4, tau=0.5):
        """
        Vectorized Armijo rule to find the step size for each element in the matrix.

        Parameters:
        - f: The objective function, which should accept a matrix and return a scalar.
        - grad_f: The gradient of the objective function, which returns a matrix.
        - X: Current point, a matrix.
        - alpha: Initial step size, a scalar or a matrix.
        - c: A constant in (0, 1), typically a small value (default is 1e-4).
        - tau: Reduction factor for step size, typically in (0, 1) (default is 0.5).

        Returns:
        - alpha: Step sizes that satisfy the Armijo condition for each element.
        """
        # Compute the initial objective function value
        f_x = f(X)
        # Compute the initial gradient and its norm squared
        grad_f_x = grad_f(X)
        norm_grad_f_x_squared = np.square(np.linalg.norm(grad_f_x, axis=(0,1), keepdims=True))
        
        # Compute the sufficient decrease condition for the entire matrix
        sufficient_decrease = f_x - c * alpha * norm_grad_f_x_squared
        
        counter = 0
        # Check the condition for each element
        while np.any(f(X - alpha * grad_f_x) > sufficient_decrease) or counter >= 10:
            # Reduce alpha for elements not satisfying the condition
            alpha *= tau
            counter += 1
        return alpha
    
class L2NormNMF(BasicNMF):
    """
    L2-norm NMF algorithm.
    """
    def __init__(self) -> None:
        super().__init__()

    def update(self, X: np.ndarray, kwargs: Dict[str, float]) -> None:
        """
        Update rule for D and R matrices using L2-norm NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - threshold (float, optional): Convergence threshold based on L2-norm. Default is 1e-6.
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        """
        threshold = kwargs.get('threshold', 1e-6)
        epsilon = kwargs.get('epsilon', 1e-7)
        # Multiplicative update rule for D and R matrices
        self.D *= np.dot(X, self.R.T) / (np.dot(np.dot(self.D, self.R), self.R.T) + epsilon)
        self.R *= np.dot(self.D.T, X) / (np.dot(np.dot(self.D.T, self.D), self.R) + epsilon)
        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R), 'fro') ** 2
        self.loss_list.append(loss)
        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((self.D - self.D_prev) ** 2, axis=(0, 1))) / self.D.size
        e_R = np.sqrt(np.sum((self.R - self.R_prev) ** 2, axis=(0, 1))) / self.R.size
        # Update previous matrices for next iteration
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        return (e_D < threshold and e_R < threshold)

class KLDivergenceNMF(BasicNMF):
    """
    KL-divergence NMF algorithm.
    """
    def __init__(self) -> None:
        """
        Initialize the KL-divergence NMF algorithm.
        """
        super().__init__()
        self.prev_kl = float('inf')

    def update(self, X: np.ndarray, kwargs: Dict[str, float]) -> None:
        """
        Update rule for D and R matrices using KL-divergence NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        - threshold (float, optional): Convergence threshold based on KL-divergence. Default is 1e-4.
        """
        epsilon = kwargs.get('epsilon', 1e-7)
        threshold = kwargs.get('threshold', 1e-4)
        # Multiplicative update rule for D and R matrices
        self.D *= np.dot(X / (np.dot(self.D, self.R) + epsilon), self.R.T) / (np.dot(np.ones(X.shape), self.R.T) + epsilon)
        self.R *= np.dot(self.D.T, X / (np.dot(self.D, self.R) + epsilon)) / (np.dot(self.D.T, np.ones(X.shape)) + epsilon)

        # Calculate KL-divergence
        XR = np.dot(self.D, self.R) + epsilon
        kl_div = np.sum(X * np.log(np.maximum(epsilon, X / (XR + epsilon))) - X + XR)
        self.loss_list.append(kl_div)
        flag = abs(kl_div - self.prev_kl) < threshold
        self.prev_kl = kl_div  # Update previous KL divergence
        return flag

class ISDivergenceNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
        self.prev_is_div = float('inf')

    def update(self, X, kwargs):
        # Get parameters from kwargs
        epsilon = kwargs.get('epsilon', 1e-7)
        lambd = kwargs.get('lambd', 1e+2)
        threshold = kwargs.get('threshold', 1e-6)
        # Update R
        DR = np.dot(self.D, self.R)
        DR = np.where(DR > 0, DR, epsilon)
        self.R *= (np.dot(self.D.T, (DR ** (-2) * X))) / (np.dot(self.D.T, DR ** (-1)) + epsilon)
        # Update D
        DR = np.dot(self.D, self.R)
        DR = np.where(DR > 0, DR, epsilon)
        self.D *= (np.dot((DR ** (-2) * X), self.R.T)) / (np.dot(DR ** (-1), self.R.T) + epsilon)
        # Normalize D and R
        self.normalize(epsilon)
        # Calculate IS-divergence
        DR = np.dot(self.D, self.R) + epsilon
        is_div = np.sum(-np.log(np.maximum(epsilon, X / DR)) + X / DR - 1)
        # Adding L2 regularization terms to the IS-divergence
        # is_div += lambd * np.linalg.norm(self.D, 'fro') ** 2 + lambd * np.linalg.norm(self.R, 'fro')**2
        self.loss_list.append(is_div)
        flag = np.abs(is_div - self.prev_is_div) < threshold
        self.prev_is_div = is_div
        return flag

class L21NormNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
    
    def update(self, X, kwargs):
        # Get parameters from kwargs
        epsilon = kwargs.get('epsilon', 1e-7)
        threshold = kwargs.get('threshold', 1e-4)
        # Multiplicative update rule for D and R matrices
        residual = X - np.dot(self.D, self.R) # residual.shape = (n_features, n_samples)
        norm_values = np.sqrt(np.sum(residual ** 2, axis=1))
        diagonal = np.diag(1.0 / (norm_values + epsilon)) # diagonal.shape = (n_features, n_features)
        # Update rule for D
        self.D *= (np.dot(np.dot(diagonal, X), self.R.T) / (np.dot(np.dot(np.dot(diagonal, self.D), self.R), self.R.T) + epsilon))
        # Update rule for R
        self.R *= (np.dot(np.dot(self.D.T, diagonal), X) / (np.dot(np.dot(np.dot(self.D.T, diagonal), self.D), self.R) + epsilon))
        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R), 'fro')
        self.loss_list.append(loss)
        # Calculate L2,1-norm based errors for convergence
        e_D = np.linalg.norm(self.D - self.D_prev, 'fro') / np.linalg.norm(self.D, 'fro')
        e_R = np.linalg.norm(self.R - self.R_prev, 'fro') / np.linalg.norm(self.R, 'fro')
        # Update previous matrices for next iteration
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        return (e_D < threshold and e_R < threshold)
        
class L1NormRegularizedNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()

    def soft_thresholding(self, x, lambd):
        """
        Soft thresholding operator.

        Parameters:
        - x (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - lambd (float): Threshold value.
        """
        return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))
    
    def update(self, X, kwargs):
        # Get parameters from kwargs
        lambd = kwargs.get('lambd', 0.2)
        threshold = kwargs.get('threshold', 1e-8)
        epsilon = kwargs.get('epsilon', 1e-7)
        # Compute the error matrix
        S = X - np.dot(self.D, self.R)
        # Soft thresholding operator
        S = self.soft_thresholding(S, lambd/2)
        # Multiplicative update rule for D and R matrices
        update_D = np.dot(S - X, self.R.T)
        self.D *=  (np.abs(update_D) - update_D) / (2 * np.dot(np.dot(self.D, self.R), self.R.T) + epsilon)
        update_R = np.dot(self.D.T, S - X)
        self.R *=  (np.abs(update_R) - update_R) / (2 * np.dot(np.dot(self.D.T, self.D), self.R) + epsilon)
        self.normalize(epsilon)
        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R) - S, 'fro') ** 2 + lambd * np.sum(np.abs(S))
        self.loss_list.append(loss)
        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((self.D - self.D_prev) ** 2, axis=(0, 1))) / self.D.size
        e_R = np.sqrt(np.sum((self.R - self.R_prev) ** 2, axis=(0, 1))) / self.R.size
        # Update previous matrices for next iteration
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        return (e_D < threshold and e_R < threshold)
    
    def matrix_init(self, X: np.ndarray, n_components: int, 
                     random_state: Union[int, np.random.RandomState, None]=None) -> None:
        return self.Kmeans(X, n_components, random_state)

class CauchyNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
    
    def compute(self, A, B, epsilon):
        """Update rule for Cauchy divergence."""
        temp = A ** 2 + 2 * B * A
        temp = np.where(temp > 0, temp, epsilon)
        return B / (A + np.sqrt(temp))

    def update(self, X, kwargs):
        # Get parameters from kwargs
        epsilon = kwargs.get('epsilon', 1e-7)
        threshold = kwargs.get('threshold', 1e-4)
        if not hasattr(self, 'prev_cauchy_div'):
            DR = np.dot(self.D, self.R)
            log_residual = np.log(DR + epsilon) - np.log(X + epsilon)
            residual = X - DR
            self.prev_cauchy_div = np.sum(log_residual + residual / (DR + epsilon))
        # Update rule for D
        DR = np.dot(self.D, self.R)
        A = 3 / 4 * np.dot((DR / (DR ** 2 + X + epsilon)), self.R.T)
        B = np.dot(1 / (DR + epsilon), self.R.T)
        self.D *= self.compute(A, B, epsilon)
        # Update rule for R
        DR = np.dot(self.D, self.R)
        A = 3 / 4 * np.dot(self.D.T, (DR / (DR ** 2 + X + epsilon)))
        B = np.dot(self.D.T, 1 / (DR + epsilon))
        self.R *= self.compute(A, B, epsilon)
        # Calculate Cauchy divergence
        DR = np.dot(self.D, self.R)
        cauchy_div = np.sum(np.log(DR + epsilon) - np.log(X + epsilon) + (X - DR) / (DR + epsilon))
        self.loss_list.append(cauchy_div)
        flag = abs(cauchy_div - self.prev_cauchy_div) < threshold
        self.prev_cauchy_div = cauchy_div  # Update previous Cauchy divergence
        return flag

class CappedNormNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
        self.loss_prev = float('inf')
    
    def matrix_init(self, X: np.ndarray, n_components: int, 
                     random_state: Union[int, np.random.RandomState, None]=None) -> None:
        return self.Kmeans(X, n_components, random_state)
    
    def update(self, X, kwargs):
        """
        Update rule for D and R matrices using Capped Norm NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - theta (float, optional): Outlier parameter. Default is 0.2.
        - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        """
        theta = kwargs.get('theta', 0.2)
        threshold = kwargs.get('threshold', 1e-3)
        epsilon = kwargs.get('epsilon', 1e-7)
        if not hasattr(self, 'I'):
            self.n_samples = X.shape[1]
            self.I = np.identity(self.n_samples)
        # Multiplicative update rule for D and R matrices
        G = self.R.T
        self.D *= np.dot(np.dot(X, self.I), G) / (np.dot(np.dot(np.dot(self.D, G.T), self.I), G) + epsilon)
        G *= np.sqrt((np.dot(np.dot(self.I, X.T), self.D)) / (np.dot(np.dot(np.dot(np.dot(self.I, G), G.T), X.T), self.D) + epsilon))
        self.R = G.T
        # Update rule for I
        diff = X - np.dot(self.D, self.R)
        norms = np.linalg.norm(diff, axis=0)
        norms /= np.max(norms)
        I = np.full_like(norms, epsilon)
        indices = np.where(norms < theta)
        I[indices] = 1 / (2 * norms[indices])
        self.I = np.diagflat(I)
        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R), 'fro') ** 2
        flag = abs(loss - self.loss_prev) < threshold
        self.loss_list.append(loss)
        self.loss_prev = loss
        return flag

class HSCostNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
        self.loss_prev = float('inf')
        # Objective function and its gradient
        self.obj_func = lambda X, D, R: np.linalg.norm(X - np.dot(D, R), 'fro')
        self.grad_D = lambda X, D, R: (np.dot((np.dot(D, R) - X), R.T)) / np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro'))
        self.grad_R = lambda X, D, R: (np.dot(D.T, (np.dot(D, R) - X))) / np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro'))

    def update(self, X, kwargs):
        """
        Update rule for D and R matrices using Hypersurface Cost NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - alpha (float, optional): Learning rate for gradient descent. Default is 0.1.
        - beta (float, optional): Learning rate for gradient descent. Default is 0.1.
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-8.
        """
        threshold = kwargs.get('threshold', 1e-8)
        if not hasattr(self, 'alpha'):
            self.alpha = np.full_like(self.D, kwargs.get('alpha', 0.1))
            self.beta = np.full_like(self.R, kwargs.get('beta', 0.1))
        c = kwargs.get('c', 1e-4)
        tau = kwargs.get('tau', 0.5)
        # Vectorized Armijo rule to update alpha and beta
        self.alpha = self.vectorized_armijo_rule(lambda D: self.obj_func(X, D, self.R), lambda D: self.grad_D(X, D, self.R), self.D, self.alpha, c, tau)
        self.beta = self.vectorized_armijo_rule(lambda R: self.obj_func(X, self.D, R), lambda R: self.grad_R(X, self.D, R), self.R, self.beta, c, tau)
        self.alpha = np.maximum(self.alpha, threshold)
        self.beta = np.maximum(self.beta, threshold)
        # Update rule for D and R
        self.D -= self.alpha * (np.dot((np.dot(self.D, self.R) - X), self.R.T)) / np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro'))
        self.R -= self.beta * (np.dot(self.D.T, (np.dot(self.D, self.R) - X))) / np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro'))
        self.D[np.where(self.D < 0)] = 0
        self.R[np.where(self.R < 0)] = 0
        # Calculate loss
        loss_current = np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro')) - 1
        self.loss_list.append(loss_current)
        flag = abs(loss_current - self.loss_prev) < threshold
        # Update previous loss for next iteration 
        self.loss_prev = loss_current
        return flag