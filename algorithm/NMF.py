import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Union, Dict
from algorithm.initialization import NICA
from algorithm.trainer import conditional_tqdm

class BasicNMF(object):
    """
    A basic framework for Non-negative Matrix Factorization (NMF) algorithms.
    """
    def __init__(self) -> None:
        """
        Initialize the basic NMF algorithm.
        """
        self.loss_list = []

    def matricx_init(self, X: np.ndarray, n_components: int, 
                     random_state: Union[int, np.random.RandomState, None]=None, verbose: bool=False) -> None:
        """
        Initialize D and R matrices using NICA algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - n_components (int): The number of components for matrix factorization.
        - random_state (int, np.random.RandomState, None): Random state for reproducibility.
        - verbose (bool, optional): Whether to print convergence information. Default is False.
        """
        # Record start time
        start_time = time.time()
        # Intialize
        self.D, self.R = NICA(X, n_components, random_state=random_state)
        # Copy D and R matrices for convergence check
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        if verbose:
            print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')

    def fit(self, X: np.ndarray, n_components: int, max_iter: int=5000, 
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
        # Initialize D and R matrices using NICA algorithm
        self.matricx_init(X, n_components, random_state, verbose)
        # Iteratively update D and R matrices until convergence
        for _ in conditional_tqdm(range(max_iter), verbose=verbose):
            # Update D and R matrices
            flag = self.update(X, kwargs)
            # Check convergence
            if flag:
                if verbose:
                    print('Converged at iteration', _)
                break
        if imshow:
            self.plot()

    def plot(self) -> None:
        plt.plot(self.loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

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

class L1NormNMF(BasicNMF):
    """
    L1-norm NMF algorithm.
    """
    def __init__(self, FISTA: bool=False) -> None:
        """
        Initialize the L1-norm NMF algorithm.

        Parameters:
        - FISTA (bool, optional): Whether to use FISTA update. Default is False.
        """
        super().__init__()
        self.FISTA = FISTA
        if FISTA:
            self.t_prev = 1

    def clip_gradient(self, gradient: np.ndarray, max_value: float) -> np.ndarray:
        """
        Clip the gradient to avoid explosion.
        
        Parameters:
        - gradient (numpy.ndarray): Gradient matrix of shape (n_features, n_samples).
        - max_value (float): Maximum value for gradient clipping.
        """
        norm = np.linalg.norm(gradient, 2)
        if norm > max_value:
            gradient = gradient / norm * max_value
        return gradient
    
    def update(self, X: np.ndarray, kwargs: Dict[str, float]) -> None:
        """
        Update rule for D and R matrices using L1-norm NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - alpha (float, optional): Learning rate for gradient descent. Default is 1e-6.
        - threshold (float, optional): Convergence threshold based on L1-norm. Default is 1e-8.
        - clip_grad_max (float, optional): Maximum value for gradient clipping. Default is 1e7.
        """
        # Get parameters from kwargs
        alpha = kwargs.get('alpha', 1e-3)
        threshold = kwargs.get('threshold', 1e-8)
        clip_grad_max = kwargs.get('clip_grad_max', 1e7)
        # Gradient with respect to the smooth part of the objective
        residual = X - np.dot(self.D, self.R)
        grad_D = - np.dot(residual, self.R.T)
        grad_R = - np.dot(self.D.T, residual)
        # Clip gradients using the inner function
        grad_D = self.clip_gradient(grad_D, max_value=clip_grad_max)
        grad_R = self.clip_gradient(grad_R, max_value=clip_grad_max)
        # Proximal step
        # For D
        update_D = self.D - alpha * grad_D
        self.D = np.sign(update_D) * np.maximum(0, np.abs(update_D) - alpha)
        # For R
        update_R = self.R - alpha * grad_R
        R = np.sign(update_R) * np.maximum(0, np.abs(update_R) - alpha)
        # Ensuring non-negativity of D and R after updates
        self.D = np.abs(self.D)
        self.R = np.abs(self.R)
        # Calculate the loss function
        loss = np.sum(np.abs(X - np.dot(self.D, self.R)))
        self.loss_list.append(loss)
        # FISTA update
        if self.FISTA:
            self.t = (1 + np.sqrt(1 + 4 * self.t_prev ** 2)) / 2
            self.D += (self.t_prev - 1) / self.t * (self.D - self.D_prev)
            self.R += (self.t_prev - 1) / self.t * (self.R - self.R_prev)
            self.t_prev = self.t
        # Calculate L1-norm based errors for convergence
        e_D = np.sum(np.abs(self.D - self.D_prev)) / self.D.size
        e_R = np.sum(np.abs(self.R - self.R_prev)) / self.R.size
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
        DR_R_neg = 1 / (np.dot(self.D, self.R) + epsilon)
        numerator_R = np.dot(self.D.T, (np.dot(np.dot(DR_R_neg, DR_R_neg.T), X)))
        denominator_R = np.dot(self.D.T, DR_R_neg)
        numerator_R += 2 * lambd * self.R
        denominator_R += 2 * lambd * self.R
        update_factor_R = numerator_R / (denominator_R + epsilon)
        update_factor_R = self.clip_by_norm(update_factor_R, 1e+3, epsilon)
        self.R *= update_factor_R
        self.R = self.clip_matrix(self.R, 1e+6, 1e-6, epsilon)

        # Update D
        DR_D_neg = 1 / (np.dot(self.D, self.R) + epsilon)
        numerator_D = np.dot(np.dot(np.dot(DR_D_neg, DR_D_neg.T), X), self.R.T)
        denominator_D = np.dot(DR_D_neg, self.R.T)
        numerator_D += 2 * lambd * self.D
        denominator_D += 2 * lambd * self.D
        update_factor_D = numerator_D / (denominator_D + epsilon)
        update_factor_D = self.clip_by_norm(update_factor_D, 1e+3, epsilon)
        self.D *= update_factor_D
        self.D = self.clip_matrix(self.D, 1e+6, 1e-6, epsilon)

        # Normalize columns of D and rows of R
        norms = np.linalg.norm(self.D, axis=0)
        non_zero_cols = norms > epsilon
        self.D[:, non_zero_cols] /= norms[non_zero_cols][np.newaxis, :]
        self.R[non_zero_cols, :] *= norms[non_zero_cols][:, np.newaxis]

        DR = np.dot(self.D, self.R) + epsilon
        is_div = np.sum(-np.log(np.maximum(epsilon, X / DR)) + X / DR - 1)
        # Adding L2 regularization terms to the IS-divergence
        is_div += lambd * np.linalg.norm(self.D, 'fro') ** 2 + lambd * np.linalg.norm(self.R, 'fro')**2
        self.loss_list.append(is_div)
        flag = np.abs(is_div - self.prev_is_div) < threshold
        self.prev_is_div = is_div
        return flag
    
    def clip_by_norm(self, array, max_norm, epsilon):
        """Clip the array based on its norm to avoid large values."""
        norm = scipy.linalg.norm(array, 2)
        if norm > max_norm:
            array = array + epsilon * np.eye(*array.shape)
        return array

    def clip_matrix(self, matrix, max_norm, min_norm, epsilon):
        """Clip the matrix based on its norm to avoid large values or small values."""
        norm = scipy.linalg.norm(matrix, 'fro')
        if norm > max_norm:
            matrix *= max_norm / (norm + epsilon)
        if norm < min_norm:
            matrix *= min_norm / (norm + epsilon)
        return matrix

class RobustNMF(BasicNMF):
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
        # Normalize columns of D and rows of R
        norms = np.sqrt(np.sum(self.D**2, axis=0))
        self.D /= norms[np.newaxis, :] + epsilon
        self.R *= norms[:, np.newaxis]

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R) - S, 'fro') ** 2 + lambd * np.sum(np.abs(S))
        self.loss_list.append(loss)

        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((self.D - self.D_prev) ** 2, axis=(0, 1))) / self.D.size
        e_R = np.sqrt(np.sum((self.R - self.R_prev) ** 2, axis=(0, 1))) / self.R.size
        # Update previous matrices for next iteration
        self.D_prev, self.R_prev = self.D.copy(), self.R.copy()
        return (e_D < threshold and e_R < threshold)

class CauchyNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
    
    def calculate_rule(self, A, B, epsilon):
        """Update rule for Cauchy divergence."""
        return B / (A + np.sqrt(A**2 + 2 * B * A + epsilon) + epsilon)

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
        self.D *= self.calculate_rule(A, B, epsilon)
        # Update rule for R
        DR = np.dot(self.D, self.R)
        A = 3 / 4 * np.dot(self.D.T, (DR / (DR ** 2 + X + epsilon)))
        B = np.dot(self.D.T, 1 / (DR + epsilon))
        self.R *= self.calculate_rule(A, B, epsilon)
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

    def update(self, X, kwargs):
        """
        Update rule for D and R matrices using Capped Norm NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - theta (float, optional): Outlier parameter. Default is 0.2.
        - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        """
        theta = kwargs.get('theta', 0.7)
        threshold = kwargs.get('threshold', 1e-8)
        epsilon = kwargs.get('epsilon', 1e-7)
        if not hasattr(self, 'I'):
            self.n_samples = X.shape[1]
            self.I = np.eye(self.n_samples)
        # Multiplicative update rule for D and R matrices
        self.D *= np.dot(np.dot(X, self.I), self.R.T) / (np.dot(np.dot(np.dot(self.D, self.R), self.I), self.R.T) + epsilon)
        self.R *= np.sqrt(np.dot(np.dot(self.D.T, X), self.I) / (np.dot(np.dot(np.dot(np.dot(self.D.T, X), self.R.T), self.R), self.I) + epsilon) + epsilon)

        # Update rule for I
        diff = X - np.dot(self.D, self.R)
        norms = np.linalg.norm(diff, axis=0)
        for j in range(self.n_samples):
            if norms[j] < theta:
                self.I[j, j] = 1 / (2 * norms[j])
            else:
                self.I[j, j] = 0

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(self.D, self.R), 'fro') ** 2
        self.loss_list.append(loss)
        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((self.D - self.D_prev) ** 2, axis=(0, 1))) / self.D.size
        e_R = np.sqrt(np.sum((self.R - self.R_prev) ** 2, axis=(0, 1))) / self.R.size
        return (e_D < threshold and e_R < threshold)

class HypersurfaceNMF(BasicNMF):
    def __init__(self) -> None:
        super().__init__()
        self.i = 0
    
    def update(self, X, kwargs):
        """
        Update rule for D and R matrices using Hypersurface NMF algorithm.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        - beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        - alpha (float, optional): Learning rate for gradient descent. Default is 1e-3.
        - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
        - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-8.
        """
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        self.alpha = kwargs.get('alpha', 1e-3)
        epsilon = kwargs.get('epsilon', 1e-7)
        threshold = kwargs.get('threshold', 1e-8)

        self.D = np.abs(self.D)
        self.R = np.abs(self.R)
        if not hasattr(self, 'm_D'):
            self.m_D, self.v_D = np.zeros_like(self.D), np.zeros_like(self.D)
            self.m_R, self.v_R = np.zeros_like(self.R), np.zeros_like(self.R)
            self.loss_prev = np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro')) - 1
        # Gradient with respect to the smooth part of the objective
        grad_D = - (np.dot(np.dot(self.D, self.R), self.R.T) - np.dot(X, self.R.T)) / (np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro')) + epsilon)
        grad_R = - (np.dot(self.D.T, np.dot(self.D, self.R)) - np.dot(self.D.T, X)) / (np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro')) + epsilon)

        # Adam update for D
        self.m_D = beta1 * self.m_D + (1 - beta1) * grad_D
        self.v_D = beta2 * self.v_D + (1 - beta2) * (grad_D ** 2)
        m_D_corr = self.m_D / (1 - beta1 ** (self.i + 1))
        v_D_corr = self.v_D / (1 - beta2 ** (self.i + 1))
        self.D -= self.alpha * m_D_corr / (np.sqrt(v_D_corr) + epsilon)

        # Adam update for R
        self.m_R = beta1 * self.m_R + (1 - beta1) * grad_R
        self.v_R = beta2 * self.v_R + (1 - beta2) * (grad_R ** 2)
        m_R_corr = self.m_R / (1 - beta1 ** (self.i + 1))
        v_R_corr = self.v_R / (1 - beta2 ** (self.i + 1))
        self.R -= self.alpha * m_R_corr / (np.sqrt(v_R_corr) + epsilon)

        # Calculate loss
        loss_current = np.sqrt(1 + np.linalg.norm(X - np.dot(self.D, self.R), 'fro')) - 1
        self.loss_list.append(loss_current)

        flag = abs(loss_current - self.loss_prev) < threshold
        # Update previous loss for next iteration 
        self.loss_prev = loss_current
        # Update iteration number
        self.i += 1
        # Adjust learning rate
        if self.i % 100 == 0:
            self.alpha *= 0.1
        return flag