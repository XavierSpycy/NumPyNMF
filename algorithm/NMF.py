import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Union
from algorithm.initialization import NICA
from algorithm.trainer import conditional_tqdm

def L2NormNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, threshold: float=1e-6, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray): 
    """
    Non-negative Matrix Factorization (NMF) algorithm using L2-norm for convergence criterion.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - threshold (float, optional): Convergence threshold based on L2-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """
    
    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')

    # Copy D and R matrices for convergence check
    D_prev, R_prev = D.copy(), R.copy()
    # Initialize loss
    loss_list = []

    # Iteratively update D and R matrices until convergence
    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Multiplicative update rule for D and R matrices
        D *= np.dot(X, R.T) / (np.dot(np.dot(D, R), R.T) + epsilon)
        R *= np.dot(D.T, X) / (np.dot(np.dot(D.T, D), R) + epsilon)

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(D, R), 'fro') ** 2
        loss_list.append(loss)

        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((D - D_prev) ** 2, axis=(0, 1))) / D.size
        e_R = np.sqrt(np.sum((R - R_prev) ** 2, axis=(0, 1))) / R.size

        # Check convergence
        if e_D < threshold and e_R < threshold:
            if verbose:
                print('Converged at iteration', _)
            break
        
        # Update previous matrices for next iteration
        D_prev, R_prev = D.copy(), R.copy()
    
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

    return D, R

def L1NormNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, alpha: float=1e-6, threshold: float=1e-8, FISTA: bool=False, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Non-negative Matrix Factorization (NMF) algorithm using L1-norm for convergence criterion.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - alpha (float, optional): Learning rate for gradient descent. Default is 1e-4.
    - threshold (float, optional): Convergence threshold based on L1-norm. Default is 1e-4.
    - FISTA (bool, optional): Whether to use FISTA update. Default is False.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """
    def clip_gradient(gradient, max_value):
        """Clip the gradient to avoid explosion."""
        norm = np.linalg.norm(gradient, 2)
        if norm > max_value:
            gradient = gradient / norm * max_value
        return gradient

    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    # Copy D and R matrices for convergence check
    D_prev, R_prev = D.copy(), R.copy()
    # Initialize loss
    loss_list = []
    # Initialize previous t for FISTA update
    if FISTA:
        t_prev = 1

    # Iteratively update D and R matrices until convergence
    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Gradient with respect to the smooth part of the objective
        residual = X - np.dot(D, R)

        grad_D = - np.dot(residual, R.T)
        grad_R = - np.dot(D.T, residual)

        # Clip gradients using the inner function
        grad_D = clip_gradient(grad_D, max_value=1e7)
        grad_R = clip_gradient(grad_R, max_value=1e7)

        # Proximal step
        # For D
        update_D = D - alpha * grad_D
        D = np.sign(update_D) * np.maximum(0, np.abs(update_D) - alpha)

        # For R
        update_R = R - alpha * grad_R
        R = np.sign(update_R) * np.maximum(0, np.abs(update_R) - alpha)

        # Ensuring non-negativity of D and R after updates
        D = np.abs(D)
        R = np.abs(R)
        
        # Calculate the loss function
        loss = np.sum(np.abs(X - np.dot(D, R)))
        loss_list.append(loss)

        # FISTA update
        if FISTA:
            t = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
            D += (t_prev - 1) / t * (D - D_prev)
            R += (t_prev - 1) / t * (R - R_prev)
            t_prev = t

        # Calculate L1-norm based errors for convergence
        e_D = np.sum(np.abs(D - D_prev)) / D.size
        e_R = np.sum(np.abs(R - R_prev)) / R.size

        # Check convergence
        if e_D < threshold and e_R < threshold:
            if verbose:
                print('Converged at iteration', _)
            break

        # Update previous matrices for next iteration
        D_prev, R_prev = D.copy(), R.copy()
        
    # Plot convergence trend
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()
        
    return D, R

def KLdivergenceNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, threshold: float=1e-4, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Non-negative Matrix Factorization (NMF) algorithm using KL-divergence as the cost function.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - threshold (float, optional): Convergence threshold based on L2-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """

    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    prev_kl = float('inf')  # Initialize previous KL divergence to infinity
    kl_div_list = []  # Initialize list to store KL divergence values

    # Iteratively update D and R matrices until convergence
    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Multiplicative update rule for D and R matrices
        D *= np.dot(X / (np.dot(D, R) + epsilon), R.T) / (np.dot(np.ones(X.shape), R.T) + epsilon)
        R *= np.dot(D.T, X / (np.dot(D, R) + epsilon)) / (np.dot(D.T, np.ones(X.shape)) + epsilon)

        # Calculate KL-divergence
        XR = np.dot(D, R) + epsilon
        kl_div = np.sum(X * np.log(np.maximum(epsilon, X / (XR + epsilon))) - X + XR)
        kl_div_list.append(kl_div)

        # Check convergence
        if abs(kl_div - prev_kl) < threshold:
            if verbose:
                print('Converged at iteration', _)
            break
        
        prev_kl = kl_div  # Update previous KL divergence
    
    # Plot convergence trend
    if imshow:
        plt.plot(kl_div_list)
        plt.xlabel('Iteration')
        plt.ylabel('KL-divergence')
        plt.grid()
        plt.show()

    return D, R

def ISdivergenceNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, threshold: float=1e-6, epsilon: float=1e-7, lambd: float=1e+2, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Non-negative Matrix Factorization (NMF) algorithm using IS-divergence as the cost function.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - threshold (float, optional): Convergence threshold based on IS Divergence. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.
    """

    def clip_by_norm(array, max_norm):
        """Clip the array based on its norm to avoid large values."""
        norm = scipy.linalg.norm(array, 2)
        if norm > max_norm:
            array = array + epsilon * np.eye(*array.shape)
        return array

    def clip_matrix(matrix, max_norm, min_norm):
        """Clip the matrix based on its norm to avoid large values or small values."""
        norm = scipy.linalg.norm(matrix, 'fro')
        if norm > max_norm:
            matrix *= max_norm / (norm + epsilon)
        if norm < min_norm:
            matrix *= min_norm / (norm + epsilon)
        return matrix

    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    prev_is_div = float('inf')
    is_div_list = []

    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        assert not np.isnan(D).any() or not np.isinf(D).any()
        assert not np.isnan(R).any() or not np.isinf(R).any()

        # Update R
        DR_R_neg = 1 / (np.dot(D, R) + epsilon)
        numerator_R = np.dot(D.T, (np.dot(np.dot(DR_R_neg, DR_R_neg.T), X)))
        denominator_R = np.dot(D.T, DR_R_neg)
        numerator_R += 2 * lambd * R
        denominator_R += 2 * lambd * R
        update_factor_R = numerator_R / (denominator_R + epsilon)
        update_factor_R = clip_by_norm(update_factor_R, 1e+3)
        R *= update_factor_R
        R = clip_matrix(R, 1e+6, 1e-6)

        # Update D
        DR_D_neg = 1 / (np.dot(D, R) + epsilon)
        numerator_D = np.dot(np.dot(np.dot(DR_D_neg, DR_D_neg.T), X), R.T)
        denominator_D = np.dot(DR_D_neg, R.T)
        numerator_D += 2 * lambd * D
        denominator_D += 2 * lambd * D
        update_factor_D = numerator_D / (denominator_D + epsilon)
        update_factor_D = clip_by_norm(update_factor_D, 1e+3)
        D *= update_factor_D
        D = clip_matrix(D, 1e+6, 1e-6)

        # Normalize columns of D and rows of R
        norms = np.linalg.norm(D, axis=0)
        non_zero_cols = norms > epsilon
        D[:, non_zero_cols] /= norms[non_zero_cols][np.newaxis, :]
        R[non_zero_cols, :] *= norms[non_zero_cols][:, np.newaxis]

        DR = np.dot(D, R) + epsilon
        is_div = np.sum(-np.log(np.maximum(epsilon, X / DR)) + X / DR - 1)
        # Adding L2 regularization terms to the IS-divergence
        is_div += lambd * np.linalg.norm(D, 'fro') ** 2 + lambd * np.linalg.norm(R, 'fro')**2
        is_div_list.append(is_div)

        # Check convergence
        if np.abs(is_div - prev_is_div) < threshold:
            if verbose:
                print('Converged at iteration', _)
            break

        prev_is_div = is_div

    if imshow:
        plt.plot(is_div_list)
        plt.xlabel('Iteration')
        plt.ylabel('IS-divergence')
        plt.grid()
        plt.show()

    return D, R

def RobustNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, threshold: float=1e-4, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Robust Non-negative Matrix Factorization (NMF) using L2,1-norm for convergence criterion.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """
    
    # Initialize D and R matrices using NICA algorithm (or any other suitable initialization)
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    D_prev, R_prev = D.copy(), R.copy()
    loss_list = [] # Initialize list to store loss values

    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Multiplicative update rule for D and R matrices
        residual = X - np.dot(D, R) # residual.shape = (n_features, n_samples)
        norm_values = np.sqrt(np.sum(residual ** 2, axis=1))
        diagonal = np.diag(1.0 / (norm_values + epsilon)) # diagonal.shape = (n_features, n_features)
        
        # Update rule for D
        D *= (np.dot(np.dot(diagonal, X), R.T) / (np.dot(np.dot(np.dot(diagonal, D), R), R.T) + epsilon))

        # Update rule for R
        R *= (np.dot(np.dot(D.T, diagonal), X) / (np.dot(np.dot(np.dot(D.T, diagonal), D), R) + epsilon))

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(D, R), 'fro')
        loss_list.append(loss)

        # Calculate L2,1-norm based errors for convergence
        e_D = np.linalg.norm(D - D_prev, 'fro') / np.linalg.norm(D, 'fro')
        e_R = np.linalg.norm(R - R_prev, 'fro') / np.linalg.norm(R, 'fro')

        # Check convergence
        if e_D < threshold and e_R < threshold:
            if verbose:
                print('Converged at iteration', _)
            break
        
        # Update previous matrices for next iteration
        D_prev, R_prev = D.copy(), R.copy()
    
    # Plot convergence trend
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

    return D, R

def HypersurfaceNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, alpha: float=1e-3, threshold: float=1e-8, epsilon: float=1e-7, beta1: float=0.9, beta2: float=0.999, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Hypersurface Non-negative Matrix Factorization (NMF) using L2-norm for convergence criterion.
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - alpha (float, optional): Learning rate for gradient descent. Default is 1e-3.
    - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
    - beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """

    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')

    loss_list = [] # Initialize list to store loss values

    # Adam parameters initialization
    m_D, v_D = np.zeros_like(D), np.zeros_like(D)
    m_R, v_R = np.zeros_like(R), np.zeros_like(R)

    # Calculate initial loss
    loss_prev = np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro')) - 1

    # Iteratively update D and R matrices until convergence
    for i in conditional_tqdm(range(max_iter), verbose=verbose):
        # Gradient with respect to the smooth part of the objective
        grad_D = - (np.dot(np.dot(D, R), R.T) - np.dot(X, R.T)) / (np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro')) + epsilon)
        grad_R = - (np.dot(D.T, np.dot(D, R)) - np.dot(D.T, X)) / (np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro')) + epsilon)

        # Adam update for D
        m_D = beta1 * m_D + (1 - beta1) * grad_D
        v_D = beta2 * v_D + (1 - beta2) * (grad_D ** 2)
        m_D_corr = m_D / (1 - beta1 ** (i + 1))
        v_D_corr = v_D / (1 - beta2 ** (i + 1))
        D -= alpha * m_D_corr / (np.sqrt(v_D_corr) + epsilon)

        # Adam update for R
        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R ** 2)
        m_R_corr = m_R / (1 - beta1 ** (i + 1))
        v_R_corr = v_R / (1 - beta2 ** (i + 1))
        R -= alpha * m_R_corr / (np.sqrt(v_R_corr) + epsilon)

        # Ensuring non-negativity of D and R after updates
        # D = np.abs(D)
        # R = np.abs(R)

        # Calculate loss
        loss_current = np.sqrt(1 + np.linalg.norm(X - np.dot(D, R), 'fro')) - 1
        loss_list.append(loss_current)

        # Check convergence
        if abs(loss_current - loss_prev) < threshold:
            if verbose:
                print('Converged at iteration', i)
            break
        
        # Update previous loss for next iteration 
        loss_prev = loss_current
    
    # Ensuring non-negativity
    D = np.abs(D)
    R = np.abs(R)

    # Plot convergence trend
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

    return D, R

def L1NormRegularizedNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, lambd: float=0.2, threshold: float=1e-8, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Non-negative Matrix Factorization (NMF) algorithm using L1 Norm Regularized as the cost function.

    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - lambd (float, optional): Regularization parameter. Default is 0.2.
    - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """

    def soft_thresholding(x, lambd):
        """
        Soft thresholding operator.

        Parameters:
        - x (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
        - lambd (float): Threshold value.
        """
    
        return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))
        
    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    # Copy D and R matrices for convergence check
    D_prev, R_prev = D.copy(), R.copy()
    # Initialize loss
    loss_list = []

    # Iteratively update D and R matrices until convergence
    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Compute the error matrix
        S = X - np.dot(D, R)
        # Soft thresholding operator
        S = soft_thresholding(S, lambd/2)
        # Multiplicative update rule for D and R matrices
        update_D = np.dot(S - X, R.T)
        D *=  (np.abs(update_D) - update_D) / (2 * np.dot(np.dot(D, R), R.T) + epsilon)
        update_R = np.dot(D.T, S - X)
        R *=  (np.abs(update_R) - update_R) / (2 * np.dot(np.dot(D.T, D), R) + epsilon)
        # Normalize columns of D and rows of R
        norms = np.sqrt(np.sum(D**2, axis=0))
        D /= norms[np.newaxis, :] + epsilon
        R *= norms[:, np.newaxis]

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(D, R) - S, 'fro') ** 2 + lambd * np.sum(np.abs(S))
        loss_list.append(loss)

        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((D - D_prev) ** 2, axis=(0, 1))) / D.size
        e_R = np.sqrt(np.sum((R - R_prev) ** 2, axis=(0, 1))) / R.size

        # Check convergence
        if e_D < threshold and e_R < threshold:
            if verbose:
                print('Converged at iteration', _)
            break

        # Update previous matrices for next iteration
        D_prev, R_prev, S_prev = D.copy(), R.copy(), S.copy()

    # Plot convergence trend
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

    return D, R

def CappedNormNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, theta: float=0.7, threshold: float=1e-8, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    Non-negative Matrix Factorization (NMF) algorithm using Capped Norm as the cost function.

    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - theta (float, optional): Outlier parameter. Default is 0.2.
    - threshold (float, optional): Convergence threshold based on L2,1-norm. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """

    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    # Copy D and R matrices for convergence check
    D_prev, R_prev = D.copy(), R.copy()
    # Initialize Identity matrix
    n_samples = X.shape[1]
    I = np.eye(n_samples)
    # Initialize loss
    loss_list = []

    # Iteratively update D and R matrices until convergence
    for _ in conditional_tqdm(range(max_iter), verbose=verbose):
        # Multiplicative update rule for D and R matrices
        D *= np.dot(np.dot(X, I), R.T) / (np.dot(np.dot(np.dot(D, R), I), R.T) + epsilon)
        R *= np.sqrt(np.dot(np.dot(D.T, X), I) / (np.dot(np.dot(np.dot(np.dot(D.T, X), R.T), R), I) + epsilon) + epsilon)

        # Update rule for I
        diff = X - np.dot(D, R)
        norms = np.linalg.norm(diff, axis=0)
        for j in range(n_samples):
            if norms[j] < theta:
                I[j, j] = 1 / (2 * norms[j])
            else:
                I[j, j] = 0

        # Calculate the loss function
        loss = np.linalg.norm(X - np.dot(D, R), 'fro') ** 2
        loss_list.append(loss)

        # Calculate L2-norm based errors for convergence
        e_D = np.sqrt(np.sum((D - D_prev) ** 2, axis=(0, 1))) / D.size
        e_R = np.sqrt(np.sum((R - R_prev) ** 2, axis=(0, 1))) / R.size
        
        # Check convergence
        if e_D < threshold and e_R < threshold:
            if verbose:
                print('Converged at iteration', _)
            break
        
        # Update previous matrices for next iteration
        D_prev, R_prev = D.copy(), R.copy()
        
    # Plot convergence trend
    if imshow:
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.grid()
        plt.show()

    return D, R

def CauchyNMF(X: np.ndarray, n_components: int, max_iter: int=5000, random_state: Union[int, np.random.RandomState, None]=None, threshold: float=1e-8, epsilon: float=1e-7, verbose: bool=True, imshow: bool=False) -> (np.ndarray, np.ndarray):
    """
    
    Parameters:
    - X (numpy.ndarray): Input data matrix of shape (n_features, n_samples).
    - n_components (int): The number of components for matrix factorization.
    - max_iter (int, optional): Maximum number of iterations. Default is 5000.
    - random_state (int, np.random.RandomState, None, optional): Random state for reproducibility. Default is None.
    - threshold (float, optional): Convergence threshold. Default is 1e-4.
    - epsilon (float, optional): Small constant added to denominator to prevent division by zero. Default is 1e-7.
    - verbose (bool, optional): Whether to print convergence information. Default is True.
    - imshow (bool, optional): Whether to plot convergence trend. Default is False.

    Returns:
    - D (numpy.ndarray): Basis matrix obtained from NMF.
    - R (numpy.ndarray): Coefficient matrix.
    """
    
    def update_rule(A, B):
        """Update rule for Cauchy divergence."""
        return B / (A + np.sqrt(A**2 + 2 * B * A + epsilon) + epsilon)
    
    # Initialize D and R matrices using NICA algorithm
    start_time = time.time()
    D, R = NICA(X, n_components, random_state=random_state)
    if verbose:
        print(f'Initialization done. Time elapsed: {time.time() - start_time:.2f} seconds.')
    # Calculate initial Cauchy divergence
    prev_cauchy_div = np.sum(np.log(np.dot(D, R) + epsilon) - np.log(X + epsilon) + (X - np.dot(D, R)) / (np.dot(D, R) + epsilon))
    # Initialize loss
    cauchy_div_list = []

    for _ in conditional_tqdm(range(max_iter), verbose=verbose):

        # Update rule for D
        DR = np.dot(D, R)
        A = 3 / 4 * np.dot((DR / (DR ** 2 + X + epsilon)), R.T)
        B = np.dot(1 / (DR + epsilon), R.T)
        D *= update_rule(A, B)

        # Update rule for R
        DR = np.dot(D, R)
        A = 3 / 4 * np.dot(D.T, (DR / (DR ** 2 + X + epsilon)))
        B = np.dot(D.T, 1 / (DR + epsilon))
        R *= update_rule(A, B)

        # Calculate Cauchy divergence
        DR = np.dot(D, R)
        cauchy_div = np.sum(np.log(DR + epsilon) - np.log(X + epsilon) + (X - DR) / (DR + epsilon))
        cauchy_div_list.append(cauchy_div)

        # Check convergence
        if abs(cauchy_div - prev_cauchy_div) < threshold:
            if verbose:
                print('Converged at iteration', _)
            break

        # Update previous Cauchy divergence for next iteration
        prev_cauchy_div = cauchy_div

    # Plot convergence trend
    if imshow:
        plt.plot(cauchy_div_list)
        plt.xlabel('Iteration')
        plt.ylabel('Cauchy divergence')
        plt.grid()
        plt.show()

    return D, R 