import numpy as np
import matplotlib.pyplot as plt

def origin_plus_noise(X_hat: np.ndarray, X_noise: np.ndarray, X: np.ndarray, image_size: tuple, reduce: int, idx: int=2) -> None:
    """
    Display the original image, the noise, and the image with added noise side by side.
    
    Parameters:
    - X_hat (numpy.ndarray): Original image data.
    - X_noise (numpy.ndarray): Noise data to be added to the original image.
    - image_size (tuple): Size of the original image as (height, width).
    - reduce (int): Factor to downscale the image dimensions.
    - idx (int, optional): Index of the image to be displayed. Default is 2.
    """

    # Calculate reduced image size based on the 'reduce' factor
    img_size = [i//reduce for i in image_size]
    
    # Retrieve the specified image from the data
    X_hat_i = X_hat[:,idx].reshape(img_size[1],img_size[0])
    X_noise_i = X_noise[:,idx].reshape(img_size[1],img_size[0])
    X_i = X[:,idx].reshape(img_size[1],img_size[0])

    # Set up the figure for displaying images
    plt.figure(figsize=(12,3))  # Adjusted size for better visualization
    
    # Display the original image
    plt.subplot(151)  # Adjusted to 1x4 grid for space to '+' and '=' symbols
    plt.imshow(X_hat_i, cmap=plt.cm.gray)
    plt.title('Image(Original)')
    plt.axis('off')  # Hide axis for a cleaner look
    
    # Place '+' symbol between images
    plt.subplot(152)
    plt.text(0.5, 0.5, '+', fontsize=20, ha='center', va='center')
    plt.axis('off')  # Hide axis
    
    # Display the noise
    plt.subplot(153)
    plt.imshow(X_noise_i, cmap=plt.cm.gray)
    plt.title('Noise')
    plt.axis('off')  # Hide axis for a cleaner look
    
    # Place '=' symbol between images
    plt.subplot(154)
    plt.text(0.5, 0.5, '=', fontsize=20, ha='center', va='center')
    plt.axis('off')  # Hide axis
    
    # Display the image with added noise
    plt.subplot(155)
    plt.imshow(X_i, cmap=plt.cm.gray)
    plt.title('Image(Noise)')
    plt.axis('off')  # Hide axis for a cleaner look
    
    # Render the figure
    plt.tight_layout()  # Ensure no overlap between subplots
    plt.show()

def origin_versus_dictrep(X: np.ndarray, D: np.ndarray, R: np.ndarray, X_noise: np.ndarray, image_size: tuple, reduce: int, idx: int) -> None:
    """
    Display the original, noise-added, and dictionary-reconstructed images side by side.
    
    Parameters:
    - X (numpy.ndarray): Original data matrix of shape (n_samples, n_features).
    - D (numpy.ndarray): Basis matrix obtained from dictionary learning.
    - R (numpy.ndarray): Coefficient matrix.
    - X_noise (numpy.ndarray): Noise-added version of the original data matrix.
    - image_size (tuple): Tuple containing the height and width of the image.
    - reduce (int): Factor by which the image size is reduced for visualization.
    - idx (int): Index of the image to display.

    Returns:
    None. The function will plot and display the images using matplotlib.
    """

    DR = np.dot(D, R).reshape(X.shape[0], X.shape[1])
    # Calculate reduced image size based on the 'reduce' factor
    img_size = [i//reduce for i in image_size]
    
    # Retrieve the specified image from the data
    X_i = X[:,idx].reshape(img_size[1],img_size[0])
    X_noise_i = X_noise[:,idx].reshape(img_size[1],img_size[0])
    DR_i = DR[:,idx].reshape(img_size[1],img_size[0])

    # Set up the figure for displaying images
    plt.figure(figsize=(12,3))  # Adjusted size for better visualization

    # Display the original image
    plt.subplot(131)
    plt.imshow(X_i, cmap=plt.cm.gray)
    plt.title('Image(Original)')
    plt.axis('off')

    # Display the reconstructed image
    plt.subplot(132)
    plt.imshow(X_noise_i, cmap=plt.cm.gray)
    plt.title('Image(Noise)')
    plt.axis('off')

    # Display the sparse coefficients
    plt.subplot(133)
    plt.imshow(DR_i, cmap=plt.cm.gray)
    plt.title('Image(Reconstructed))')
    plt.axis('off')

    # Render the figure
    plt.tight_layout()
    plt.show()
    
    return X_i, X_noise_i, DR_i

def origin_noise_dictrep(X: np.ndarray, X_noise: np.ndarray, D: np.ndarray, R: np.ndarray, image_size: tuple, reduce: int, idx: int) -> None:
    """
    Display the original image, its noise version, and its dictionary-reconstructed representation side by side.
    
    Parameters:
    - X (numpy.ndarray): Original data matrix of shape (n_samples, n_features).
    - X_noise (numpy.ndarray): Noise-added version of the original data matrix.
    - D (numpy.ndarray): Basis matrix obtained from dictionary learning.
    - R (numpy.ndarray): Coefficient matrix.
    - image_size (tuple): Tuple containing the height and width of the image.
    - reduce (int): Factor by which the image size is reduced for visualization.
    - idx (int): Index of the image to display.

    Returns:
    None. The function will plot and display the images using matplotlib.
    """
    
    DR = np.dot(D, R).reshape(X.shape[0], X.shape[1])
    # Calculate reduced image size based on the 'reduce' factor
    img_size = [i//reduce for i in image_size]
    
    # Retrieve the specified image from the data
    X_i = X[:,idx].reshape(img_size[1],img_size[0])
    X_noise_i = X_noise[:,idx].reshape(img_size[1],img_size[0])
    DR_i = DR[:,idx].reshape(img_size[1],img_size[0])

    # Set up the figure for displaying images
    plt.figure(figsize=(12,3))  # Adjusted size for better visualization

    # Display the original image
    plt.subplot(131)
    plt.imshow(X_i, cmap=plt.cm.gray)
    plt.title('Image(Original)')
    plt.axis('off')

    # Display the noise
    plt.subplot(132)
    plt.imshow(X_noise_i, cmap=plt.cm.gray)
    plt.title('Image(Noise)')
    plt.axis('off')

    # Display the reconstructed image
    plt.subplot(133)
    plt.imshow(DR_i, cmap=plt.cm.gray)
    plt.title('Image(Reconstructed)')
    plt.axis('off')

    # Render the figure
    plt.tight_layout()
    plt.show()