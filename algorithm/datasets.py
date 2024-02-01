import os

import numpy as np
from PIL import Image

def validate_data_directory(root: str) -> None:
    """
    Validate the data directory.
    
    Parameters:
    - root (str): Path to the dataset.
    """

    # Check 1: root exists.
    if not os.path.exists(root):
        raise FileNotFoundError(f'{root} does not exist!')
    
    # Check 2: data directory is not empty.
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        raise FileNotFoundError(f'{root} is empty!')

    # Check 3: each subdirectory contains at least one image.
    for subdir in subdirs:
        pgm_files = [f for f in os.listdir(os.path.join(root, subdir)) if f.endswith('.pgm')]
        if not pgm_files:
            raise FileNotFoundError(f'{os.path.join(root, subdir)} does not contain any image!')
    

def load_data(root: str='data/CroppedYaleB', reduce: int=1, global_centering: bool=True, local_centering: bool=True) -> (np.ndarray, np.ndarray): 
    """
    Load ORL (or Extended YaleB) dataset into a numpy array.
    
    Parameters:
    - root (str): Path to the dataset.
    - reduce (int): Scale factor for downscaling images.
    - global_centering (bool): If True, apply global centering.
    - local_centering (bool): If True, apply local centering.

    Returns:
    - images (numpy.ndarray): Image data.
    - labels (numpy.ndarray): Image labels.
    """

    # Validate the data directory.
    validate_data_directory(root)

    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
            
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # Load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.

            # Reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])

            # Convert image to numpy array.
            img = np.asarray(img).reshape((-1,1))

            # Collect data and label.
            images.append(img)
            labels.append(i)

    # Concatenate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    # Convert to float64 for numerical stability
    images = images.astype(np.float64)

    # Global centering.
    if global_centering:
        images -= images.mean(axis=0)
    
    # Local centering.
    if local_centering:
        images -= images.mean(axis=1).reshape(-1, 1)

    return images, labels


def get_image_size(root: str='code/dataCroppedYaleB') -> tuple:
    """
    Get the size of images in the dataset.
    
    Parameters:
    - root (str): Path to the dataset.

    Returns:
    - img_size (tuple): Size of each image as (width, height).
    """

    # Validate the data directory.
    validate_data_directory(root)

    img_size = None  # Initialize variable to hold image size

    for person in sorted(os.listdir(root)):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):
            
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # Load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # Grey image.

            # Reduce computation complexity.
            img = img.resize([s for s in img.size])

            # Store the image size and return immediately
            return img.size  # (width, height)