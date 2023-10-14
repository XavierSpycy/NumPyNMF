from tqdm import tqdm
from typing import Generator

def conditional_tqdm(iterable: range, verbose: bool = True) -> Generator[int, None, None]:
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