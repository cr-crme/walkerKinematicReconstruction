import numpy as np


def differentiate(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Differentiate data along the columns (NDof x nTime)

    Parameters
    ----------
    data: np.ndarray
        The data to differentiate
    dt: float
        The delta time
    """

    out = np.ndarray(data.shape) * np.nan
    out[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * dt)
    return out
