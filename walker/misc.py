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


def to_rotation_matrix(
    angles: np.ndarray,
    angle_sequence: str,
    translations: tuple[float | int, float | int, float | int] = None,
) -> np.ndarray:
    """
    Construct a SegmentCoordinateSystemReal from angles and translations

    Parameters
    ----------
    angles
        The actual angles
    angle_sequence
        The angle sequence of the angles
    translations
        The XYZ translations. If it is None, then the matrix is 3x3
    """
    n = angles.shape[1]
    ones = np.ones((1, 1, n))
    zeros = np.zeros((1, 1, n))
    matrix = {
        "x": lambda x: np.concatenate(
            (
                np.concatenate((ones, zeros, zeros), axis=1),
                np.concatenate((zeros, np.cos(x), -np.sin(x)), axis=1),
                np.concatenate((zeros, np.sin(x), np.cos(x)), axis=1),
            ),
            axis=0),
        "y": lambda y: np.concatenate(
            (
                np.concatenate((np.cos(y), zeros, np.sin(y)), axis=1),
                np.concatenate((zeros, ones, zeros), axis=1),
                np.concatenate((-np.sin(y), zeros, np.cos(y)), axis=1),
            ),
            axis=0),
        "z": lambda z: np.concatenate(
            (
                np.concatenate((np.cos(z), -np.sin(z), zeros), axis=1),
                np.concatenate((np.sin(z), np.cos(z), zeros), axis=1),
                np.concatenate((zeros, zeros, ones), axis=1),
            ),
            axis=0),
    }
    rt = np.repeat(np.identity(4 if translations is not None else 3)[:, :, np.newaxis], n, axis=2)
    for angle, axis in zip(angles, angle_sequence):
        rt[:3, :3, :] = np.einsum("ijk,jlk->ilk", rt[:3, :3, :], matrix[axis](angle[np.newaxis, np.newaxis, ...]))
    if translations is not None:
        rt[:3, 3, :] = translations
    return rt


def to_euler(rt, sequence: str) -> np.ndarray:
    if sequence == "xyz":
        rx = np.arctan2(-rt[1, 2, :], rt[2, 2, :])
        ry = np.arcsin(rt[0, 2, :])
        rz = np.arctan2(-rt[0, 1, :], rt[0, 0, :])
    else:
        raise NotImplementedError("This sequence is not implemented yet")

    return np.concatenate((rx[np.newaxis, :], ry[np.newaxis, :], rz[np.newaxis, :]), axis=0)
