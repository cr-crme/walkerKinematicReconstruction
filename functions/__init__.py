import os

import numpy as np
import scipy

from c3d_modifier import remove_markers
from walker import BiomechanicsTools


def reconstruct_with_occlusions(tools: BiomechanicsTools, path: str, markers_to_occlude: tuple[str, ...]) -> np.ndarray:
    """
    Reconstruct kinematics, but simulate marker occlusions on the requested markers

    Parameters
    ----------
    tools
        The personalized kinematic model
    path
        The path of the C3D file to reconstruct
    markers_to_occlude
        The name of the markers to occlude (replace them with a nan in the C3D)

    Returns
    -------
    The kinematics reconstructed is placed in tools.q and returned
    """

    temporary_c3d_path = "tp.c3d"
    remove_markers(path, temporary_c3d_path, markers_to_occlude)
    tools.process_kinematics(temporary_c3d_path, visualize=False)
    os.remove(temporary_c3d_path)
    return tools.q


def normalize_into_cycles(
    tools: BiomechanicsTools, data: np.ndarray, side: str, len_output: int = 100
) -> tuple[np.ndarray, ...]:
    """
    Extract the cycles and put them in 0 to 100% of the cycle, for the required side

    Parameters
    ----------
    tools
        The personalized kinematic model with a c3d file loaded
    data
        The data (1d) to normalize
    side
        The side ('right' or 'left') to extract to
    len_output
        The number of data point to put the output

    Returns
    -------
    All the cycles in a tuple
    """
    cycles = tools.get_cycles(side)

    out = []
    for i in range(len(cycles) - 1):
        if data[cycles[i]] == 0.0 or data[cycles[i + 1]] == 0.0:  # Do not keep incomplete cycle
            continue
        tck = scipy.interpolate.splrep(np.arange(cycles[i + 1] - cycles[i]), data[cycles[i] : cycles[i + 1]])
        out.append(scipy.interpolate.splev(np.linspace(0, cycles[i + 1] - cycles[i], len_output), tck))

    return tuple(out)


def compute_skew_angle(matrix1, matrix2):
    pass
    # TODO
