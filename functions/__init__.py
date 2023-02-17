import os

import numpy as np

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


def compute_skew_angle(matrix1, matrix2):
    pass
    # TODO
