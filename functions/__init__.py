from dataclasses import dataclass
import os
from enum import Enum

import numpy as np
import scipy

from c3d_modifier import remove_markers
from walker import BiomechanicsTools


class DoF(Enum):
    TRANS_X = "TransX"
    TRANS_Y = "TransY"
    TRANS_Z = "TransZ"
    ROT_X = "RotX"
    ROT_Y = "RotY"
    ROT_Z = "RotZ"


class RelativeTo(Enum):
    PARENT = "parent"
    VERTICAL = "vertical"


class Side(Enum):
    RIGHT = "Right"
    LEFT = "Left"


@dataclass
class DoFCondition:
    name: str
    segments: tuple[str, ...]
    dof: DoF
    sides: tuple[Side, ...]
    relative_to: RelativeTo = RelativeTo.PARENT

    def __post_init__(self):
        if  len(self.segments) != len(self.sides):
            raise ValueError("segments and sides must be of same size")


@dataclass
class MarkerOcclusionCondition:
    name: str
    remove_indices: tuple
    color: str


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
    tools: BiomechanicsTools, data: np.ndarray, side: Side, len_output: int = 101
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
        The side (Side.RIGHT or Side.LEFT) to extract to
    len_output
        The number of data point to put the output

    Returns
    -------
    All the cycles in a tuple
    """
    cycles = tools.get_cycles(side.value)

    out = []
    for i in range(len(cycles) - 1):
        if data[cycles[i]] == 0.0 or data[cycles[i + 1]] == 0.0:  # Do not keep incomplete cycle
            continue
        tck = scipy.interpolate.splrep(np.arange(cycles[i + 1] - cycles[i]), data[cycles[i] : cycles[i + 1]])
        out.append(scipy.interpolate.splev(np.linspace(0, cycles[i + 1] - cycles[i], len_output), tck))

    return tuple(out)


def extract_dof_condition(tools: BiomechanicsTools, dof_condition: DoFCondition, segment_name: str) -> np.ndarray:
    """
    Extract the cycles from tools.q according to the specified DoFCondition

    Parameters
    ----------
    tools
        The personalized kinematic model with the kinematics reconstructed
    dof_condition
        The condition of degrees of freedom to extract the cycle
    segment_name
        The segment_name to extract information from

    Returns
    -------
    The data for the specified DoF
    """
    # Get some aliases
    segment_names = tuple(s.name().to_string() for s in tools.model.segments())
    dof_names = tuple(n.to_string() for n in tools.model.nameDof())
    segment_idx = segment_names.index(segment_name)
    dof_name = dof_condition.dof.value
    dof_idx = dof_names.index(f"{segment_name}_{dof_name}")

    # Extract and reexpress the data if needed
    if dof_condition.relative_to == RelativeTo.VERTICAL:
        index = tools.model.segment(segment_idx).getDofIdx(dof_name)
        data = tools.relative_to_vertical(segment_name, "xyz", tools.q)[index, :]
    else:
        data = tools.q[dof_idx, :]
    return data
