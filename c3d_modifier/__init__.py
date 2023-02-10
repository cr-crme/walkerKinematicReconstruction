import ezc3d
import numpy as np


def remove_markers(file_path: str, save_path: str, marker_to_remove: tuple[str, ...]) -> None:
    """
    Remove the values of the specified marker by changing them for a nan

    Parameters
    ----------
    file_path: str
        The path of the C3D file to modify
    save_path: str
        The path of the new C3D file
    marker_to_remove: list[str, ...]
        The list of markers to remove from the file
    """
    c3d = ezc3d.c3d(file_path)

    marker_indices = tuple(c3d["parameters"]["POINT"]["LABELS"]["value"].index(n) for n in marker_to_remove)
    c3d["data"]["points"][:, marker_indices, :] = np.nan
    c3d.write(save_path)
