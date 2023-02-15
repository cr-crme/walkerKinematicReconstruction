import os

from walker import BiomechanicsTools
from matplotlib import pyplot as plt
import numpy as np

from c3d_modifier import remove_markers


# --- Options --- #
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
)
markers_to_remove = (
    (),
    ("LPSI", "RPSI", "LASI", "RASI"),
    ("LPSI", "RPSI"),
)
colors = ("r", "g", "b", "m", "k")
dof_to_compare = ("Pelvis_RotY", "RFemur_RotY", "RTibia_RotY", "RFoot_RotY", "LFemur_RotY", "LTibia_RotY", "LFoot_RotY",)
# --------------- #


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
    tools.process_kinematic_reconstruction(temporary_c3d_path, visualize=False)
    os.remove(temporary_c3d_path)
    return tools.q


def main():
    # Sanity check
    if len(colors) < len(markers_to_remove):
        raise ValueError("The number of conditions should match")

    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=False)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Reconstruct kinematics but simulate marker occlusions
    q_all_trials = []
    for trial in trials:
        q_all_trials.append([reconstruct_with_occlusions(tools, trial, markers) for markers in markers_to_remove])

    # TODO : reconstruct pelvis relative to vertical
    # Plot the results
    dof_names = tuple(n.to_string() for n in tools.model.nameDof())
    for name in dof_to_compare:
        dof = dof_names.index(name)
        plt.figure()
        plt.title(f"DoF: {name}")
        for q_trials in q_all_trials:
            for condition, color in zip(q_trials, colors):
                plt.plot(np.unwrap(condition[dof, :]) * 180 / np.pi, color)
    plt.show()


if __name__ == "__main__":
    main()
