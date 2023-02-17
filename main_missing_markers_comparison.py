from walker import BiomechanicsTools
from matplotlib import pyplot as plt
import numpy as np

from functions import reconstruct_with_occlusions


# --- Options --- #
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marche_01.c3d",
)
markers_to_remove = (
    (),
    ("LPSI", "RPSI", "LASI", "RASI"),
    ("LPSI", "RPSI"),
)
colors = ("r", "g", "b", "m", "k")
dof_to_compare = {
    "Pelvis": {"dof": ("RotY", ), "relative_to_vertical": False},
    "RFemur": {"dof": ("RotY", ), "relative_to_vertical": True},
    "RTibia": {"dof": ("RotY", ), "relative_to_vertical": False},
    "RFoot": {"dof": ("RotY", ), "relative_to_vertical": False},
    "LFemur": {"dof": ("RotY", ), "relative_to_vertical": True},
    "LTibia": {"dof": ("RotY", ), "relative_to_vertical": False},
    "LFoot": {"dof": ("RotY", ), "relative_to_vertical": False},
}
# --------------- #


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
        q_tp = []
        for markers in markers_to_remove:
            q = reconstruct_with_occlusions(tools, trial, markers)
            q_tp.append(q)
        q_all_trials.append(q_tp)

    # Plot the results
    dof_names = tuple(n.to_string() for n in tools.model.nameDof())
    segment_names = tuple(s.name().to_string() for s in tools.model.segments())
    for segment_name in dof_to_compare:
        segment_idx = segment_names.index(segment_name)
        for dof_name in dof_to_compare[segment_name]["dof"]:
            dof_idx = dof_names.index(f"{segment_name}_{dof_name}")
            plt.figure()
            plt.title(f"Segment: {segment_name}, {dof_name}")
            for q_trials in q_all_trials:
                for condition, color in zip(q_trials, colors):
                    if dof_to_compare[segment_name]["relative_to_vertical"]:
                        index = tools.model.segment(segment_idx).getDofIdx(dof_name)
                        data = tools.relative_to_vertical(segment_name, "xyz", condition)[index, :]
                    else:
                        data = condition[dof_idx, :]
                    plt.plot(data * 180 / np.pi, color)
    plt.show()


if __name__ == "__main__":
    main()
