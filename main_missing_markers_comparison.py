from walker import BiomechanicsTools
from matplotlib import pyplot as plt
import numpy as np

from functions import reconstruct_with_occlusions, normalize_into_cycles


# --- Options --- #
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marche_01.c3d",
    "data/pilote2/2023-01-19_AP_test_marche_02.c3d",
    "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
    "data/pilote2/2023-01-19_AP_test_marchecrouch_06.c3d",
)
marker_conditions = {
    "All markers": {"remove_markers": (), "color": "r"},
    "No pelvis": {"remove_markers": ("LPSI", "RPSI", "LASI", "RASI"), "color": "g"},
    "Back pelvis only": {"remove_markers": ("LASI", "RASI"), "color": "b"},
}
dof_conditions = {
    "Pelvis (right cycle)": {"segment": "Pelvis", "dof": "RotY", "relative_to_vertical": False, "side": "right"},
    "Pelvis (left cycle)": {"segment": "Pelvis", "dof": "RotY", "relative_to_vertical": False, "side": "left"},
    "Right Hip (vertical)": {"segment": "RFemur", "dof": "RotY", "relative_to_vertical": True, "side": "right"},
    "Right Hip": {"segment": "RFemur", "dof": "RotY", "relative_to_vertical": False, "side": "right"},
    "Right Knee": {"segment": "RTibia", "dof": "RotY", "relative_to_vertical": False, "side": "right"},
    "Right Ankle": {"segment": "RFoot", "dof": "RotY", "relative_to_vertical": False, "side": "right"},
    "Left Hip (vertical)": {"segment": "LFemur", "dof": "RotY", "relative_to_vertical": True, "side": "left"},
    "Left Hip": {"segment": "LFemur", "dof": "RotY", "relative_to_vertical": False, "side": "left"},
    "Left Knee": {"segment": "LTibia", "dof": "RotY", "relative_to_vertical": False, "side": "left"},
    "Left Ankle": {"segment": "LFoot", "dof": "RotY", "relative_to_vertical": False, "side": "left"},
}
# --------------- #


def main():
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=False)
    tools.personalize_model(static_trial)

    # Reconstruct kinematics but simulate marker occlusions
    dof_names = tuple(n.to_string() for n in tools.model.nameDof())
    segment_names = tuple(s.name().to_string() for s in tools.model.segments())
    results = {dof: {marker: [] for marker in marker_conditions} for dof in dof_conditions}
    for trial in trials:
        for marker_key in marker_conditions:
            q = reconstruct_with_occlusions(tools, trial, marker_conditions[marker_key]["remove_markers"])

            for dof_key in dof_conditions:
                segment_name = dof_conditions[dof_key]["segment"]
                segment_idx = segment_names.index(segment_name)
                dof_name = dof_conditions[dof_key]["dof"]
                dof_idx = dof_names.index(f"{segment_name}_{dof_name}")
                # Select the dof to print
                if dof_conditions[dof_key]["relative_to_vertical"]:
                    index = tools.model.segment(segment_idx).getDofIdx(dof_name)
                    data = tools.relative_to_vertical(segment_name, "xyz", q)[index, :]
                else:
                    data = q[dof_idx, :]

                # Separate the data in cycles
                if dof_conditions[dof_key]["side"] is not None:
                    cycles = normalize_into_cycles(tools, data, dof_conditions[dof_key]["side"])
                else:
                    cycles = (data,)
                results[dof_key][marker_key].append(cycles)

    # Plot the results
    for dof_key in dof_conditions:
        dof_name = dof_conditions[dof_key]["dof"]
        plt.figure()
        plt.title(f"Joint: {dof_key}, about {dof_name[-1]}")
        plt.xlabel("Cycle (%)")
        plt.ylabel(f"{dof_key} angle (degree)")
        for marker_key in marker_conditions:
            for trial in results[dof_key][marker_key]:
                for cycle in trial:
                    plt.plot(cycle * 180 / np.pi, marker_conditions[marker_key]["color"])
    plt.show()


if __name__ == "__main__":
    main()
