from walker import BiomechanicsTools
from matplotlib import pyplot as plt
import numpy as np

from functions import (
    extract_dof_condition,
    normalize_into_cycles,
    reconstruct_with_occlusions,
    DoF,
    Side,
    RelativeTo,
    DoFCondition,
    MarkerOcclusionCondition,
)


# --- Options --- #
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marche_01.c3d",
    "data/pilote2/2023-01-19_AP_test_marche_02.c3d",
    "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
    "data/pilote2/2023-01-19_AP_test_marchecrouch_06.c3d",
)
marker_conditions = (
    MarkerOcclusionCondition(name="All markers", remove_indices=(), color="k"),
    MarkerOcclusionCondition(name="No pelvis", remove_indices=("LPSI", "RPSI", "LASI", "RASI"), color="g"),
    MarkerOcclusionCondition(name="Back pelvis only", remove_indices=("LASI", "RASI"), color="b"),
)
dof_conditions = (
    DoFCondition(name="Pelvis (Right cycle)", segment="Pelvis", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.RIGHT),
    DoFCondition(name="Pelvis (Left cycle)", segment="Pelvis", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.LEFT),
    DoFCondition(name="Right Hip (vertical)", segment="RFemur", dof=DoF.ROT_Y, relative_to=RelativeTo.VERTICAL, side=Side.RIGHT),
    DoFCondition(name="Right Hip", segment="RFemur", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.RIGHT),
    DoFCondition(name="Right Knee", segment="RTibia", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.RIGHT),
    DoFCondition(name="Right Ankle", segment="RFoot", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.RIGHT),
    DoFCondition(name="Left Hip", segment="LFemur", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.LEFT),
    DoFCondition(name="Left Knee", segment="LTibia", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.LEFT),
    DoFCondition(name="Left Ankle", segment="LFoot", dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, side=Side.LEFT),
)
# --------------- #


def main():
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=False)
    tools.personalize_model(static_trial)

    # Reconstruct kinematics but simulate marker occlusions
    results = {dof.name: {marker.name: [] for marker in marker_conditions} for dof in dof_conditions}
    for trial in trials:
        for marker_condition in marker_conditions:
            reconstruct_with_occlusions(tools, trial, marker_condition.remove_indices)
            for dof_condition in dof_conditions:
                dof_data = extract_dof_condition(tools, dof_condition)
                cycles = normalize_into_cycles(tools, dof_data, dof_condition.side)
                results[dof_condition.name][marker_condition.name].append(cycles)

    # Plot the results
    for dof_condition in dof_conditions:
        dof_name = dof_condition.dof.value
        plt.figure()
        plt.title(f"Joint: {dof_condition.name}, about {dof_name[-1]}")
        plt.xlabel("Cycle (%)")
        plt.ylabel(f"{dof_condition.name} angle (degree)")
        for marker_condition in marker_conditions:
            for trial in results[dof_condition.name][marker_condition.name]:
                for cycle in trial:
                    plt.plot(cycle * 180 / np.pi, marker_condition.color)
    plt.show()


if __name__ == "__main__":
    main()
