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
    # "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
    # "data/pilote2/2023-01-19_AP_test_marchecrouch_06.c3d",
)
marker_conditions = (
    MarkerOcclusionCondition(name="All markers", remove_indices=(), color="k"),
    MarkerOcclusionCondition(name="No pelvis", remove_indices=("LPSI", "RPSI", "LASI", "RASI"), color="g"),
    MarkerOcclusionCondition(name="Back pelvis only", remove_indices=("LASI", "RASI"), color="b"),
)
reference_marker_condition = "All markers"
dof_conditions = (
    DoFCondition(name="Pelvis", segments=("Pelvis", "Pelvis"), dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, sides=(Side.RIGHT, Side.LEFT)),
    DoFCondition(name="Hip (vertical)", segments=("RFemur", "LFemur"), dof=DoF.ROT_Y, relative_to=RelativeTo.VERTICAL, sides=(Side.RIGHT, Side.LEFT)),
    DoFCondition(name="Hip", segments=("RFemur", "LFemur"), dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, sides=(Side.RIGHT, Side.LEFT)),
    DoFCondition(name="Knee", segments=("RTibia", "LTibia"), dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, sides=(Side.RIGHT, Side.LEFT)),
    DoFCondition(name="Ankle", segments=("RFoot", "LFoot"), dof=DoF.ROT_Y, relative_to=RelativeTo.PARENT, sides=(Side.RIGHT, Side.LEFT)),
)
# --------------- #


def main():
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=False)
    tools.personalize_model(static_trial)

    # Reconstruct kinematics but simulate marker occlusions
    results = {dof.name: {marker.name: {side: [] for side in dof.sides} for marker in marker_conditions} for dof in dof_conditions}
    for trial in trials:
        # Reconstruct all the kinematics
        for marker_condition in marker_conditions:
            reconstruct_with_occlusions(tools, trial, marker_condition.remove_indices)
            for dof_condition in dof_conditions:
                for side, segment_name in zip(dof_condition.sides, dof_condition.segments):
                    dof_data = extract_dof_condition(tools, dof_condition, segment_name)
                    cycles = normalize_into_cycles(tools, dof_data, side)
                    results[dof_condition.name][marker_condition.name][side].append(cycles)

    # Compare all the kinematics to reference
    rmse = {dof.name: {marker.name: [] for marker in marker_conditions} for dof in dof_conditions}
    rmsae = {dof.name: {marker.name: [] for marker in marker_conditions} for dof in dof_conditions}
    for marker_condition in marker_conditions:
        for dof_condition in dof_conditions:
            for trial in range(len(trials)):
                for side in dof_condition.sides:
                    for cycle in range(len(results[dof_condition.name][marker_condition.name][side][trial])):
                        diff = results[dof_condition.name][marker_condition.name][side][trial][cycle] - results[dof_condition.name][reference_marker_condition][side][trial][cycle]
                        rmse[dof_condition.name][marker_condition.name].append(np.sqrt(np.mean(diff**2)))
                        rmsae[dof_condition.name][marker_condition.name].append(np.sqrt(np.mean(np.abs(diff) ** 2)))

    # Show RMSE in a table
    table = "| Degree of freedom | " + " | ".join(dof_condition.name for dof_condition in dof_conditions) + " |\n"
    table += "|:---:|" + ":---:|".join("" for _ in dof_conditions) + ":---:|\n"
    for marker_condition in marker_conditions:
        if marker_condition.name == reference_marker_condition:
            continue
        table += f"| {marker_condition.name} (random error) | " + \
                 " | ".join(f"{np.mean(rmse[dof_condition.name][marker_condition.name]) * 180 / np.pi:1.3f}"
                            for dof_condition in dof_conditions) + \
                 " |\n"
        table += f"| {marker_condition.name} (absolute error) | " + \
                 " | ".join(f"{np.mean(rmsae[dof_condition.name][marker_condition.name]) * 180 / np.pi:1.3f}"
                            for dof_condition in dof_conditions) + \
                 " |\n"
    print(table)

    # Plot the results
    for dof_condition in dof_conditions:
        dof_name = dof_condition.dof.value
        plt.figure()
        rmse_title = "; ".join([f"{m.name} = {np.mean(rmse[dof_condition.name][m.name]) * 180 / np.pi:1.1f}°" for m in marker_conditions if m.name != reference_marker_condition])
        plt.title(f"Joint: {dof_condition.name}, about {dof_name[-1]}\nRMSE: {rmse_title}")
        plt.xlabel("Cycle (%)")
        plt.ylabel(f"{dof_condition.name} angle (degree)")
        for marker_condition in marker_conditions:
            cycles = []
            for side in results[dof_condition.name][marker_condition.name]:
                for trial in results[dof_condition.name][marker_condition.name][side]:
                    for cycle in trial:
                        cycles.append(cycle[:, np.newaxis])
            cycles = np.concatenate(cycles, axis=1)
            cycles_mean = np.nanmean(cycles, axis=1) * 180 / np.pi
            cycles_std = np.nanstd(cycles, axis=1) * 180 / np.pi

            t = np.linspace(0, cycles_mean.shape[0] - 1, cycles_mean.shape[0])
            plt.plot(t, cycles_mean, marker_condition.color, label=marker_condition.name)
            plt.fill_between(t, cycles_mean - cycles_std, cycles_mean + cycles_std, alpha=0.3, facecolor=marker_condition.color)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
