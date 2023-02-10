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
)
colors = ("r", "b")
dof_to_compare = ("RFemur_RotY", "RTibia_RotY", "RFoot_RotY", "LFemur_RotY", "LTibia_RotY", "LFoot_RotY",)
# --------------- #

# Sanity check
if len(colors) != len(markers_to_remove):
    raise ValueError("The number of conditions should match")

# Generate the personalized kinematic model
tools = BiomechanicsTools(body_mass=100)
tools.personalize_model(static_trial, kinematic_model_file_path)

# Perform some biomechanical computation
all_q = []
for trial in trials:
    q_tp = []
    for markers in markers_to_remove:
        remove_markers(trial, "tp.c3d", markers)
        tools.process_trial("tp.c3d", compute_automatic_events=False, only_compute_kinematics=True)
        q_tp.append(tools.q)
        os.remove("tp.c3d")
    all_q.append(q_tp)

# TODO : reconstruct pelvis relative to vertical

for name in dof_to_compare:
    dof = tuple(n.to_string() for n in tools.model.nameDof()).index(name)
    plt.figure()
    plt.title(f"DoF: {name}")
    for q in all_q:
        for condition, color in zip(q, colors):
            plt.plot(np.unwrap(condition[dof, :]) * 180 / np.pi, color)
plt.show()
