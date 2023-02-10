from walker import BiomechanicsTools
from matplotlib import pyplot as plt

from c3d_modifier import remove_markers


# Options
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
)
markers_to_remove = (
    (),
    ('LPSI', 'RPSI', 'LASI', 'RASI'),
)

# Generate the personalized kinematic model
tools = BiomechanicsTools(body_mass=100)
tools.personalize_model(static_trial, kinematic_model_file_path)

# Perform some biomechanical computation
q = []
for trial in trials:
    q_tp = []
    for markers in markers_to_remove:
        remove_markers(trial, "tp.c3d", markers)
        tools.process_trial("tp.c3d", compute_automatic_events=False, only_compute_kinematics=True)
        q_tp.append(tools.q)
    q.append(q_tp)

plt.plot(q[0][0][0, :], 'r')
plt.plot(q[0][1][0, :], 'b')
plt.show()





