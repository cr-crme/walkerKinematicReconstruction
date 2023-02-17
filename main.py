from walker import BiomechanicsTools

# --- Options --- #
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote2/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    "data/pilote2/2023-01-19_AP_test_marchecrouch_05.c3d",
    "data/pilote2/2023-01-19_AP_test_marchecrouch_06.c3d",
)
# --------------- #


def main():
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=False)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Perform some biomechanical computation
    for trial in trials:
        tools.process_trial(trial, compute_automatic_events=False)

    # TODO: Bioviz vizual bug with the end of the trial when resizing the window
    # TODO: Record a tutorial
