from walker import BiomechanicsTools

# --- Options --- #
data_path = "data/projet_coucou/sujet3/"
kinematic_model_file_path = "temporary.bioMod"
static_trial = f"{data_path}/2023-01-19_AP_test_statique_TROCH.c3d"
trials = (
    f"{data_path}/2023-01-19_AP_test_marchecrouch_05.c3d",
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
