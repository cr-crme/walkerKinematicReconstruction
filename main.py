from walker import BiomechanicsTools

# Options
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote/Audrey_19_mai_statique.c3d"
trials = ("data/pilote/Audrey_19_mai_marche4.c3d", "data/pilote/Audrey_19_mai_marche6.c3d")

# Generate the personalized kinematic model
tools = BiomechanicsTools(body_mass=100)
tools.personalize_model(static_trial, kinematic_model_file_path)

# Perform some biomechanical computation
for trial in trials:
    tools.process_trial(trial)
