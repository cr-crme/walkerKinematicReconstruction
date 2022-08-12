import os

import bioviz

from walker import WalkerModel

# Options
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote/Audrey_19_mai_statique.c3d"
trial = "data/pilote/Audrey_19_mai_marche4.c3d"

# Generate the personalized kinematic model
model = WalkerModel()
model.generate_personalized_model(static_trial, kinematic_model_file_path)

# Reconstruct the kinematics of a trial
model.reconstruct_kinematics(trial)

# Write the c3d as if it was the plug in gate output
model.to_c3d()

# Print the model
b = bioviz.Viz(kinematic_model_file_path)
b.load_movement(model.q)
b.load_experimental_markers(trial)
b.exec()

# Clean up
os.remove(kinematic_model_file_path)
