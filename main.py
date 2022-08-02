import os

import bioviz

from walker import WalkerModel

# Options
kinematic_model_file_path = "temporary.bioMod"
static_trial = "data/pilote/Audrey_19_mai_statique.c3d"

# Generate the personalized kinematic model
model = WalkerModel()
model.generate_personalized(static_trial, kinematic_model_file_path)

# Print the model
bioviz.Viz(kinematic_model_file_path).exec()

# Clean up
os.remove(kinematic_model_file_path)
