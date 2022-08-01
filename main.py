import os

import biorbd
from biorbd import Segment, Marker, KinematicModelGeneric,Axis
import bioviz
import ezc3d

kinematic_model_file_path = "temporary.bioMod"
c3d = ezc3d.c3d("data/pilote/Audrey_19_mai_statique.c3d")


# 'LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN', 'RSHO',
# 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
# 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE', 'LHUM', 'RHUM', 'LTROC', 'RTROC', 'LTHID', 'RTHID', 'LKNM', 'RKNM',
# 'LTIBP', 'LTIBD', 'RTIBP', 'RTIBD', 'LANKM', 'L5MH', 'RANKM', 'R5MH', '*51'

model = KinematicModelGeneric()
model.add_segment("PELVIS", translations="yz", rotations="x")
model.set_rt(
    segment_name="PELVIS",
    origin_markers=("LPSI", "RPSI", "LASI", "RASI"),
    first_axis_name=Axis.Name.Y,
    first_axis_markers=(("LPSI", "RPSI"), ("LASI", "RASI")),
    second_axis_name=Axis.Name.X,
    second_axis_markers=(("LPSI", "RPSI", "LASI", "RASI"), ("RPSI", "RASI")),
    axis_to_keep=Axis.Name.Y,
)

# Collapse the model to an actual personalized model
model.generate_personalized(c3d, "temporary.bioMod")

#
# # The trunk segment
# mid_back_trunk = Marker.from_data(
#     c3d=c3d,
#     name="PELVIS",
#     data_names=("LFHD",),
#     parent_name="TRUNK",
# )
# trunk = Segment(
#     name="TRUNK",
#     translations="yz",
#     rotations="x",
#     mesh=((0, 0, 0), (0, 0, 0.53)),
#     markers=(trunk_marker_pelvis,),
# )
#
# # The head segment
# top_head_marker_head = Marker.from_data(
#     data=data,
#     name="TOP_HEAD",
#     parent_name="HEAD",
# )
# head = Segment(
#     name="HEAD",
#     parent_name="TRUNK",
#     rt="0 0 0 xyz 0 0 0.53",
#     mesh=((0, 0, 0), (0, 0, 0.24)),
#     markers=(top_head_marker_head,),
# )
#
# # The arm segment
# shoulder_marker = Marker(
#     name="SHOULDER",
#     parent_name="UPPER_ARM",
#     position=(0, 0, 0),
# )
# upper_arm = Segment(
#     name="UPPER_ARM",
#     parent_name=trunk.name,
#     rt="0 0 0 xyz 0 0 0.53",
#     rotations="x",
#     mesh=((0, 0, 0), (0, 0, -0.28)),
#     markers=(shoulder_marker,),
# )
#
# elbow_marker = Marker(
#     name="ELBOW",
#     parent_name="LOWER_ARM",
#     position=(0, 0, 0),
# )
# lower_arm = Segment(
#     name="LOWER_ARM",
#     parent_name=upper_arm.name,
#     rt="0 0 0 xyz 0 0 -0.28",
#     mesh=((0, 0, 0), (0, 0, -0.27)),
#     markers=(elbow_marker,),
# )
#
# wrist_marker = Marker(
#     name="WRIST",
#     parent_name="HAND",
#     position=(0, 0, 0),
# )
# finger_marker = Marker(
#     name="FINGER",
#     parent_name="HAND",
#     position=(0, 0, -0.19),
# )
# hand = Segment(
#     name="HAND",
#     parent_name=lower_arm.name,
#     rt="0 0 0 xyz 0 0 -0.27",
#     mesh=((0, 0, 0), (0, 0, -0.19)),
#     markers=(wrist_marker, finger_marker)
# )
#
# # The thigh segment
# thigh = Segment(
#     name="THIGH",
#     parent_name=trunk.name,
#     rotations="x",
#     mesh=((0, 0, 0), (0, 0, -0.42)),
# )
#
# # The shank segment
# knee_marker = Marker(
#     name="KNEE",
#     parent_name="SHANK",
#     position=(0, 0, 0),
# )
# shank = Segment(
#     name="SHANK",
#     parent_name=thigh.name,
#     rt="0 0 0 xyz 0 0 -0.42",
#     rotations="x",
#     mesh=((0, 0, 0), (0, 0, -0.43)),
#     markers=(knee_marker,),
# )
#
# # The foot segment
# ankle_marker = Marker(
#     name="ANKLE",
#     parent_name="FOOT",
#     position=(0, 0, 0),
# )
# toe_marker = Marker(
#     name="TOE",
#     parent_name="FOOT",
#     position=(0, 0, 0.25),
# )
# foot = Segment(
#     name="FOOT",
#     parent_name=shank.name,
#     rt="0 0 0 xyz 0 0 -0.43",
#     rotations="x",
#     mesh=((0, 0, 0), (0, 0, 0.25)),
#     markers=(ankle_marker, toe_marker,),
# )
#
# # Put the model together, print it and print it to a bioMod file
# kinematic_chain = KinematicChain(segments=(trunk, head, upper_arm, lower_arm, hand, thigh, shank, foot))
# kinematic_chain.write(kinematic_model_file_path)
#
# model = biorbd.Model(kinematic_model_file_path)
# from bioviz.model_creation.__init__ import BiorbdUtils
# print(BiorbdUtils.get_marker_names(model))
# bioviz.Viz(kinematic_model_file_path).exec()
#
# os.remove(kinematic_model_file_path)
