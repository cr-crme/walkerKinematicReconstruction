from biorbd import KinematicModelGeneric, Axis
import bioviz
import ezc3d


kinematic_model_file_path = "temporary.bioMod"
c3d = ezc3d.c3d("data/pilote/Audrey_19_mai_statique.c3d")

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
model.add_marker("PELVIS", "mid_pelvis", ("LPSI", "RPSI", "LASI", "RASI"))
model.add_marker("PELVIS", "LPSI")
model.add_marker("PELVIS", "RPSI")
model.add_marker("PELVIS", "LASI")
model.add_marker("PELVIS", "RASI")

model.add_segment("TRUNK", parent_name="PELVIS", rotations="x")
model.set_rt(
    segment_name="TRUNK",
    origin_markers="CLAV",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=(("T10", "STRN"), ("C7", "CLAV")),
    second_axis_name=Axis.Name.Y,
    second_axis_markers=(("T10", "C7"), ("STRN", "CLAV")),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("TRUNK", "T10")
model.add_marker("TRUNK", "C7")
model.add_marker("TRUNK", "STRN")
model.add_marker("TRUNK", "CLAV")

model.add_segment("HEAD", parent_name="TRUNK")
model.set_rt(
    segment_name="HEAD",
    origin_markers=("LBHD", "RBHD", "LFHD", "RFHD"),
    first_axis_name=Axis.Name.X,
    first_axis_markers=(("LBHD", "LFHD"), ("RBHD", "RFHD")),
    second_axis_name=Axis.Name.Y,
    second_axis_markers=(("LBHD", "RBHD"), ("LFHD", "RFHD")),
    axis_to_keep=Axis.Name.Y,
)
model.add_marker("HEAD", "LBHD")
model.add_marker("HEAD", "RBHD")
model.add_marker("HEAD", "LFHD")
model.add_marker("HEAD", "RFHD")

model.add_segment("RUPPER_ARM", parent_name="TRUNK", rotations="zx")
model.set_rt(
    segment_name="RUPPER_ARM",
    origin_markers="RSHO",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("RELB", "RSHO"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("RWRB", "RWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("RUPPER_ARM", "RSHO")
model.add_marker("RUPPER_ARM", "RELB")

model.add_segment("RLOWER_ARM", parent_name="RUPPER_ARM", rotations="x")
model.set_rt(
    segment_name="RLOWER_ARM",
    origin_markers="RELB",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=(("RWRB", "RWRA"), "RELB"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("RWRB", "RWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("RLOWER_ARM", "RWRB")
model.add_marker("RLOWER_ARM", "RWRA")

model.add_segment("RHAND", parent_name="RLOWER_ARM", rotations="x")
model.set_rt(
    segment_name="RHAND",
    origin_markers=("RWRB", "RWRA"),
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("RFIN", ("RWRB", "RWRA")),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("RWRB", "RWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("RHAND", "RFIN")

model.add_segment("LUPPER_ARM", parent_name="TRUNK", rotations="zx")
model.set_rt(
    segment_name="LUPPER_ARM",
    origin_markers="LSHO",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("LELB", "LSHO"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("LWRB", "LWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("LUPPER_ARM", "LSHO")
model.add_marker("LUPPER_ARM", "LELB")

model.add_segment("LLOWER_ARM", parent_name="LUPPER_ARM", rotations="x")
model.set_rt(
    segment_name="LLOWER_ARM",
    origin_markers="LELB",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=(("LWRB", "LWRA"), "LELB"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("LWRB", "LWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("LLOWER_ARM", "LWRB")
model.add_marker("LLOWER_ARM", "LWRA")

model.add_segment("LHAND", parent_name="LLOWER_ARM", rotations="x")
model.set_rt(
    segment_name="LHAND",
    origin_markers=("LWRB", "LWRA"),
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("LFIN", ("LWRB", "LWRA")),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("LWRB", "LWRA"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("LHAND", "LFIN")

model.add_segment("RTHIGH", parent_name="PELVIS", rotations="x")
model.set_rt(
    segment_name="RTHIGH",
    origin_markers="RTROC",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("RKNE", "RTROC"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("RKNM", "RKNE"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("RTHIGH", "RTROC")
model.add_marker("RTHIGH", "RKNE")
model.add_marker("RTHIGH", "RKNM")

model.add_segment("RLEG", parent_name="RTHIGH", rotations="x")
model.set_rt(
    segment_name="RLEG",
    origin_markers=("RKNM", "RKNE"),
    first_axis_name=Axis.Name.Z,
    first_axis_markers=(("RANKM", "RANK"), ("RKNM", "RKNE")),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("RKNM", "RKNE"),
    axis_to_keep=Axis.Name.X,
)
model.add_marker("RLEG", "RANKM")
model.add_marker("RLEG", "RANK")

model.add_segment("RFOOT", parent_name="RLEG", rotations="x")
model.set_rt(
    segment_name="RFOOT",
    origin_markers=("RANKM", "RANK"),
    first_axis_name=Axis.Name.X,
    first_axis_markers=("RANKM", "RANK"),
    second_axis_name=Axis.Name.Y,
    second_axis_markers=(("RANKM", "RANK"), "RTOE"),
    axis_to_keep=Axis.Name.X,
)
model.add_marker("RFOOT", "RTOE")
model.add_marker("RFOOT", "R5MH")

model.add_segment("LTHIGH", parent_name="PELVIS", rotations="x")
model.set_rt(
    segment_name="LTHIGH",
    origin_markers="LTROC",
    first_axis_name=Axis.Name.Z,
    first_axis_markers=("LKNE", "LTROC"),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("LKNE", "LKNM"),
    axis_to_keep=Axis.Name.Z,
)
model.add_marker("LTHIGH", "LTROC")
model.add_marker("LTHIGH", "LKNE")
model.add_marker("LTHIGH", "LKNM")

model.add_segment("LLEG", parent_name="LTHIGH", rotations="x")
model.set_rt(
    segment_name="LLEG",
    origin_markers=("LKNM", "LKNE"),
    first_axis_name=Axis.Name.Z,
    first_axis_markers=(("LANKM", "LANK"), ("LKNM", "LKNE")),
    second_axis_name=Axis.Name.X,
    second_axis_markers=("LKNE", "LKNM"),
    axis_to_keep=Axis.Name.X,
)
model.add_marker("LLEG", "LANKM")
model.add_marker("LLEG", "LANK")

model.add_segment("LFOOT", parent_name="LLEG", rotations="x")
model.set_rt(
    segment_name="LFOOT",
    origin_markers=("LANKM", "LANK"),
    first_axis_name=Axis.Name.X,
    first_axis_markers=("LANK", "LANKM"),
    second_axis_name=Axis.Name.Y,
    second_axis_markers=(("LANKM", "LANK"), "LTOE"),
    axis_to_keep=Axis.Name.X,
)
model.add_marker("LFOOT", "LTOE")
model.add_marker("LFOOT", "L5MH")

# 'LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN', 'RSHO',
# 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
# 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE', 'LHUM', 'RHUM', 'LTROC', 'RTROC', 'LTHID', 'RTHID', 'LKNM', 'RKNM',
# 'LTIBP', 'LTIBD', 'RTIBP', 'RTIBD', 'LANKM', 'L5MH', 'RANKM', 'R5MH', '*51'

# Collapse the model to an actual personalized model
model.generate_personalized(c3d, kinematic_model_file_path)

bioviz.Viz(kinematic_model_file_path).exec()

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
