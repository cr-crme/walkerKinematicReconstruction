from biorbd import model_creation


class SimplePluginGait(model_creation.GenericBiomechanicalModel):
    """
    This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
    https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus)
    """
    def __init__(self):
        super(SimplePluginGait, self).__init__()
        self._define_kinematic_model()
        self._define_dynamic_model()

    def _define_kinematic_model(self):
        # Pelvis is verified
        # Thorax is verified
        # Head is verified

        self.add_segment(
            "Pelvis",
            translations="xyz",
            rotations="xyz",
            segment_coordinate_system=model_creation.SegmentCoordinateSystem(
                origin=model_creation.Equation(
                    marker_names=("LPSI", "RPSI", "LASI", "RASI"),
                    function=lambda LPSI, RPSI, LASI, RASI: (LPSI + RPSI + LASI + RASI) / 4,
                ),
                first_axis=(model_creation.Axis.Name.X, model_creation.Equation("RASI - (LPSI + RPSI) / 2")),
                second_axis=(model_creation.Axis.Name.Y, model_creation.Equation("LASI - RASI")),
                axis_to_keep=model_creation.Axis.Name.Y,
            ),
        )
        # self.add_marker("Pelvis", "SACR", is_technical=False, is_anatomical=True)
        self.add_marker("Pelvis", "LPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "LASI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RASI", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "Thorax",
        #     parent_name="Pelvis",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="CLAV",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=(("T10", "STRN"), ("C7", "CLAV")),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=(("T10", "C7"), ("STRN", "CLAV")),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("Thorax", "T10", is_technical=True, is_anatomical=True)
        # self.add_marker("Thorax", "C7", is_technical=True, is_anatomical=True)
        # self.add_marker("Thorax", "STRN", is_technical=True, is_anatomical=True)
        # self.add_marker("Thorax", "CLAV", is_technical=True, is_anatomical=True)
        # self.add_marker("Thorax", "RBAK", is_technical=False, is_anatomical=False)
        #
        # self.add_segment(
        #     "Head",
        #     parent_name="Thorax",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("LFHD", "RFHD"),
        #         first_axis_name=model_creation.Axis.Name.X,
        #         first_axis_markers=(("LBHD", "RBHD"), ("LFHD", "RFHD")),
        #         second_axis_name=model_creation.Axis.Name.Y,
        #         second_axis_markers=("RFHD", "LFHD"),
        #         axis_to_keep=model_creation.Axis.Name.X,
        #     ),
        # )
        # self.add_marker("Head", "LBHD", is_technical=True, is_anatomical=True)
        # self.add_marker("Head", "RBHD", is_technical=True, is_anatomical=True)
        # self.add_marker("Head", "LFHD", is_technical=True, is_anatomical=True)
        # self.add_marker("Head", "RFHD", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "Humerus",
        #     parent_name="Thorax",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="RSHO",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("RELB", "RSHO"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("RWRB", "RWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("Humerus", "RSHO", is_technical=True, is_anatomical=True)
        # self.add_marker("Humerus", "RELB", is_technical=True, is_anatomical=True)
        # self.add_marker("Humerus", "RHUM", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "RLOWER_ARM",
        #     parent_name="Humerus",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="RELB",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=(("RWRB", "RWRA"), "RELB"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("RWRB", "RWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("RLOWER_ARM", "RWRB", is_technical=True, is_anatomical=True)
        # self.add_marker("RLOWER_ARM", "RWRA", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "RHAND",
        #     parent_name="RLOWER_ARM",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("RWRB", "RWRA"),
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("RFIN", ("RWRB", "RWRA")),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("RWRB", "RWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("RHAND", "RFIN", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "LUPPER_ARM",
        #     parent_name="Thorax",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="LSHO",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("LELB", "LSHO"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("LWRB", "LWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("LUPPER_ARM", "LSHO", is_technical=True, is_anatomical=True)
        # self.add_marker("LUPPER_ARM", "LELB", is_technical=True, is_anatomical=True)
        # self.add_marker("LUPPER_ARM", "LHUM", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "LLOWER_ARM",
        #     parent_name="LUPPER_ARM",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="LELB",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=(("LWRB", "LWRA"), "LELB"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("LWRB", "LWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("LLOWER_ARM", "LWRB", is_technical=True, is_anatomical=True)
        # self.add_marker("LLOWER_ARM", "LWRA", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "LHAND",
        #     parent_name="LLOWER_ARM",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("LWRB", "LWRA"),
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("LFIN", ("LWRB", "LWRA")),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("LWRB", "LWRA"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("LHAND", "LFIN", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "RTHIGH",
        #     parent_name="Pelvis",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="RTROC",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("RKNE", "RTROC"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("RKNM", "RKNE"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("RTHIGH", "RTROC", is_technical=True, is_anatomical=True)
        # self.add_marker("RTHIGH", "RKNE", is_technical=True, is_anatomical=True)
        # self.add_marker("RTHIGH", "RKNM", is_technical=False, is_anatomical=True)
        # self.add_marker("RTHIGH", "RTHI", is_technical=True, is_anatomical=False)
        # self.add_marker("RTHIGH", "RTHID", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "RLEG",
        #     parent_name="RTHIGH",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("RKNM", "RKNE"),
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=(("RANKM", "RANK"), ("RKNM", "RKNE")),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("RKNM", "RKNE"),
        #         axis_to_keep=model_creation.Axis.Name.X,
        #     ),
        # )
        # self.add_marker("RLEG", "RANKM", is_technical=False, is_anatomical=True)
        # self.add_marker("RLEG", "RANK", is_technical=True, is_anatomical=True)
        # self.add_marker("RLEG", "RTIBP", is_technical=True, is_anatomical=False)
        # self.add_marker("RLEG", "RTIB", is_technical=True, is_anatomical=False)
        # self.add_marker("RLEG", "RTIBD", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "RFOOT",
        #     parent_name="RLEG",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("RANKM", "RANK"),
        #         first_axis_name=model_creation.Axis.Name.X,
        #         first_axis_markers=("RANKM", "RANK"),
        #         second_axis_name=model_creation.Axis.Name.Y,
        #         second_axis_markers=("RHEE", "RTOE"),
        #         axis_to_keep=model_creation.Axis.Name.X,
        #     ),
        # )
        # self.add_marker("RFOOT", "RTOE", is_technical=True, is_anatomical=True)
        # self.add_marker("RFOOT", "R5MH", is_technical=True, is_anatomical=True)
        # self.add_marker("RFOOT", "RHEE", is_technical=True, is_anatomical=True)
        #
        # self.add_segment(
        #     "LTHIGH",
        #     parent_name="Pelvis",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers="LTROC",
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=("LKNE", "LTROC"),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("LKNE", "LKNM"),
        #         axis_to_keep=model_creation.Axis.Name.Z,
        #     ),
        # )
        # self.add_marker("LTHIGH", "LTROC", is_technical=True, is_anatomical=True)
        # self.add_marker("LTHIGH", "LKNE", is_technical=True, is_anatomical=True)
        # self.add_marker("LTHIGH", "LKNM", is_technical=False, is_anatomical=True)
        # self.add_marker("LTHIGH", "LTHI", is_technical=True, is_anatomical=False)
        # self.add_marker("LTHIGH", "LTHID", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "LLEG",
        #     parent_name="LTHIGH",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("LKNM", "LKNE"),
        #         first_axis_name=model_creation.Axis.Name.Z,
        #         first_axis_markers=(("LANKM", "LANK"), ("LKNM", "LKNE")),
        #         second_axis_name=model_creation.Axis.Name.X,
        #         second_axis_markers=("LKNE", "LKNM"),
        #         axis_to_keep=model_creation.Axis.Name.X,
        #     ),
        # )
        # self.add_marker("LLEG", "LANKM", is_technical=False, is_anatomical=True)
        # self.add_marker("LLEG", "LANK", is_technical=True, is_anatomical=True)
        # self.add_marker("LLEG", "LTIBP", is_technical=True, is_anatomical=False)
        # self.add_marker("LLEG", "LTIB", is_technical=True, is_anatomical=False)
        # self.add_marker("LLEG", "LTIBD", is_technical=True, is_anatomical=False)
        #
        # self.add_segment(
        #     "LFOOT",
        #     parent_name="LLEG",
        #     rotations="xyz",
        #     segment_coordinate_system=model_creation.SegmentCoordinateSystem(
        #         origin_markers=("LANKM", "LANK"),
        #         first_axis_name=model_creation.Axis.Name.X,
        #         first_axis_markers=("LANK", "LANKM"),
        #         second_axis_name=model_creation.Axis.Name.Y,
        #         second_axis_markers=("LHEE", "LTOE"),
        #         axis_to_keep=model_creation.Axis.Name.X,
        #     ),
        # )
        # self.add_marker("LFOOT", "LTOE", is_technical=True, is_anatomical=True)
        # self.add_marker("LFOOT", "L5MH", is_technical=True, is_anatomical=True)
        # self.add_marker("LFOOT", "LHEE", is_technical=True, is_anatomical=True)

    def _define_dynamic_model(self):
        pass

    @property
    def dof_index(self) -> dict[str, int]:
        return {
            "LHip": 36,  # Left hip flexion
            "LKnee": 39,  # Left knee flexion
            "LAnkle": 42,  # Left ankle flexion
            "LAbsAnkle": 42,  # Left ankle flexion
            "RHip": 27,  # Right hip flexion
            "RKnee": 30,  # Right knee flexion
            "RAnkle": 33,  # Right ankle flexion
            "RAbsAnkle": 33,  # Right ankle flexion
            "LShoulder": 18,  # Left shoulder flexion
            "LElbow": 21,  # Left elbow flexion
            "LWrist": 24,  # Left wrist flexion
            "RShoulder": 9,  # Right shoulder flexion
            "RElbow": 12,  # Right elbow flexion
            "RWrist": 15,  # Right wrist flexion
            "LNeck": None,
            "RNeck": None,
            "LSpine": None,
            "RSpine": None,
            "LHead": None,
            "RHead": None,
            "LThorax": 6,  # Trunk flexion
            "RThorax": 6,  # Trunk flexion
            "LPelvis": 3,  # Pelvis flexion
            "RPelvis": 3,  # Pelvis flexion
        }
