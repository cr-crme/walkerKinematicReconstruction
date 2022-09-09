from biorbd.model_creation import Axis, BiomechanicalModel, SegmentCoordinateSystem


class SimplePluginGait(BiomechanicalModel):
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
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["LPSI"] + m["RPSI"] + m["LASI"] + m["RASI"]) / 4,
                first_axis=Axis(Axis.Name.X, start=lambda m: (m["LPSI"] + m["RPSI"]) / 2, end="RASI"),
                second_axis=Axis(Axis.Name.Y, start="RASI", end="LASI"),
                axis_to_keep=Axis.Name.Y,
            ),
        )
        # self.add_marker("Pelvis", "SACR", is_technical=False, is_anatomical=True)
        self.add_marker("Pelvis", "LPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "LASI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RASI", is_technical=True, is_anatomical=True)

        self.add_segment(
            "Thorax",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: m["CLAV"],
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m: (m["T10"] + m["STRN"]) / 2, end=lambda m: (m["C7"] + m["CLAV"]) / 2
                ),
                second_axis=Axis(
                    Axis.Name.X, start=lambda m: (m["T10"] + m["C7"]) / 2, end=lambda m: (m["STRN"] + m["CLAV"]) / 2
                ),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("Thorax", "T10", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "C7", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "STRN", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "CLAV", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "RBAK", is_technical=False, is_anatomical=False)

        self.add_segment(
            "Head",
            parent_name="Thorax",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["LFHD"] + m["RFHD"]) / 2,
                first_axis=Axis(
                    Axis.Name.X, start=lambda m: (m["LBHD"] + m["RBHD"]) / 2, end=lambda m: (m["LFHD"] + m["RFHD"]) / 2
                ),
                second_axis=Axis(Axis.Name.Y, start="RFHD", end="LFHD"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("Head", "LBHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "RBHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "LFHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "RFHD", is_technical=True, is_anatomical=True)

        self.add_segment(
            "Humerus",
            parent_name="Thorax",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="RSHO",
                first_axis=Axis(Axis.Name.Z, start="RELB", end="RSHO"),
                second_axis=Axis(Axis.Name.X, start="RWRB", end="RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("Humerus", "RSHO", is_technical=True, is_anatomical=True)
        self.add_marker("Humerus", "RELB", is_technical=True, is_anatomical=True)
        self.add_marker("Humerus", "RHUM", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RLOWER_ARM",
            parent_name="Humerus",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="RELB",
                first_axis=Axis(Axis.Name.Z, start=lambda m: (m["RWRB"] + m["RWRA"]) / 1, end="RELB"),
                second_axis=Axis(Axis.Name.X, start="RWRB", end="RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RLOWER_ARM", "RWRB", is_technical=True, is_anatomical=True)
        self.add_marker("RLOWER_ARM", "RWRA", is_technical=True, is_anatomical=True)

        self.add_segment(
            "RHAND",
            parent_name="RLOWER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["RWRB"] + m["RWRA"]) / 2,
                first_axis=Axis(Axis.Name.Z, start="RFIN", end=lambda m: (m["RWRB"] + m["RWRA"]) / 2),
                second_axis=Axis(Axis.Name.X, start="RWRB", end="RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RHAND", "RFIN", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LUPPER_ARM",
            parent_name="Thorax",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LSHO",
                first_axis=Axis(Axis.Name.Z, start="LELB", end="LSHO"),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LUPPER_ARM", "LSHO", is_technical=True, is_anatomical=True)
        self.add_marker("LUPPER_ARM", "LELB", is_technical=True, is_anatomical=True)
        self.add_marker("LUPPER_ARM", "LHUM", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LLOWER_ARM",
            parent_name="LUPPER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LELB",
                first_axis=Axis(Axis.Name.Z, start=lambda m: (m["LWRB"] + m["LWRA"]) / 2, end="LELB"),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LLOWER_ARM", "LWRB", is_technical=True, is_anatomical=True)
        self.add_marker("LLOWER_ARM", "LWRA", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LHAND",
            parent_name="LLOWER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["LWRB"] + m["LWRA"]) / 2,
                first_axis=Axis(Axis.Name.Z, start="LFIN", end=lambda m: (m["LWRB"] + m["LWRA"]) / 2),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LHAND", "LFIN", is_technical=True, is_anatomical=True)

        self.add_segment(
            "RTHIGH",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="RTROC",
                first_axis=Axis(Axis.Name.Z, start="RKNE", end="RTROC"),
                second_axis=Axis(Axis.Name.X, start="RKNM", end="RKNE"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RTHIGH", "RTROC", is_technical=True, is_anatomical=True)
        self.add_marker("RTHIGH", "RKNE", is_technical=True, is_anatomical=True)
        self.add_marker("RTHIGH", "RKNM", is_technical=False, is_anatomical=True)
        self.add_marker("RTHIGH", "RTHI", is_technical=True, is_anatomical=False)
        self.add_marker("RTHIGH", "RTHID", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RLEG",
            parent_name="RTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["RKNM"] + m["RKNE"]) / 2,
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m: (m["RANKM"] + m["RANK"]) / 2, end=lambda m: (m["RKNM"] + m["RKNE"]) / 2
                ),
                second_axis=Axis(Axis.Name.X, start="RKNM", end="RKNE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("RLEG", "RANKM", is_technical=False, is_anatomical=True)
        self.add_marker("RLEG", "RANK", is_technical=True, is_anatomical=True)
        self.add_marker("RLEG", "RTIBP", is_technical=True, is_anatomical=False)
        self.add_marker("RLEG", "RTIB", is_technical=True, is_anatomical=False)
        self.add_marker("RLEG", "RTIBD", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RFOOT",
            parent_name="RLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["RANKM"] + m["RANK"]) / 2,
                first_axis=Axis(Axis.Name.X, start="RANKM", end="RANK"),
                second_axis=Axis(Axis.Name.Y, start="RHEE", end="RTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("RFOOT", "RTOE", is_technical=True, is_anatomical=True)
        self.add_marker("RFOOT", "R5MH", is_technical=True, is_anatomical=True)
        self.add_marker("RFOOT", "RHEE", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LTHIGH",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LTROC",
                first_axis=Axis(Axis.Name.Z, start="LKNE", end="LTROC"),
                second_axis=Axis(Axis.Name.X, start="LKNE", end="LKNM"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LTHIGH", "LTROC", is_technical=True, is_anatomical=True)
        self.add_marker("LTHIGH", "LKNE", is_technical=True, is_anatomical=True)
        self.add_marker("LTHIGH", "LKNM", is_technical=False, is_anatomical=True)
        self.add_marker("LTHIGH", "LTHI", is_technical=True, is_anatomical=False)
        self.add_marker("LTHIGH", "LTHID", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LLEG",
            parent_name="LTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["LKNM"] + m["LKNE"]) / 2,
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m: (m["LANKM"] + m["LANK"]) / 2, end=lambda m: (m["LKNM"] + m["LKNE"]) / 2
                ),
                second_axis=Axis(Axis.Name.X, start="LKNE", end="LKNM"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("LLEG", "LANKM", is_technical=False, is_anatomical=True)
        self.add_marker("LLEG", "LANK", is_technical=True, is_anatomical=True)
        self.add_marker("LLEG", "LTIBP", is_technical=True, is_anatomical=False)
        self.add_marker("LLEG", "LTIB", is_technical=True, is_anatomical=False)
        self.add_marker("LLEG", "LTIBD", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LFOOT",
            parent_name="LLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m: (m["LANKM"] + m["LANK"]) / 2,
                first_axis=Axis(Axis.Name.X, start="LANK", end="LANKM"),
                second_axis=Axis(Axis.Name.Y, start="LHEE", end="LTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("LFOOT", "LTOE", is_technical=True, is_anatomical=True)
        self.add_marker("LFOOT", "L5MH", is_technical=True, is_anatomical=True)
        self.add_marker("LFOOT", "LHEE", is_technical=True, is_anatomical=True)

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
