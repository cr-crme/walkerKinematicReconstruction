from biorbd import KinematicModelGeneric, Axis
import ezc3d


class WalkerModel:
    def __init__(self):
        self.model = self._generate_generic_model()

    @staticmethod
    def _generate_generic_model():
        model = KinematicModelGeneric()
        model.add_segment("PELVIS", translations="yz", rotations="xyz")
        model.set_rt(
            segment_name="PELVIS",
            origin_markers=("LPSI", "RPSI", "LASI", "RASI"),
            first_axis_name=Axis.Name.Y,
            first_axis_markers=(("LPSI", "RPSI"), ("LASI", "RASI")),
            second_axis_name=Axis.Name.X,
            second_axis_markers=(("LPSI", "RPSI", "LASI", "RASI"), ("RPSI", "RASI")),
            axis_to_keep=Axis.Name.Y,
        )
        model.add_marker("PELVIS", "LPSI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "RPSI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "LASI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "RASI", is_technical=True, is_anatomical=True)

        model.add_segment("TRUNK", parent_name="PELVIS", rotations="xyz")
        model.set_rt(
            segment_name="TRUNK",
            origin_markers="CLAV",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=(("T10", "STRN"), ("C7", "CLAV")),
            second_axis_name=Axis.Name.Y,
            second_axis_markers=(("T10", "C7"), ("STRN", "CLAV")),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("TRUNK", "T10", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "C7", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "STRN", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "CLAV", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "RBAK", is_technical=False, is_anatomical=False)

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
        model.add_marker("HEAD", "LBHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "RBHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "LFHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "RFHD", is_technical=True, is_anatomical=True)

        model.add_segment("RUPPER_ARM", parent_name="TRUNK", rotations="xyz")
        model.set_rt(
            segment_name="RUPPER_ARM",
            origin_markers="RSHO",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("RELB", "RSHO"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("RWRB", "RWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("RUPPER_ARM", "RSHO", is_technical=True, is_anatomical=True)
        model.add_marker("RUPPER_ARM", "RELB", is_technical=True, is_anatomical=True)
        model.add_marker("RUPPER_ARM", "RHUM", is_technical=True, is_anatomical=False)

        model.add_segment("RLOWER_ARM", parent_name="RUPPER_ARM", rotations="xyz")
        model.set_rt(
            segment_name="RLOWER_ARM",
            origin_markers="RELB",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=(("RWRB", "RWRA"), "RELB"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("RWRB", "RWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("RLOWER_ARM", "RWRB", is_technical=True, is_anatomical=True)
        model.add_marker("RLOWER_ARM", "RWRA", is_technical=True, is_anatomical=True)

        model.add_segment("RHAND", parent_name="RLOWER_ARM", rotations="xyz")
        model.set_rt(
            segment_name="RHAND",
            origin_markers=("RWRB", "RWRA"),
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("RFIN", ("RWRB", "RWRA")),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("RWRB", "RWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("RHAND", "RFIN", is_technical=True, is_anatomical=True)

        model.add_segment("LUPPER_ARM", parent_name="TRUNK", rotations="xyz")
        model.set_rt(
            segment_name="LUPPER_ARM",
            origin_markers="LSHO",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("LELB", "LSHO"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("LWRB", "LWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("LUPPER_ARM", "LSHO", is_technical=True, is_anatomical=True)
        model.add_marker("LUPPER_ARM", "LELB", is_technical=True, is_anatomical=True)
        model.add_marker("LUPPER_ARM", "LHUM", is_technical=True, is_anatomical=False)

        model.add_segment("LLOWER_ARM", parent_name="LUPPER_ARM", rotations="xyz")
        model.set_rt(
            segment_name="LLOWER_ARM",
            origin_markers="LELB",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=(("LWRB", "LWRA"), "LELB"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("LWRB", "LWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("LLOWER_ARM", "LWRB", is_technical=True, is_anatomical=True)
        model.add_marker("LLOWER_ARM", "LWRA", is_technical=True, is_anatomical=True)

        model.add_segment("LHAND", parent_name="LLOWER_ARM", rotations="xyz")
        model.set_rt(
            segment_name="LHAND",
            origin_markers=("LWRB", "LWRA"),
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("LFIN", ("LWRB", "LWRA")),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("LWRB", "LWRA"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("LHAND", "LFIN", is_technical=True, is_anatomical=True)

        model.add_segment("RTHIGH", parent_name="PELVIS", rotations="xyz")
        model.set_rt(
            segment_name="RTHIGH",
            origin_markers="RTROC",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("RKNE", "RTROC"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("RKNM", "RKNE"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("RTHIGH", "RTROC", is_technical=True, is_anatomical=True)
        model.add_marker("RTHIGH", "RKNE", is_technical=True, is_anatomical=True)
        model.add_marker("RTHIGH", "RKNM", is_technical=False, is_anatomical=True)
        model.add_marker("RTHIGH", "RTHI", is_technical=True, is_anatomical=False)
        model.add_marker("RTHIGH", "RTHID", is_technical=True, is_anatomical=False)

        model.add_segment("RLEG", parent_name="RTHIGH", rotations="xyz")
        model.set_rt(
            segment_name="RLEG",
            origin_markers=("RKNM", "RKNE"),
            first_axis_name=Axis.Name.Z,
            first_axis_markers=(("RANKM", "RANK"), ("RKNM", "RKNE")),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("RKNM", "RKNE"),
            axis_to_keep=Axis.Name.X,
        )
        model.add_marker("RLEG", "RANKM", is_technical=False, is_anatomical=True)
        model.add_marker("RLEG", "RANK", is_technical=True, is_anatomical=True)
        model.add_marker("RLEG", "RTIBP", is_technical=True, is_anatomical=False)
        model.add_marker("RLEG", "RTIB", is_technical=True, is_anatomical=False)
        model.add_marker("RLEG", "RTIBD", is_technical=True, is_anatomical=False)

        model.add_segment("RFOOT", parent_name="RLEG", rotations="xyz")
        model.set_rt(
            segment_name="RFOOT",
            origin_markers=("RANKM", "RANK"),
            first_axis_name=Axis.Name.X,
            first_axis_markers=("RANKM", "RANK"),
            second_axis_name=Axis.Name.Y,
            second_axis_markers=("RHEE", "RTOE"),
            axis_to_keep=Axis.Name.X,
        )
        model.add_marker("RFOOT", "RTOE", is_technical=True, is_anatomical=True)
        model.add_marker("RFOOT", "R5MH", is_technical=True, is_anatomical=True)
        model.add_marker("RFOOT", "RHEE", is_technical=True, is_anatomical=True)

        model.add_segment("LTHIGH", parent_name="PELVIS", rotations="xyz")
        model.set_rt(
            segment_name="LTHIGH",
            origin_markers="LTROC",
            first_axis_name=Axis.Name.Z,
            first_axis_markers=("LKNE", "LTROC"),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("LKNE", "LKNM"),
            axis_to_keep=Axis.Name.Z,
        )
        model.add_marker("LTHIGH", "LTROC", is_technical=True, is_anatomical=True)
        model.add_marker("LTHIGH", "LKNE", is_technical=True, is_anatomical=True)
        model.add_marker("LTHIGH", "LKNM", is_technical=False, is_anatomical=True)
        model.add_marker("LTHIGH", "LTHI", is_technical=True, is_anatomical=False)
        model.add_marker("LTHIGH", "LTHID", is_technical=True, is_anatomical=False)

        model.add_segment("LLEG", parent_name="LTHIGH", rotations="xyz")
        model.set_rt(
            segment_name="LLEG",
            origin_markers=("LKNM", "LKNE"),
            first_axis_name=Axis.Name.Z,
            first_axis_markers=(("LANKM", "LANK"), ("LKNM", "LKNE")),
            second_axis_name=Axis.Name.X,
            second_axis_markers=("LKNE", "LKNM"),
            axis_to_keep=Axis.Name.X,
        )
        model.add_marker("LLEG", "LANKM", is_technical=False, is_anatomical=True)
        model.add_marker("LLEG", "LANK", is_technical=True, is_anatomical=True)
        model.add_marker("LLEG", "LTIBP", is_technical=True, is_anatomical=False)
        model.add_marker("LLEG", "LTIB", is_technical=True, is_anatomical=False)
        model.add_marker("LLEG", "LTIBD", is_technical=True, is_anatomical=False)

        model.add_segment("LFOOT", parent_name="LLEG", rotations="xyz")
        model.set_rt(
            segment_name="LFOOT",
            origin_markers=("LANKM", "LANK"),
            first_axis_name=Axis.Name.X,
            first_axis_markers=("LANK", "LANKM"),
            second_axis_name=Axis.Name.Y,
            second_axis_markers=("LHEE", "LTOE"),
            axis_to_keep=Axis.Name.X,
        )
        model.add_marker("LFOOT", "LTOE", is_technical=True, is_anatomical=True)
        model.add_marker("LFOOT", "L5MH", is_technical=True, is_anatomical=True)
        model.add_marker("LFOOT", "LHEE", is_technical=True, is_anatomical=True)
        return model

    def generate_personalized(self, static_trial_path: str, model_path: str):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial_path
            The path of the c3d file of the static trial
        model_path
            The path of the generated bioMod file
        """

        self.model.generate_personalized(c3d=ezc3d.c3d(static_trial_path), save_path=model_path)
