import biorbd
from biorbd import KinematicModelGeneric, Axis, SegmentCoordinateSystem
import ezc3d
import numpy as np


def suffix_to_all(values: tuple[str, ...] | list[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{n}{suffix}" for n in values)


class WalkerModel:
    def __init__(self):
        self.generic_model = self._generate_generic_model()
        self.model = None

        self.is_kinematic_reconstructed: bool = False
        self.c3d: ezc3d.c3d | None = None
        self.q: np.ndarray = np.ndarray(())
        self.qdot: np.ndarray = np.ndarray(())
        self.qddot: np.ndarray = np.ndarray(())

    @property
    def is_model_loaded(self):
        return self.model is not None

    @staticmethod
    def _generate_generic_model():
        model = KinematicModelGeneric()
        model.add_segment(
            "PELVIS",
            translations="xyz",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="PELVIS",
                origin_markers=("LPSI", "RPSI", "LASI", "RASI"),
                first_axis_name=Axis.Name.Y,
                first_axis_markers=(("LPSI", "RPSI"), ("LASI", "RASI")),
                second_axis_name=Axis.Name.X,
                second_axis_markers=(("LPSI", "RPSI", "LASI", "RASI"), ("RPSI", "RASI")),
                axis_to_keep=Axis.Name.Y,
            ),
        )
        model.add_marker("PELVIS", "LPSI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "RPSI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "LASI", is_technical=True, is_anatomical=True)
        model.add_marker("PELVIS", "RASI", is_technical=True, is_anatomical=True)

        model.add_segment(
            "TRUNK",
            parent_name="PELVIS",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="TRUNK",
                origin_markers="CLAV",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=(("T10", "STRN"), ("C7", "CLAV")),
                second_axis_name=Axis.Name.Y,
                second_axis_markers=(("T10", "C7"), ("STRN", "CLAV")),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("TRUNK", "T10", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "C7", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "STRN", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "CLAV", is_technical=True, is_anatomical=True)
        model.add_marker("TRUNK", "RBAK", is_technical=False, is_anatomical=False)

        model.add_segment(
            "HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="HEAD",
                origin_markers=("LBHD", "RBHD", "LFHD", "RFHD"),
                first_axis_name=Axis.Name.X,
                first_axis_markers=(("LBHD", "LFHD"), ("RBHD", "RFHD")),
                second_axis_name=Axis.Name.Y,
                second_axis_markers=(("LBHD", "RBHD"), ("LFHD", "RFHD")),
                axis_to_keep=Axis.Name.Y,
            ),
        )
        model.add_marker("HEAD", "LBHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "RBHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "LFHD", is_technical=True, is_anatomical=True)
        model.add_marker("HEAD", "RFHD", is_technical=True, is_anatomical=True)

        model.add_segment(
            "RUPPER_ARM",
            parent_name="TRUNK",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RUPPER_ARM",
                origin_markers="RSHO",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("RELB", "RSHO"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("RWRB", "RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("RUPPER_ARM", "RSHO", is_technical=True, is_anatomical=True)
        model.add_marker("RUPPER_ARM", "RELB", is_technical=True, is_anatomical=True)
        model.add_marker("RUPPER_ARM", "RHUM", is_technical=True, is_anatomical=False)

        model.add_segment(
            "RLOWER_ARM",
            parent_name="RUPPER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RLOWER_ARM",
                origin_markers="RELB",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=(("RWRB", "RWRA"), "RELB"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("RWRB", "RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("RLOWER_ARM", "RWRB", is_technical=True, is_anatomical=True)
        model.add_marker("RLOWER_ARM", "RWRA", is_technical=True, is_anatomical=True)

        model.add_segment(
            "RHAND",
            parent_name="RLOWER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RHAND",
                origin_markers=("RWRB", "RWRA"),
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("RFIN", ("RWRB", "RWRA")),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("RWRB", "RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("RHAND", "RFIN", is_technical=True, is_anatomical=True)

        model.add_segment(
            "LUPPER_ARM",
            parent_name="TRUNK",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LUPPER_ARM",
                origin_markers="LSHO",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("LELB", "LSHO"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("LWRB", "LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("LUPPER_ARM", "LSHO", is_technical=True, is_anatomical=True)
        model.add_marker("LUPPER_ARM", "LELB", is_technical=True, is_anatomical=True)
        model.add_marker("LUPPER_ARM", "LHUM", is_technical=True, is_anatomical=False)

        model.add_segment(
            "LLOWER_ARM",
            parent_name="LUPPER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LLOWER_ARM",
                origin_markers="LELB",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=(("LWRB", "LWRA"), "LELB"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("LWRB", "LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("LLOWER_ARM", "LWRB", is_technical=True, is_anatomical=True)
        model.add_marker("LLOWER_ARM", "LWRA", is_technical=True, is_anatomical=True)

        model.add_segment(
            "LHAND",
            parent_name="LLOWER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LHAND",
                origin_markers=("LWRB", "LWRA"),
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("LFIN", ("LWRB", "LWRA")),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("LWRB", "LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("LHAND", "LFIN", is_technical=True, is_anatomical=True)

        model.add_segment(
            "RTHIGH",
            parent_name="PELVIS",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RTHIGH",
                origin_markers="RTROC",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("RKNE", "RTROC"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("RKNM", "RKNE"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("RTHIGH", "RTROC", is_technical=True, is_anatomical=True)
        model.add_marker("RTHIGH", "RKNE", is_technical=True, is_anatomical=True)
        model.add_marker("RTHIGH", "RKNM", is_technical=False, is_anatomical=True)
        model.add_marker("RTHIGH", "RTHI", is_technical=True, is_anatomical=False)
        model.add_marker("RTHIGH", "RTHID", is_technical=True, is_anatomical=False)

        model.add_segment(
            "RLEG",
            parent_name="RTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RLEG",
                origin_markers=("RKNM", "RKNE"),
                first_axis_name=Axis.Name.Z,
                first_axis_markers=(("RANKM", "RANK"), ("RKNM", "RKNE")),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("RKNM", "RKNE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        model.add_marker("RLEG", "RANKM", is_technical=False, is_anatomical=True)
        model.add_marker("RLEG", "RANK", is_technical=True, is_anatomical=True)
        model.add_marker("RLEG", "RTIBP", is_technical=True, is_anatomical=False)
        model.add_marker("RLEG", "RTIB", is_technical=True, is_anatomical=False)
        model.add_marker("RLEG", "RTIBD", is_technical=True, is_anatomical=False)

        model.add_segment(
            "RFOOT",
            parent_name="RLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="RFOOT",
                origin_markers=("RANKM", "RANK"),
                first_axis_name=Axis.Name.X,
                first_axis_markers=("RANKM", "RANK"),
                second_axis_name=Axis.Name.Y,
                second_axis_markers=("RHEE", "RTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        model.add_marker("RFOOT", "RTOE", is_technical=True, is_anatomical=True)
        model.add_marker("RFOOT", "R5MH", is_technical=True, is_anatomical=True)
        model.add_marker("RFOOT", "RHEE", is_technical=True, is_anatomical=True)

        model.add_segment(
            "LTHIGH",
            parent_name="PELVIS",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LTHIGH",
                origin_markers="LTROC",
                first_axis_name=Axis.Name.Z,
                first_axis_markers=("LKNE", "LTROC"),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("LKNE", "LKNM"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        model.add_marker("LTHIGH", "LTROC", is_technical=True, is_anatomical=True)
        model.add_marker("LTHIGH", "LKNE", is_technical=True, is_anatomical=True)
        model.add_marker("LTHIGH", "LKNM", is_technical=False, is_anatomical=True)
        model.add_marker("LTHIGH", "LTHI", is_technical=True, is_anatomical=False)
        model.add_marker("LTHIGH", "LTHID", is_technical=True, is_anatomical=False)

        model.add_segment(
            "LLEG",
            parent_name="LTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LLEG",
                origin_markers=("LKNM", "LKNE"),
                first_axis_name=Axis.Name.Z,
                first_axis_markers=(("LANKM", "LANK"), ("LKNM", "LKNE")),
                second_axis_name=Axis.Name.X,
                second_axis_markers=("LKNE", "LKNM"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        model.add_marker("LLEG", "LANKM", is_technical=False, is_anatomical=True)
        model.add_marker("LLEG", "LANK", is_technical=True, is_anatomical=True)
        model.add_marker("LLEG", "LTIBP", is_technical=True, is_anatomical=False)
        model.add_marker("LLEG", "LTIB", is_technical=True, is_anatomical=False)
        model.add_marker("LLEG", "LTIBD", is_technical=True, is_anatomical=False)

        model.add_segment(
            "LFOOT",
            parent_name="LLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                segment_name="LFOOT",
                origin_markers=("LANKM", "LANK"),
                first_axis_name=Axis.Name.X,
                first_axis_markers=("LANK", "LANKM"),
                second_axis_name=Axis.Name.Y,
                second_axis_markers=("LHEE", "LTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        model.add_marker("LFOOT", "LTOE", is_technical=True, is_anatomical=True)
        model.add_marker("LFOOT", "L5MH", is_technical=True, is_anatomical=True)
        model.add_marker("LFOOT", "LHEE", is_technical=True, is_anatomical=True)
        return model

    def generate_personalized_model(
        self, static_trial_path: str, model_path: str, first_frame: int = 0, last_frame: int = -1
    ):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial_path
            The path of the c3d file of the static trial
        model_path
            The path of the generated bioMod file
        first_frame
            The first frame of the data to use
        last_frame
            The last frame of the data to use
        """

        self.generic_model.generate_personalized(
            data_path=static_trial_path, save_path=model_path, first_frame=first_frame, last_frame=last_frame
        )
        self.model = biorbd.Model(model_path)

    def reconstruct_kinematics(self, trial: str) -> np.ndarray:
        """
        Reconstruct the kinematics of the specified trial assuming a biorbd model is loaded

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from

        Returns
        -------
        The matrix nq x ntimes of the reconstructed kinematics
        """

        if not self.is_model_loaded:
            raise RuntimeError("The biorbd model must be loaded. You can do so by calling generate_personalized_model")

        marker_names = tuple(n.to_string() for n in self.model.technicalMarkerNames())

        self.c3d = ezc3d.c3d(trial)
        labels = self.c3d["parameters"]["POINT"]["LABELS"]["value"]
        data = self.c3d["data"]["points"]
        n_frames = data.shape[2]
        index_in_c3d = np.array(tuple(labels.index(name) if name in labels else -1 for name in marker_names))
        markers_in_c3d = np.ndarray((3, len(index_in_c3d), n_frames)) * np.nan
        markers_in_c3d[:, index_in_c3d >= 0, :] = data[:3, index_in_c3d[index_in_c3d >= 0], :] / 1000  # To meter

        # Create a Kalman filter structure
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        params = biorbd.KalmanParam(freq)
        kalman = biorbd.KalmanReconsMarkers(self.model, params)

        # Perform the kalman filter for each frame (the first frame is much longer than the next)
        q = biorbd.GeneralizedCoordinates(self.model)
        qdot = biorbd.GeneralizedVelocity(self.model)
        qddot = biorbd.GeneralizedAcceleration(self.model)
        self.q = np.ndarray((self.model.nbQ(), n_frames))
        self.qdot = np.ndarray((self.model.nbQ(), n_frames))
        self.qddot = np.ndarray((self.model.nbQ(), n_frames))
        for i in range(n_frames):
            kalman.reconstructFrame(self.model, np.reshape(markers_in_c3d[:, :, i].T, -1), q, qdot, qddot)
            self.q[:, i] = q.to_array()
            self.qdot[:, i] = qdot.to_array()
            self.qddot[:, i] = qddot.to_array()

        self.is_kinematic_reconstructed = True
        return self.q

    @property
    def joint_angle_names(self) -> tuple[str, ...]:
        return (
            "LHip",
            "LKnee",
            "LAnkle",
            "LAbsAnkle",
            "RHip",
            "RKnee",
            "RAnkle",
            "RAbsAnkle",
            "LShoulder",
            "LElbow",
            "LWrist",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LNeck",
            "RNeck",
            "LSpine",
            "RSpine",
            "LHead",
            "RHead",
            "LThorax",
            "RThorax",
            "LPelvis",
            "RPelvis",
        )

    def to_c3d(self):
        if not self.is_kinematic_reconstructed:
            raise RuntimeError("Kinematics should be reconstructed before writing to c3d. "
                               "Please call 'kinematic_reconstruction'")

        c3d = ezc3d.c3d()

        # Fill it with points, angles, power, force, moment
        c3d['parameters']['POINT']['RATE']['value'] = [int(self.c3d["parameters"]["POINT"]["RATE"]["value"][0])]
        point_names = [name.to_string() for name in self.model.markerNames()]
        point_names.extend(suffix_to_all(self.joint_angle_names, "Angles"))
        point_names.extend(suffix_to_all(self.joint_angle_names, "Power"))
        point_names.extend(suffix_to_all(self.joint_angle_names, "Force"))
        point_names.extend(suffix_to_all(self.joint_angle_names, "Moment"))

        # Transfer the marker data to the new c3d
        c3d['parameters']['POINT']['LABELS']['value'] = point_names
        n_frame = self.c3d["header"]["points"]["last_frame"] - self.c3d["header"]["points"]["first_frame"] + 1
        data = np.ndarray((4, len(point_names), n_frame)) * np.nan
        data[3, ...] = 1
        for i, name_in_c3d in enumerate(self.c3d["parameters"]["POINT"]["LABELS"]["value"]):
            if name_in_c3d[0] == "*":
                continue
            # Make sure it is in the right order
            data[:, point_names.index(name_in_c3d), :] = self.c3d["data"]["points"][:, i, :]
        # Todo: put data in Angles, Power, Force and Moment

        # Dispatch the analog data
        c3d['parameters']['ANALOG']['RATE']['value'][0] = self.c3d['parameters']['ANALOG']['RATE']['value'][0]
        # TODO: RENDU ICI
        c3d['parameters']['ANALOG']['LABELS']['value'] = (
        'analog1', 'analog2', 'analog3', 'analog4', 'analog5', 'analog6')
        c3d['data']['analogs'] = np.random.rand(1, 6, 1000)
        c3d['data']['analogs'][0, 0, :] = 4
        c3d['data']['analogs'][0, 1, :] = 5
        c3d['data']['analogs'][0, 2, :] = 6
        c3d['data']['analogs'][0, 3, :] = 7
        c3d['data']['analogs'][0, 4, :] = 8
        c3d['data']['analogs'][0, 5, :] = 9

        # Add a custom parameter to the POINT group
        c3d.add_parameter("POINT", "newParam", [1, 2, 3])

        # Add a custom parameter a new group
        c3d.add_parameter("NewGroup", "newParam", ["MyParam1", "MyParam2"])

        # Write the data
        c3d.write("path_to_c3d.c3d")