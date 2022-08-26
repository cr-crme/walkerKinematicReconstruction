import biorbd
from biorbd import KinematicModelGeneric, Axis, SegmentCoordinateSystem
import ezc3d
import numpy as np
from matplotlib import pyplot as plt
import scipy

from .misc import differentiate


def suffix_to_all(values: tuple[str, ...] | list[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{n}{suffix}" for n in values)


class WalkerModel:
    def __init__(self):
        self.generic_model = self._generate_generic_model()
        self.model = None

        self.is_kinematic_reconstructed: bool = False
        self.c3d: ezc3d.c3d | None = None
        self.t: np.ndarray = np.ndarray(())
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

        self.c3d = ezc3d.c3d(trial, extract_forceplat_data=True)
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
        frame_rate = self.c3d["header"]["points"]["frame_rate"]
        first_frame = self.c3d["header"]["points"]["first_frame"]
        last_frame = self.c3d["header"]["points"]["last_frame"]
        self.t = np.linspace(first_frame / frame_rate, last_frame / frame_rate, last_frame - first_frame + 1)
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
    def joint_angle_names(self) -> dict[str, int]:
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

    def markers_to_array(self, q: np.ndarray) -> np.ndarray:
        """
        Get all markers position from a position q in the format (3 x NMarker x NTime)
        """
        markers = np.ndarray((3, self.model.nbMarkers(), len(q)))
        for i, q_tp in enumerate(q):
            markers[:, :, i] = np.array([mark.to_array() for mark in self.model.markers(q_tp)]).T
        return markers

    def _find_events(self) -> tuple[int, tuple[str, ...], tuple[str, ...], tuple[tuple[float, ...], tuple[float, ...]]]:
        """
        Returns
        -------
        number of events (int)
            The number of events
        event_contexts (tuple[str, ...])
            If a specific event arrived the on 'Left' or on the 'Right'
        event_labels (tuple[str, ...])
            If a specific event is a 'Foot Strike' or a Foot Off'
        event_times (tuple[tuple[float, ...], tuple[float]])
            The time for a specific event. the first row should be all zeros for some unknown reason
        """

        def find_foot_events(heel_marker_name: str, toe_marker_name: str):
            # The algorithm is to take the lowest velocity of the heel,
            # then find the first time this velocity hits 0; this is the heel strike.
            # The maximum velocity after that point is just prior to the toe off. Therefore take the medium point between
            # highest toe velocity and that point (TODO)

            markers = biorbd.markers_to_array(self.model, self.q)
            heel_idx = biorbd.marker_index(self.model, heel_marker_name)
            toe_idx = biorbd.marker_index(self.model, toe_marker_name)
            heel_height = markers[(2,), heel_idx, :]
            heel_velocity = differentiate(heel_height, self.t[1] - self.t[0])
            toe_height = markers[(2,), toe_idx, :]
            toe_velocity = differentiate(toe_height, self.t[1] - self.t[0])

            t_peaks_strike = []
            t_peak_preheelstrike = scipy.signal.find_peaks(-heel_velocity[0, :], height=0.5)[0]
            for i in t_peak_preheelstrike:
                # find the first time the signal crosses 0 from that lowest point
                t_peaks_strike.append(i + np.argmax(np.diff(np.sign(heel_velocity[:, i:])) != 0))

            t_peaks_toeoff = scipy.signal.find_peaks(heel_velocity[0, :], height=0.5)[0]

            # plt.plot(self.t, heel_height[0, :])
            plt.plot(self.t, toe_velocity[0, :])
            plt.plot(self.t[t_peaks_strike], toe_velocity[0, t_peaks_strike], 'bo')
            plt.plot(self.t[t_peaks_toeoff], toe_velocity[0, t_peaks_toeoff], 'ro')


            f = 0
            ratio = int(self.c3d["header"]["analogs"]["frame_rate"]/self.c3d["header"]["points"]["frame_rate"])
            force_data = self.c3d["data"]["platform"][f]["force"][2, ::ratio]
            max_force = max(force_data)
            plt.plot(self.t, force_data / max_force)

            # f = 1
            # ratio = int(self.c3d["header"]["analogs"]["frame_rate"]/self.c3d["header"]["points"]["frame_rate"])
            # force_data = self.c3d["data"]["platform"][f]["force"][2, ::ratio]
            # max_force = max(force_data)
            # plt.plot(force_data / max_force)
            #
            # f = 2
            # ratio = int(self.c3d["header"]["analogs"]["frame_rate"]/self.c3d["header"]["points"]["frame_rate"])
            # force_data = self.c3d["data"]["platform"][f]["force"][2, ::ratio]
            # max_force = max(force_data)
            # plt.plot(force_data / max_force)


        # right_foot_strikes = find_foot_events("RTOE")
        left_foot_strikes = find_foot_events("LHEE", "LTOE")

        plt.show()

        events_number = 3
        events_contexts = ("Left", "Right", "Left")
        events_labels = ("Foot Strike", "Foot Strike", "Foot Off")
        events_times = ((0, 0, 0), (-1, -2, -3))

        return events_number, events_contexts, events_labels, events_times

    def to_c3d(self):
        if not self.is_kinematic_reconstructed:
            raise RuntimeError("Kinematics should be reconstructed before writing to c3d. "
                               "Please call 'kinematic_reconstruction'")

        c3d = ezc3d.c3d()

        # Fill it with points, angles, power, force, moment
        c3d["parameters"]["POINT"]["RATE"]["value"] = [int(self.c3d["parameters"]["POINT"]["RATE"]["value"][0])]
        c3d.add_parameter("POINT", "ANGLE_UNITS", ["deg"])
        point_names = [name.to_string() for name in self.model.markerNames()]
        point_names.extend(suffix_to_all(tuple(self.joint_angle_names.keys()), "Angles"))
        point_names.extend(suffix_to_all(tuple(self.joint_angle_names.keys()), "Power"))
        point_names.extend(suffix_to_all(tuple(self.joint_angle_names.keys()), "Force"))
        point_names.extend(suffix_to_all(tuple(self.joint_angle_names.keys()), "Moment"))
        c3d["parameters"]["POINT"]["UNITS"] = self.c3d["parameters"]["POINT"]["UNITS"]

        # Transfer the marker data to the new c3d
        c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
        n_frame = self.c3d["header"]["points"]["last_frame"] - self.c3d["header"]["points"]["first_frame"] + 1
        data = np.ndarray((4, len(point_names), n_frame)) * np.nan
        data[3, ...] = 1
        for i, name_in_c3d in enumerate(self.c3d["parameters"]["POINT"]["LABELS"]["value"]):
            if name_in_c3d[0] == "*":
                continue
            # Make sure it is in the right order
            data[:, point_names.index(name_in_c3d), :] = self.c3d["data"]["points"][:, i, :]

        # Dispatch the kinematics and kinematics
        # Todo: put data in Power, Force and Moment
        for joint, idx in self.joint_angle_names.items():
            if idx is None:
                continue
            data[:, point_names.index(f"{joint}Angles"), :] = self.q[idx, :]
        c3d["data"]["points"] = data

        # Find and add events
        events_number, events_contexts, events_labels, events_times = self._find_events()
        c3d.add_parameter("EVENT", "USED", (events_number,))
        c3d.add_parameter("EVENT", "CONTEXTS", events_contexts)
        c3d.add_parameter("EVENT", "LABELS", events_labels)
        c3d.add_parameter("EVENT", "TIMES", events_times)

        # Dispatch the analog and force_plate data
        c3d["parameters"]["ANALOG"] = self.c3d["parameters"]["ANALOG"]
        for i, label in enumerate(self.c3d[ "parameters"]["ANALOG"]["LABELS"]["value"]):
            label = label.replace("Force.", "")
            label = label.replace("Moment.", "")
            c3d["parameters"]["ANALOG"]["LABELS"]["value"][i] = label

        c3d["parameters"]["FORCE_PLATFORM"] = self.c3d["parameters"]["FORCE_PLATFORM"]
        c3d["data"]["analogs"] = self.c3d["data"]["analogs"]

        # Write the data
        c3d.write("/home/pariterre/Programmation/gaitAnalysisGUI/data/path_to_c3d.c3d")
