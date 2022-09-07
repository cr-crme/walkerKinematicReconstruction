import itertools

import biorbd
import biorbd as brbd
from biorbd import model_creation
import ezc3d
import numpy as np
import scipy

from .misc import differentiate
from .plugin_gait_kinematic_model import SimplePluginGait


def suffix_to_all(values: tuple[str, ...] | list[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{n}{suffix}" for n in values)


class BiomechanicsTools:
    def __init__(self):
        self.generic_model = SimplePluginGait()
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

    def _generate_center_of_mass(self):
        pass

    def personalize_model(self, static_trial: str, model_path: str):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial
            The path of the c3d file of the static trial to create the model from
        model_path
            The path of the generated bioMod file
        """

        self.generic_model.write(save_path=model_path, data=model_creation.C3dData(static_trial))
        self.model = brbd.Model(model_path)

    def reconstruct_kinematics(self, trial: str) -> np.ndarray:
        """
        Reconstruct the kinematics of the specified trial assuming a biorbd model is loaded using a Kalman filter

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

        self.c3d = ezc3d.c3d(trial, extract_forceplat_data=True)
        self.t, self.q, self.qdot, self.qddot = biorbd.extended_kalman_filter(self.model, trial)
        self.is_kinematic_reconstructed = True

        return self.q

    def find_feet_events(
        self
    ) -> tuple[int, tuple[str, ...], tuple[str, ...], tuple[tuple[float, ...], tuple[float, ...]]]:
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
            """
            Finds the events where the foot interacts with the ground. The return is expected to have an equal number
            of heel strikes and toe off.

            The algorithm is to take the lowest velocity of the heel, then to find the first time this velocity hits 0;
            this is the heel strike.
            The maximum heel velocity after that point is prior to the toe off and the highest toe velocity is just
            after. The toe off is therefore 80% of that distance towards the max velocity

            Parameters
            ----------
            heel_marker_name
                The name of the heel marker in the model
            toe_marker_name
                The name of the toe marker in the model

            Returns
            -------
            The 'heel strikes' and 'toe off' event. The number of each is expected to be equal
            """

            markers = brbd.markers_to_array(self.model, self.q)
            heel_idx = brbd.marker_index(self.model, heel_marker_name)
            toe_idx = brbd.marker_index(self.model, toe_marker_name)
            heel_height = markers[(2,), heel_idx, :]
            heel_velocity = differentiate(heel_height, self.t[1] - self.t[0])
            toe_height = markers[(2,), toe_idx, :]
            toe_velocity = differentiate(toe_height, self.t[1] - self.t[0])
            toe_acceleration = differentiate(toe_velocity, self.t[1] - self.t[0])

            idx_peaks_heel_strike = []
            idx_peak_pre_heel_strike = scipy.signal.find_peaks(-heel_velocity[0, :], height=0.5)[0]
            for heel_idx in idx_peak_pre_heel_strike:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_heel_strike.append(heel_idx + np.argmax(np.diff(np.sign(heel_velocity[:, heel_idx:])) != 0))

            idx_peaks_toe_off = []
            t_peaks_pre_toe_off = scipy.signal.find_peaks(heel_velocity[0, :], height=0.5)[0]
            for toe_idx in t_peaks_pre_toe_off:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_post_toe_off = np.argmax(np.diff(np.sign(toe_acceleration[:, toe_idx:])) != 0)
                idx_peaks_toe_off.append(int(toe_idx + 0.8 * idx_peaks_post_toe_off))

            # Associate each heel strike with its toe off
            first_toe_off_idx = -1
            for i, toe in enumerate(idx_peaks_toe_off):
                if toe > idx_peaks_heel_strike[0]:
                    first_toe_off_idx = i
                    break
            last_heel_strike_idx = -1
            for i, heel in enumerate(reversed(idx_peaks_heel_strike)):
                if heel < idx_peaks_toe_off[-1]:
                    last_heel_strike_idx = len(idx_peaks_heel_strike) - i
                    break

            if first_toe_off_idx == -1 or last_heel_strike_idx == -1:
                Warning('No heel strikes that correspond to the toe offs were found')

            heel_strikes = idx_peaks_heel_strike[:last_heel_strike_idx]
            toe_off = idx_peaks_toe_off[first_toe_off_idx:]
            if len(heel_strikes) != len(toe_off):
                Warning('The number of heel strikes and toe off does not match')

            return heel_strikes, toe_off

        left_foot_events = find_foot_events("LHEE", "LTOE")
        right_foot_events = find_foot_events("RHEE", "RTOE")
        # From that point, it is assumed that `len(events[0]) == len(events[1])`, that is there are equal number of
        # foot strikes and toe off

        events_number = (len(left_foot_events[0]) + len(right_foot_events[0])) * 2
        events_contexts = ("Left",) * len(left_foot_events[0]) * 2 + ("Right",) * len(right_foot_events[0]) * 2
        events_labels = ("Foot Strike", "Foot Off") * int(events_number / 2)
        events_times = (
            (0,) * events_number,
            self.t[np.array(tuple(itertools.chain(  # flatten the left/right, heel strike/toe off
                *[[heel, toe] for heel, toe in zip(*left_foot_events)] +
                [[heel, toe] for heel, toe in zip(*right_foot_events)]
            )))]
        )
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
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Angles"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Power"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Force"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Moment"))
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
        # Todo: put data of Power, Force and Moment
        for dof, idx in self.generic_model.dof_index.items():
            if idx is None:
                continue
            data[:, point_names.index(f"{dof}Angles"), :] = self.q[idx, :]
        c3d["data"]["points"] = data

        # Find and add events
        events_number, events_contexts, events_labels, events_times = self.find_feet_events()
        c3d.add_parameter("EVENT", "USED", (events_number,))
        c3d.add_parameter("EVENT", "CONTEXTS", events_contexts)
        c3d.add_parameter("EVENT", "LABELS", events_labels)
        c3d.add_parameter("EVENT", "TIMES", events_times)

        # Copy the header
        for element in self.c3d["header"]:
            for item in self.c3d["header"][element]:
                c3d["header"][element][item] = self.c3d["header"][element][item]

        # Dispatch the analog and force_plate data
        c3d["parameters"]["ANALOG"] = self.c3d["parameters"]["ANALOG"]
        for i, label in enumerate(self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]):
            label = label.replace("Force.", "")
            label = label.replace("Moment.", "")
            c3d["parameters"]["ANALOG"]["LABELS"]["value"][i] = label

        c3d["parameters"]["FORCE_PLATFORM"] = self.c3d["parameters"]["FORCE_PLATFORM"]
        c3d["data"]["analogs"] = self.c3d["data"]["analogs"]

        # Write the data
        c3d.write("tata.c3d")
