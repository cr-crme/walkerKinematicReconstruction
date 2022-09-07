import biorbd
import numpy as np


class Param:
    """
    This is a convenient class to hold inertia parameter information of the dynamic model below
    """
    def __init__(
        self,
        marker_names: tuple[str, ...],
        mass: float,
        center_of_mass: float,
        radii: tuple[float, float, float],
    ):
        """
        Parameters
        ----------
        marker_names
            The name of the markers that creates the vector to compute the length of the segment from medial to lateral
        mass
            The relative mass of the total body mass
        center_of_mass
            The position of the center of mass as a percentage of the distance from medial to distal
        radii
            The xx, yy, zz, (Sagittal, Transverse, Longitudinal) radii of giration
        """
        self.marker_names = marker_names
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.radii = radii


class PlugInGaitDynamicModel:
    """
    This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
    https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus) of the GenericDynamicModel protocol of
    biorbd.model_creation
    The kinematic model should be named as such (SEGMENT_NAME: from 'MEDIAL_MARKER' to 'LATERAL_MARKER'):
        *'Pelvis': 'LPSI', 'RPSI', 'LASI', 'RASI'
        *'Thorax': 'C7'
        *'Head': 'LBHD', 'RBHD', 'LFHD', 'RFHD'
        'Humerus': 'LBHD
        'LOWER_ARM': from 'ELBOW' to 'WRIST'
        'HAND': from 'WRIST' to 'FINGER'
        'THIGH': from 'PELVIS' to 'KNEE'
        'SHANK': from 'KNEE' to 'ANKLE'
        'FOOT': from 'ANKLE' to 'TOE'
    *The marker names for this segment do not follow medial to lateral logic since the length is computed
    """

    def __init__(self, mass: float, model: biorbd.Model):
        """
        Parameters
        ----------
        mass
            The mass of the subject
        model
            The biorbd model. This is to compute lengths
        """

        self.mass = mass
        self.model = model

        # Produce some easy to access variables
        self.q_zero = np.zeros((model.nbQ()))
        self.marker_names = [name.to_string() for name in model.markerNames()]

        # This is the actual copy of the Plugin Gait table
        self.table = {
            "Pelvis": Param(
                marker_names=("TOP_HEAD", "SHOULDER"),
                mass=0.0694,
                center_of_mass=0.5002,
                radii=(0.303, 0.315, 0.261),
            ),
            "TRUNK": Param(
                marker_names=("SHOULDER", "PELVIS"), mass=0.4346, center_of_mass=0.5138, radii=(0.328, 0.306, 0.169)
            ),
            "UPPER_ARM": Param(
                marker_names=("SHOULDER", "ELBOW"),
                mass=0.0271 * 2,
                center_of_mass=0.5772,
                radii=(0.285, 0.269, 0.158),
            ),
            "LOWER_ARM": Param(
                marker_names=("ELBOW", "WRIST"), mass=0.0162 * 2, center_of_mass=0.4574, radii=(0.276, 0.265, 0.121)
            ),
            "HAND": Param(
                marker_names=("WRIST", "FINGER"),
                mass=0.0061 * 2,
                center_of_mass=0.7900,
                radii=(0.628, 0.513, 0.401),
            ),
            "THIGH": Param(
                marker_names=("PELVIS", "KNEE"), mass=0.1416 * 2, center_of_mass=0.4095, radii=(0.329, 0.329, 0.149)
            ),
            "SHANK": Param(
                marker_names=("KNEE", "ANKLE"), mass=0.0433 * 2, center_of_mass=0.4459, radii=(0.255, 0.249, 0.103)
            ),
            "FOOT": Param(
                marker_names=("ANKLE", "TOE"), mass=0.0137 * 2, center_of_mass=0.4415, radii=(0.257, 0.245, 0.124)
            ),
        }

    def segment_mass(self, segment: str):
        return self.table[segment].mass * self.mass

    def segment_length(self, segment: str):
        table = self.table[segment]

        # Find the position of the markers when the model is in resting position
        marker_positions = np.array([marker.to_array() for marker in self.model.markers(self.q_zero)]).transpose()

        # Find the index of the markers required to compute the length of the segment
        idx_proximal = self.marker_names.index(table.marker_names[0])
        idx_distal = self.marker_names.index(table.marker_names[1])

        # Compute the Euclidian distance between the two positions
        return np.linalg.norm(marker_positions[:, idx_distal] - marker_positions[:, idx_proximal])

    def segment_center_of_mass(self, segment: str, inverse_proximal: bool = False):
        # This method will compute the length of the required segment based on the biorbd model and required markers
        # If inverse_proximal is set to True, then the value is returned from the distal position
        table = self.table[segment]

        # Find the position of the markers when the model is in resting position
        marker_positions = np.array([marker.to_array() for marker in self.model.markers(self.q_zero)]).transpose()

        # Find the index of the markers required to compute the length of the segment
        idx_proximal = self.marker_names.index(table.marker_names[0])
        idx_distal = self.marker_names.index(table.marker_names[1])

        # Compute the position of the center of mass
        if inverse_proximal:
            center_of_mass = (1 - table.center_of_mass) * (
                marker_positions[:, idx_proximal] - marker_positions[:, idx_distal]
            )
        else:
            center_of_mass = table.center_of_mass * (
                marker_positions[:, idx_distal] - marker_positions[:, idx_proximal]
            )
        return tuple(center_of_mass)  # convert the result to a Tuple which is good practise

    def segment_moment_of_inertia(self, segment: str):
        mass = self.segment_mass(segment)
        length = self.segment_length(segment)
        radii = self.table[segment].radii

        return mass * (length * radii[0]) ** 2, mass * (length * radii[1]) ** 2, mass * (length * radii[2]) ** 2

