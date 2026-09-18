from __future__ import annotations

from pathlib import Path

from ...model import OpenSimModel, _register_geometry_search_path, import_opensim
from ._data import DEFAULT_DATASET, resolve_reference
from ._mapping import segment_scale_factors

__all__ = ["User"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL_PATH = _ASSETS_DIR / "rajagopalaiulrich2023.osim"
DEFAULT_MESHES_DIR = _ASSETS_DIR / "meshes"


class User(OpenSimModel):
    """An ANSUR-based anthropometric user backed by an OpenSim model.

    Parameters
    ----------
    gender : str
        Sex code, either ``"M"`` or ``"F"``.
    height : float or None, optional
        Requested stature in centimetres. If provided, its ANSUR percentile
        takes precedence over ``percentile``.
    percentile : float, optional
        Common ANSUR percentile when ``height`` is not provided. Defaults to
        ``50.0``.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR reference CSV.
    model_path : str or pathlib.Path, optional
        Path to the OpenSim model file.
    """

    def __init__(
        self,
        gender: str,
        height: float | None = None,
        percentile: float = 50.0,
        *,
        dataset: str | Path = DEFAULT_DATASET,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        """Resolve anthropometry, load the base model, and scale it."""
        self._reference = resolve_reference(gender, height, percentile, dataset)
        # Register the mesh directory before the model file is loaded: the
        # bodies' attached Mesh geometry resolves its file immediately while
        # the model is being built, not lazily when show() runs. self isn't
        # a full OpenSimModel yet (super().__init__ hasn't run), so this
        # can't go through the self.add_geometry_directory instance method.
        _register_geometry_search_path(import_opensim(), DEFAULT_MESHES_DIR)
        super().__init__(model_path)
        self.add_geometry_directory(DEFAULT_MESHES_DIR)
        baseline = resolve_reference(self.gender, percentile=50.0, dataset=dataset)
        factors = segment_scale_factors(self._reference, baseline)
        self.scale_bodies(self._expand_bilateral_bodies(factors))

    @property
    def gender(self):
        """Return the normalized sex code used for the reference."""
        return self._reference.gender

    @property
    def height(self):
        """Return resolved stature in centimetres."""
        return self._reference.height_cm

    @property
    def percentile(self):
        """Return the effective ANSUR percentile."""
        return self._reference.percentile

    @property
    def anthropometry(self):
        """Return all resolved anthropometric measurements."""
        return self._reference

    def _expand_bilateral_bodies(self, factors: dict[str, tuple[float, float, float]]):
        expanded: dict[str, tuple[float, float, float]] = {}
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if self.bodies.contains(candidate):
                    expanded[candidate] = values
        return expanded

    def set_left_hip_flexionextension(self, degrees: float):
        """Set left hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_l", degrees)

    def set_right_hip_flexionextension(self, degrees: float):
        """Set right hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_r", degrees)

    def set_left_hip_adduction(self, degrees: float):
        """Set left hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_l", degrees)

    def set_right_hip_adduction(self, degrees: float):
        """Set right hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_r", degrees)

    def set_left_hip_rotation(self, degrees: float):
        """Set left hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_l", degrees)

    def set_right_hip_rotation(self, degrees: float):
        """Set right hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_r", degrees)

    def set_left_knee_flexionextension(self, degrees: float):
        """Set left knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_l", degrees)

    def set_right_knee_flexionextension(self, degrees: float):
        """Set right knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_r", degrees)

    def set_left_ankle_flexiondorsiflexion(self, degrees: float):
        """Set left ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_l", degrees)

    def set_right_ankle_flexiondorsiflexion(self, degrees: float):
        """Set right ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_r", degrees)

    def set_left_subtalar_inversion(self, degrees: float):
        """Set left subtalar inversion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks subtalar coordinates.
        """
        self.set_coordinate_degrees("subtalar_angle_l", degrees)

    def set_right_subtalar_inversion(self, degrees: float):
        """Set right subtalar inversion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks subtalar coordinates.
        """
        self.set_coordinate_degrees("subtalar_angle_r", degrees)

    def set_left_mtp_flexion(self, degrees: float):
        """Set left metatarsophalangeal flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks metatarsophalangeal coordinates.
        """
        self.set_coordinate_degrees("mtp_angle_l", degrees)

    def set_right_mtp_flexion(self, degrees: float):
        """Set right metatarsophalangeal flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks metatarsophalangeal coordinates.
        """
        self.set_coordinate_degrees("mtp_angle_r", degrees)

    def set_left_shoulder_flexion(self, degrees: float):
        """Set left shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_l", degrees)

    def set_right_shoulder_flexion(self, degrees: float):
        """Set right shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_r", degrees)

    def set_left_shoulder_adduction(self, degrees: float):
        """Set left shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_l", degrees)

    def set_right_shoulder_adduction(self, degrees: float):
        """Set right shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_r", degrees)

    def set_left_shoulder_rotation(self, degrees: float):
        """Set left shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_l", degrees)

    def set_right_shoulder_rotation(self, degrees: float):
        """Set right shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_r", degrees)

    def set_left_elbow_flexion(self, degrees: float):
        """Set left elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_l", degrees)

    def set_right_elbow_flexion(self, degrees: float):
        """Set right elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_r", degrees)

    def set_left_wrist_flexion(self, degrees: float):
        """Set left wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_l", degrees)

    def set_right_wrist_flexion(self, degrees: float):
        """Set right wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_r", degrees)

    def set_left_wrist_deviation(self, degrees: float):
        """Set left wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_l", degrees)

    def set_right_wrist_deviation(self, degrees: float):
        """Set right wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_r", degrees)

    def set_left_forearm_pronation(self, degrees: float):
        """Set left forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_l", degrees)

    def set_right_forearm_pronation(self, degrees: float):
        """Set right forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_r", degrees)

    def set_lumbar_extension(self, degrees: float):
        """Set lumbar extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_extension", degrees)

    def set_lumbar_bending(self, degrees: float):
        """Set lumbar lateral bending coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_bending", degrees)

    def set_lumbar_rotation(self, degrees: float):
        """Set lumbar rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_rotation", degrees)
