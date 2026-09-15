from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import DEFAULT_DATASET, AnthropometricReference, resolve_reference
from .mapping import segment_scale_factors
from .model import DEFAULT_MODEL, OpenSimModel


class OpensimUser:
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
        model_path: str | Path = DEFAULT_MODEL,
    ) -> None:
        """Resolve anthropometry, load a private model, and scale it."""
        self._reference = resolve_reference(gender, height, percentile, dataset)
        self._model = OpenSimModel(model_path)
        baseline = resolve_reference(self.gender, percentile=50.0, dataset=dataset)
        factors = segment_scale_factors(self._reference, baseline)
        self._model.scale_bodies(self._expand_bilateral_bodies(factors))

    @property
    def gender(self) -> str:
        """Return the normalized sex code used for the reference."""
        return self._reference.gender

    @property
    def height(self) -> float:
        """Return resolved stature in centimetres."""
        return self._reference.height_cm

    @property
    def percentile(self) -> float:
        """Return the effective ANSUR percentile."""
        return self._reference.percentile

    @property
    def anthropometry(self) -> AnthropometricReference:
        """Return all resolved anthropometric measurements."""
        return self._reference

    @property
    def model(self) -> Any:
        """Return the scaled OpenSim model instance."""
        return self._model.model

    @property
    def state(self) -> Any:
        """Return the current OpenSim state used for posture."""
        return self._model.state

    @property
    def bodies(self) -> Any:
        """Return the model body set."""
        return self._model.bodies

    @property
    def joints(self) -> Any:
        """Return the model joint set."""
        return self._model.joints

    @property
    def muscles(self) -> Any:
        """Return the model muscle set."""
        return self._model.muscles

    @property
    def markers(self) -> Any:
        """Return the model marker set."""
        return self._model.markers

    @property
    def coordinates(self) -> Any:
        """Return the model coordinate set."""
        return self._model.coordinates

    def body(self, name: str) -> Any:
        """Return a body by name.

        Parameters
        ----------
        name : str
            OpenSim body name.
        """
        return self._model.body(name)

    def joint(self, name: str) -> Any:
        """Return a joint by name.

        Parameters
        ----------
        name : str
            OpenSim joint name.
        """
        return self._model.joint(name)

    def muscle(self, name: str) -> Any:
        """Return a muscle by name.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        """
        return self._model.muscle(name)

    def marker(self, name: str) -> Any:
        """Return a marker by name.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        """
        return self._model.marker(name)

    def coordinate(self, name: str) -> Any:
        """Return a coordinate by name.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        return self._model.coordinate(name)

    def set_coordinate_locked(self, name: str, locked: bool) -> None:
        """Lock or unlock any coordinate.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        locked : bool
            Whether the coordinate should reject new values.
        """
        self._model.set_coordinate_locked(name, locked)

    def coordinate_locked(self, name: str) -> bool:
        """Return whether a coordinate is currently locked.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        return self._model.coordinate_locked(name)

    def set_coordinate_range(
        self,
        name: str,
        min_degrees: float,
        max_degrees: float,
    ) -> None:
        """Set the allowed range of motion of any coordinate, in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        min_degrees : float
            Lower bound of the range, in degrees.
        max_degrees : float
            Upper bound of the range, in degrees.
        """
        self._model.set_coordinate_range(name, min_degrees, max_degrees)

    def coordinate_range(self, name: str) -> tuple[float, float]:
        """Return the allowed range of motion of any coordinate, in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        return self._model.coordinate_range(name)

    def set_marker_location(self, name: str, x: float, y: float, z: float) -> None:
        """Set a marker's offset within its parent frame, in metres.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        x, y, z : float
            Offset coordinates in metres.
        """
        self._model.set_marker_location(name, x, y, z)

    def marker_location(self, name: str) -> tuple[float, float, float]:
        """Return a marker's offset within its parent frame, in metres.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        """
        return self._model.marker_location(name)

    def set_muscle_max_isometric_force(self, name: str, newtons: float) -> None:
        """Set a muscle's maximum isometric force, in newtons.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        newtons : float
            New maximum isometric force.
        """
        self._model.set_muscle_max_isometric_force(name, newtons)

    def muscle_max_isometric_force(self, name: str) -> float:
        """Return a muscle's maximum isometric force, in newtons."""
        return self._model.muscle_max_isometric_force(name)

    def set_muscle_optimal_fiber_length(self, name: str, meters: float) -> None:
        """Set a muscle's optimal fiber length, in metres.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        meters : float
            New optimal fiber length.
        """
        self._model.set_muscle_optimal_fiber_length(name, meters)

    def muscle_optimal_fiber_length(self, name: str) -> float:
        """Return a muscle's optimal fiber length, in metres."""
        return self._model.muscle_optimal_fiber_length(name)

    def set_muscle_tendon_slack_length(self, name: str, meters: float) -> None:
        """Set a muscle's tendon slack length, in metres.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        meters : float
            New tendon slack length.
        """
        self._model.set_muscle_tendon_slack_length(name, meters)

    def muscle_tendon_slack_length(self, name: str) -> float:
        """Return a muscle's tendon slack length, in metres."""
        return self._model.muscle_tendon_slack_length(name)

    def set_muscle_pennation_angle(self, name: str, degrees: float) -> None:
        """Set a muscle's pennation angle at optimal fiber length, in degrees.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        degrees : float
            New pennation angle.
        """
        self._model.set_muscle_pennation_angle(name, degrees)

    def muscle_pennation_angle(self, name: str) -> float:
        """Return a muscle's pennation angle at optimal fiber length, in degrees."""
        return self._model.muscle_pennation_angle(name)

    def export(self, file: str | Path) -> Path:
        """Export the current model to an OpenSim ``.osim`` file.

        The exported file includes the anthropometric scaling applied at
        construction time and the posture set through the coordinate
        setters, and can be opened directly in OpenSim.

        Parameters
        ----------
        file : str or pathlib.Path
            Destination path for the exported model file.

        Returns
        -------
        pathlib.Path
            Path to the written file.
        """
        return self._model.export(file)

    def _expand_bilateral_bodies(
        self, factors: dict[str, tuple[float, float, float]]
    ) -> dict[str, tuple[float, float, float]]:
        expanded: dict[str, tuple[float, float, float]] = {}
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if self.bodies.contains(candidate):
                    expanded[candidate] = values
        return expanded

    def set_coordinate_degrees(self, name: str, degrees: float) -> None:
        """Set any unlocked coordinate using degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        degrees : float
            Coordinate value in degrees.

        Raises
        ------
        ValueError
            If the coordinate is locked or ``degrees`` is not finite.
        """
        self._model.set_coordinate_degrees(name, degrees)

    def coordinate_degrees(self, name: str) -> float:
        """Return any coordinate value in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.

        Returns
        -------
        float
            Current coordinate value in degrees.
        """
        return self._model.coordinate_degrees(name)

    def set_left_hip_flexionextension(self, degrees: float) -> None:
        """Set left hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_l", degrees)

    def set_right_hip_flexionextension(self, degrees: float) -> None:
        """Set right hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_r", degrees)

    def set_left_hip_adduction(self, degrees: float) -> None:
        """Set left hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_l", degrees)

    def set_right_hip_adduction(self, degrees: float) -> None:
        """Set right hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_r", degrees)

    def set_left_hip_rotation(self, degrees: float) -> None:
        """Set left hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_l", degrees)

    def set_right_hip_rotation(self, degrees: float) -> None:
        """Set right hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_r", degrees)

    def set_left_knee_flexionextension(self, degrees: float) -> None:
        """Set left knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_l", degrees)

    def set_right_knee_flexionextension(self, degrees: float) -> None:
        """Set right knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_r", degrees)

    def set_left_ankle_flexiondorsiflexion(self, degrees: float) -> None:
        """Set left ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_l", degrees)

    def set_right_ankle_flexiondorsiflexion(self, degrees: float) -> None:
        """Set right ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_r", degrees)

    def set_left_subtalar_inversion(self, degrees: float) -> None:
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

    def set_right_subtalar_inversion(self, degrees: float) -> None:
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

    def set_left_mtp_flexion(self, degrees: float) -> None:
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

    def set_right_mtp_flexion(self, degrees: float) -> None:
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

    def set_left_shoulder_flexion(self, degrees: float) -> None:
        """Set left shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_l", degrees)

    def set_right_shoulder_flexion(self, degrees: float) -> None:
        """Set right shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_r", degrees)

    def set_left_shoulder_adduction(self, degrees: float) -> None:
        """Set left shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_l", degrees)

    def set_right_shoulder_adduction(self, degrees: float) -> None:
        """Set right shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_r", degrees)

    def set_left_shoulder_rotation(self, degrees: float) -> None:
        """Set left shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_l", degrees)

    def set_right_shoulder_rotation(self, degrees: float) -> None:
        """Set right shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_r", degrees)

    def set_left_elbow_flexion(self, degrees: float) -> None:
        """Set left elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_l", degrees)

    def set_right_elbow_flexion(self, degrees: float) -> None:
        """Set right elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_r", degrees)

    def set_left_wrist_flexion(self, degrees: float) -> None:
        """Set left wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_l", degrees)

    def set_right_wrist_flexion(self, degrees: float) -> None:
        """Set right wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_r", degrees)

    def set_left_wrist_deviation(self, degrees: float) -> None:
        """Set left wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_l", degrees)

    def set_right_wrist_deviation(self, degrees: float) -> None:
        """Set right wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_r", degrees)

    def set_left_forearm_pronation(self, degrees: float) -> None:
        """Set left forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_l", degrees)

    def set_right_forearm_pronation(self, degrees: float) -> None:
        """Set right forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_r", degrees)

    def set_lumbar_extension(self, degrees: float) -> None:
        """Set lumbar extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_extension", degrees)

    def set_lumbar_bending(self, degrees: float) -> None:
        """Set lumbar lateral bending coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_bending", degrees)

    def set_lumbar_rotation(self, degrees: float) -> None:
        """Set lumbar rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_rotation", degrees)
