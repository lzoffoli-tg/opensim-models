from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2] / "assets" / "rajagopalaiulrich2023.osim"
)


def import_opensim() -> Any:
    """Import and return the OpenSim Python bindings.

    Returns
    -------
    module
        Imported ``opensim`` module.

    Raises
    ------
    RuntimeError
        If the native OpenSim Python bindings are not installed or cannot be
        loaded by the active Python environment.
    """
    try:
        import opensim
    except ImportError as error:
        raise RuntimeError(
            "OpenSim Python bindings are required to create an OpensimUser. "
            "Install a compatible OpenSim/Conda environment first."
        ) from error
    return opensim


class OpenSimModel:
    """Small, version-tolerant facade over an OpenSim model instance.

    Parameters
    ----------
    model_path : str or pathlib.Path, optional
        Path to the OpenSim ``.osim`` model file.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    RuntimeError
        If the OpenSim bindings cannot be imported.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        """Load the model, unlock its coordinates, and initialize its state."""
        self.opensim = import_opensim()
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model = self.opensim.Model(str(self.model_path))
        self._unlock_coordinates()
        self.state = self.model.initSystem()

    def _unlock_coordinates(self) -> None:
        """Unlock every coordinate so the user can fully repose the model.

        The base model locks a few coordinates (e.g. subtalar, MTP, wrist)
        for gait-simulation purposes. Those locks are not backed by
        constraints, so removing them is safe and lets ``OpensimUser``
        expose posture setters for the full coordinate set.
        """
        coordinates = self.model.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinates.get(index).set_locked(False)

    @property
    def bodies(self) -> Any:
        """Return the model's OpenSim body set."""
        return self.model.getBodySet()

    @property
    def joints(self) -> Any:
        """Return the model's OpenSim joint set."""
        return self.model.getJointSet()

    @property
    def muscles(self) -> Any:
        """Return the model's OpenSim muscle set."""
        return self.model.getMuscles()

    @property
    def markers(self) -> Any:
        """Return the model's OpenSim marker set."""
        return self.model.getMarkerSet()

    @property
    def coordinates(self) -> Any:
        """Return the model's OpenSim coordinate set."""
        return self.model.getCoordinateSet()

    def body(self, name: str) -> Any:
        """Return a body by its OpenSim name.

        Parameters
        ----------
        name : str
            OpenSim body name.

        Returns
        -------
        opensim.Body
            Matching body object.
        """
        return self.bodies.get(name)

    def joint(self, name: str) -> Any:
        """Return a joint by its OpenSim name.

        Parameters
        ----------
        name : str
            OpenSim joint name.

        Returns
        -------
        opensim.Joint
            Matching joint object.
        """
        return self.joints.get(name)

    def muscle(self, name: str) -> Any:
        """Return a muscle by its OpenSim name.

        Parameters
        ----------
        name : str
            OpenSim muscle name.

        Returns
        -------
        opensim.Muscle
            Matching muscle object.
        """
        return self.muscles.get(name)

    def marker(self, name: str) -> Any:
        """Return a marker by its OpenSim name.

        Parameters
        ----------
        name : str
            OpenSim marker name.

        Returns
        -------
        opensim.Marker
            Matching marker object.
        """
        return self.markers.get(name)

    def coordinate(self, name: str) -> Any:
        """Return a coordinate by its OpenSim name.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.

        Returns
        -------
        opensim.Coordinate
            Matching coordinate object.
        """
        return self.coordinates.get(name)

    def set_coordinate_degrees(self, name: str, degrees: float) -> None:
        """Set a coordinate value in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        degrees : float
            New coordinate value in degrees.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite.
        """
        if not np.isfinite(degrees):
            raise ValueError("degrees must be finite")
        coordinate = self.coordinate(name)
        if coordinate.get_locked():
            raise ValueError(f"OpenSim coordinate {name!r} is locked")
        coordinate.setValue(self.state, float(np.deg2rad(degrees)), False)
        self.model.realizePosition(self.state)

    def coordinate_degrees(self, name: str) -> float:
        """Read a coordinate value converted from radians to degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.

        Returns
        -------
        float
            Current coordinate value in degrees.
        """
        coordinate = self.coordinate(name)
        return float(np.rad2deg(coordinate.getValue(self.state)))

    def set_coordinate_locked(self, name: str, locked: bool) -> None:
        """Lock or unlock a coordinate.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        locked : bool
            Whether the coordinate should reject new values.
        """
        self.coordinate(name).set_locked(bool(locked))

    def coordinate_locked(self, name: str) -> bool:
        """Return whether a coordinate is currently locked.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        return bool(self.coordinate(name).get_locked())

    def set_coordinate_range(self, name: str, min_degrees: float, max_degrees: float) -> None:
        """Set the allowed range of motion of a coordinate, in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        min_degrees : float
            Lower bound of the range, in degrees.
        max_degrees : float
            Upper bound of the range, in degrees.

        Raises
        ------
        ValueError
            If a bound is not finite or ``min_degrees >= max_degrees``.
        """
        if not (np.isfinite(min_degrees) and np.isfinite(max_degrees)):
            raise ValueError("range bounds must be finite")
        if min_degrees >= max_degrees:
            raise ValueError("min_degrees must be less than max_degrees")
        coordinate = self.coordinate(name)
        coordinate.setRangeMin(float(np.deg2rad(min_degrees)))
        coordinate.setRangeMax(float(np.deg2rad(max_degrees)))

    def coordinate_range(self, name: str) -> tuple[float, float]:
        """Return the allowed range of motion of a coordinate, in degrees.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        coordinate = self.coordinate(name)
        return (
            float(np.rad2deg(coordinate.getRangeMin())),
            float(np.rad2deg(coordinate.getRangeMax())),
        )

    def set_marker_location(self, name: str, x: float, y: float, z: float) -> None:
        """Set a marker's offset within its parent frame, in metres.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        x, y, z : float
            Offset coordinates in metres.

        Raises
        ------
        ValueError
            If a coordinate is not finite.
        """
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("location must be finite")
        self.marker(name).set_location(self.opensim.Vec3(float(x), float(y), float(z)))

    def marker_location(self, name: str) -> tuple[float, float, float]:
        """Return a marker's offset within its parent frame, in metres.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        """
        location = self.marker(name).get_location()
        return (location.get(0), location.get(1), location.get(2))

    def set_muscle_max_isometric_force(self, name: str, newtons: float) -> None:
        """Set a muscle's maximum isometric force, in newtons.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        newtons : float
            New maximum isometric force. Must be strictly positive.
        """
        self.muscle(name).setMaxIsometricForce(self._positive(newtons))

    def muscle_max_isometric_force(self, name: str) -> float:
        """Return a muscle's maximum isometric force, in newtons."""
        return float(self.muscle(name).getMaxIsometricForce())

    def set_muscle_optimal_fiber_length(self, name: str, meters: float) -> None:
        """Set a muscle's optimal fiber length, in metres.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        meters : float
            New optimal fiber length. Must be strictly positive.
        """
        self.muscle(name).setOptimalFiberLength(self._positive(meters))

    def muscle_optimal_fiber_length(self, name: str) -> float:
        """Return a muscle's optimal fiber length, in metres."""
        return float(self.muscle(name).getOptimalFiberLength())

    def set_muscle_tendon_slack_length(self, name: str, meters: float) -> None:
        """Set a muscle's tendon slack length, in metres.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        meters : float
            New tendon slack length. Must be strictly positive.
        """
        self.muscle(name).setTendonSlackLength(self._positive(meters))

    def muscle_tendon_slack_length(self, name: str) -> float:
        """Return a muscle's tendon slack length, in metres."""
        return float(self.muscle(name).getTendonSlackLength())

    def set_muscle_pennation_angle(self, name: str, degrees: float) -> None:
        """Set a muscle's pennation angle at optimal fiber length, in degrees.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        degrees : float
            New pennation angle. Must be finite and within ``[0, 90)``.
        """
        if not np.isfinite(degrees) or not (0.0 <= degrees < 90.0):
            raise ValueError("pennation angle must be a finite value in [0, 90) degrees")
        self.muscle(name).setPennationAngleAtOptimalFiberLength(float(np.deg2rad(degrees)))

    def muscle_pennation_angle(self, name: str) -> float:
        """Return a muscle's pennation angle at optimal fiber length, in degrees."""
        return float(np.rad2deg(self.muscle(name).getPennationAngleAtOptimalFiberLength()))

    @staticmethod
    def _positive(value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"value must be strictly positive, got {value!r}")
        return float(value)

    def scale_bodies(self, factors: dict[str, tuple[float, float, float]]) -> None:
        """Scale the complete model through OpenSim's native ScaleSet pipeline.

        Parameters
        ----------
        factors : dict[str, tuple[float, float, float]]
            Body names mapped to positive ``(x, y, z)`` scale factors.

        Raises
        ------
        ValueError
            If a scale factor is non-finite or not strictly positive.
        """
        scale_set = self.opensim.ScaleSet()
        for body_name, axes in factors.items():
            if not self.bodies.contains(body_name):
                continue
            if any(not np.isfinite(value) or value <= 0 for value in axes):
                raise ValueError(f"Invalid scale factor for body {body_name!r}: {axes}")
            scale = self.opensim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(self.opensim.Vec3(*axes))
            scale.setApply(True)
            scale_set.adoptAndAppend(scale)
        self.model.scale(self.state, scale_set, True)
        self.state = self.model.initSystem()

    def export(self, model_path: str | Path) -> Path:
        """Save the current model, including scaling and posture, as ``.osim``.

        OpenSim keeps coordinate values in the ``State`` rather than in the
        model itself, so each coordinate's default value is synced from the
        current state before serializing; otherwise the exported file would
        reopen in the model's original, unposed configuration.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Destination path for the exported ``.osim`` file.

        Returns
        -------
        pathlib.Path
            Path to the written file.
        """
        coordinates = self.coordinates
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            coordinate.setDefaultValue(coordinate.getValue(self.state))
        destination = Path(model_path)
        self.model.printToXML(str(destination))
        return destination
