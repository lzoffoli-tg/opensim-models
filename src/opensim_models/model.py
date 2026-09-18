from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["import_opensim", "OpenSimModel"]


def _ensure_visualizer_dll_path() -> None:
    """Make the conda ``Library/bin`` DLLs visible to the visualizer process.

    On Windows, ``simbody-visualizer.exe`` depends on native DLLs (Simbody,
    freeglut, etc.) that live in the active environment's ``Library/bin``
    directory. When Python is launched without going through conda's shell
    activation, that directory is missing from ``PATH`` and the visualizer
    process fails to start; OpenSim then reports this as a generic
    ``initSystem`` exception with no indication that a DLL was involved.
    """
    if not sys.platform.startswith("win"):
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(library_bin) not in path_entries:
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")


def _register_geometry_search_path(opensim: Any, directory: str | Path) -> None:
    """Register ``directory`` with OpenSim's own native mesh file search.

    A ``Mesh`` resolves its file immediately when its owning component's
    properties are finalized (e.g. right when ``attachGeometry`` runs, or
    while a ``.osim`` file with existing mesh-bearing bodies is being
    loaded) -- not lazily when the model is later shown. This must
    therefore run *before* that finalization, which is also before an
    :class:`OpenSimModel` may exist to call yet (see
    ``models.user.User.__init__``, which loads a ``.osim`` file with
    existing meshes and so must register this ahead of
    ``super().__init__()``) -- hence a free function taking the ``opensim``
    module directly, rather than a method requiring ``self.opensim``.
    :meth:`OpenSimModel.add_geometry_directory` calls this for the common
    case where an instance already exists.
    """
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(Path(directory).resolve()))


def _iter_set(component_set: Any) -> Any:
    for index in range(component_set.getSize()):
        yield component_set.get(index)


def _unique_component_name(prefix: str, name: str, existing: set[str]) -> str:
    candidate = f"{prefix}_{name}"
    suffix = 2
    while candidate in existing:
        candidate = f"{prefix}_{name}_{suffix}"
        suffix += 1
    return candidate


def _unique_body_name(raw_name: str, existing: set[str]) -> str:
    """Sanitize a CAD part name into a valid, unique OpenSim body name."""
    sanitized = (
        "".join(char if char.isalnum() else "_" for char in raw_name).strip("_")
        or "body"
    )
    if sanitized[0].isdigit():
        sanitized = f"body_{sanitized}"
    candidate = sanitized
    suffix = 2
    while candidate in existing:
        candidate = f"{sanitized}_{suffix}"
        suffix += 1
    return candidate


def _patch_sockets(root: Any, renamed_paths: dict[str, str]) -> None:
    for component in (root, *root.getComponentsList()):
        for socket_name in component.getSocketNames():
            socket = component.updSocket(socket_name)
            path = socket.getConnecteePath()
            if path in renamed_paths:
                socket.setConnecteePath(renamed_paths[path])


def import_opensim() -> Any:
    """Import and return the OpenSim Python bindings.

    Fixes up the native ``PATH`` (see :func:`_ensure_visualizer_dll_path`)
    before importing, so any OpenSim usage -- not just visualization -- runs
    with the required native libraries reachable.

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
    _ensure_visualizer_dll_path()
    try:
        import opensim
    except ImportError as error:
        raise RuntimeError(
            "OpenSim Python bindings are required to create an OpenSimModel. "
            "Install a compatible OpenSim/Conda environment first."
        ) from error
    opensim.Logger.removeFileSink()  # OpenSim otherwise writes opensim.log to the CWD
    return opensim


class OpenSimModel:
    """Generic, composable facade over an OpenSim model instance.

    Parameters
    ----------
    model_path : str, pathlib.Path or None, optional
        Path to an OpenSim ``.osim`` model file. When ``None``, an empty
        in-memory model is created instead (used internally to build the
        result of :meth:`__add__`).

    Raises
    ------
    FileNotFoundError
        If ``model_path`` is given and does not exist.
    RuntimeError
        If the OpenSim bindings cannot be imported.
    """

    _MERGE_SETS = (
        ("bodyset", "updBodySet"),
        ("jointset", "updJointSet"),
        ("forceset", "updForceSet"),
        ("markerset", "updMarkerSet"),
        ("constraintset", "updConstraintSet"),
        ("controllerset", "updControllerSet"),
        ("contactgeometryset", "updContactGeometrySet"),
        ("probeset", "updProbeSet"),
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Load or create the model, unlock its coordinates, and initialize its state."""
        self.opensim = import_opensim()
        if model_path is None:
            self.model_path = None
            self.model = self.opensim.Model()
        else:
            self.model_path = Path(model_path)
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)
            self.model = self.opensim.Model(str(self.model_path))
        self._unlock_coordinates()
        self.state = self.model.initSystem()
        self._geometry_dirs: list[Path] = []
        self._anchor_names: set[str] = set()
        self._merged: dict[int, list[tuple[str, str]]] = {}
        self._visualizer: Any | None = None

    def copy(self) -> "OpenSimModel":
        """Return an independent, deep copy of this model.

        Copies every instance attribute -- so this works unmodified for any
        subclass (e.g. :class:`.User` or :class:`.Screen`) without it
        needing its own ``copy()`` override -- then replaces the parts that
        must be independent (the underlying ``opensim.Model``, its
        ``State``, and the mutable bookkeeping collections) with real
        copies rather than shared references. The current posture and
        velocities are preserved.

        Returns
        -------
        OpenSimModel
            A new, independent instance of ``type(self)``.
        """
        self._sync_coordinate_defaults()

        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.model = self.model.clone()
        clone.state = clone.model.initSystem()
        clone._geometry_dirs = list(self._geometry_dirs)
        clone._anchor_names = set(self._anchor_names)
        clone._merged = dict(self._merged)
        clone._visualizer = None
        return clone

    @classmethod
    def from_step(
        cls,
        step_path: str | Path,
        *,
        mesh_dir: str | Path | None = None,
        density: float = 1000.0,
        densities: dict[str, float] | None = None,
        add_free_joints: bool = True,
        linear_deflection: float = 0.5e-3,
        angular_deflection: float = 0.25,
        as_one_object: bool = True,
    ) -> "OpenSimModel":
        """Build a model from a CAD assembly.

        By default (``as_one_object=True``), every solid found in the STEP
        file is welded into a single rigid ``opensim.Body``: their masses,
        centres of mass and inertia tensors are combined (parallel-axis
        theorem) into one set of mass properties, and each solid's
        triangulated mesh is attached to that one body. With
        ``as_one_object=False``, each solid instead becomes its own
        ``opensim.Body``, as in previous versions of this method.

        In both cases, mass and inertia come from each solid's geometry
        (volume times ``density``), a triangulated mesh of each solid is
        written to disk and attached to its body, and, because a plain CAD
        assembly carries no kinematic information, bodies are (by default)
        connected to ground with 6-dof :class:`opensim.FreeJoint`\\ s placed
        at the assembly's original position, so the model loads and
        simulates immediately with the assembly's original layout; replace
        those joints with the assembly's real kinematic chain before relying
        on the model's dynamics.

        Parameters
        ----------
        step_path : str or pathlib.Path
            Path to a ``.step``/``.stp`` file.
        mesh_dir : str, pathlib.Path or None, optional
            Directory to write the generated mesh files to. Defaults to a
            ``{step_path.stem}_meshes`` folder next to ``step_path``.
        density : float, optional
            Density, in kg/m^3, used for solids not listed in ``densities``.
            STEP files rarely carry material data, so this is a generic
            placeholder (default ``1000.0``); pass real values for physically
            accurate mass and inertia.
        densities : dict[str, float] or None, optional
            Per-part density overrides, keyed by the part name as read from
            the STEP file.
        add_free_joints : bool, optional
            When ``True`` (default), attach each body to ground with a
            ``FreeJoint`` at its original CAD position and initialize the
            model. When ``False``, bodies are added without joints and the
            caller must connect them and call ``model.model.initSystem()``
            before using the model.
        linear_deflection, angular_deflection : float, optional
            Tessellation tolerances (metres, radians) forwarded to the mesh
            generator; smaller values produce finer, larger meshes.
        as_one_object : bool, optional
            When ``True`` (default), combine every solid into a single
            ``opensim.Body`` regardless of how many solids/components the
            STEP file contains. When ``False``, generate one body per solid.

        Returns
        -------
        OpenSimModel
            Model containing either one combined body (``as_one_object=True``)
            or one body per solid found in the STEP file
            (``as_one_object=False``).

        Raises
        ------
        FileNotFoundError
            If ``step_path`` does not exist.
        ValueError
            If the STEP file contains no solids.
        RuntimeError
            If the ``pythonocc-core`` (``OCC``) package is not installed.
        """
        from ._cad_import import (
            _combine_mass_properties,
            _read_step_solids,
            _solid_mass_properties,
            _write_solid_mesh,
        )

        step_path = Path(step_path)
        if not step_path.is_file():
            raise FileNotFoundError(step_path)
        destination_dir = (
            Path(mesh_dir)
            if mesh_dir is not None
            else step_path.parent / f"{step_path.stem}_meshes"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)

        model = cls(model_path=None)
        model.add_geometry_directory(destination_dir)
        used_names: set[str] = set()
        parts = []
        for part_name, solid in _read_step_solids(step_path):
            unique_name = _unique_body_name(part_name, used_names)
            used_names.add(unique_name)
            part_density = (densities or {}).get(part_name, density)
            mass, center_of_mass, inertia = _solid_mass_properties(solid, part_density)
            parts.append((unique_name, solid, mass, center_of_mass, inertia))

        if as_one_object:
            body_name = _unique_body_name(step_path.stem, set())
            mass, center_of_mass, inertia = _combine_mass_properties(
                [(mass, com, inertia) for _, _, mass, com, inertia in parts]
            )
            body = model.opensim.Body(
                body_name,
                mass,
                model.opensim.Vec3(0, 0, 0),
                model.opensim.Inertia(*inertia),
            )
            model.model.addBody(body)
            for part_name, solid, _, _, _ in parts:
                mesh_paths = _write_solid_mesh(
                    solid,
                    center_of_mass,
                    destination_dir / f"{body_name}_{part_name}.stl",
                    linear_deflection,
                    angular_deflection,
                )
                for mesh_path in mesh_paths:
                    body.attachGeometry(model.opensim.Mesh(mesh_path.name))

            if add_free_joints:
                joint = model.opensim.FreeJoint(
                    f"{body_name}_joint",
                    model.model.getGround(),
                    model.opensim.Vec3(*center_of_mass),
                    model.opensim.Vec3(0, 0, 0),
                    body,
                    model.opensim.Vec3(0, 0, 0),
                    model.opensim.Vec3(0, 0, 0),
                )
                model.model.addJoint(joint)
        else:
            for body_name, solid, mass, center_of_mass, inertia in parts:
                mesh_paths = _write_solid_mesh(
                    solid,
                    center_of_mass,
                    destination_dir / f"{body_name}.stl",
                    linear_deflection,
                    angular_deflection,
                )

                body = model.opensim.Body(
                    body_name,
                    mass,
                    model.opensim.Vec3(0, 0, 0),
                    model.opensim.Inertia(*inertia),
                )
                model.model.addBody(body)
                for mesh_path in mesh_paths:
                    body.attachGeometry(model.opensim.Mesh(mesh_path.name))

                if add_free_joints:
                    joint = model.opensim.FreeJoint(
                        f"{body_name}_joint",
                        model.model.getGround(),
                        model.opensim.Vec3(*center_of_mass),
                        model.opensim.Vec3(0, 0, 0),
                        body,
                        model.opensim.Vec3(0, 0, 0),
                        model.opensim.Vec3(0, 0, 0),
                    )
                    model.model.addJoint(joint)

        if add_free_joints:
            model.model.finalizeConnections()
            model.state = model.model.initSystem()
        return model

    def _unlock_coordinates(self) -> None:
        """Unlock every coordinate so callers can fully repose the model.

        The base model locks a few coordinates (e.g. subtalar, MTP, wrist)
        for gait-simulation purposes. Those locks are not backed by
        constraints, so removing them is safe and lets specific models
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

    def set_body_mass(self, name: str, kilograms: float) -> None:
        """Set a body's mass and rebuild the system so the change takes effect.

        A body's mass feeds into the multibody system's mass matrix, which
        OpenSim builds once in ``initSystem()``: changing the property alone
        would have no effect on dynamics until the system is rebuilt, so
        this calls :meth:`reinitialize` afterward, preserving the current
        posture and velocities.

        Parameters
        ----------
        name : str
            OpenSim body name.
        kilograms : float
            New mass, in kilograms. Must be strictly positive.

        Raises
        ------
        ValueError
            If ``kilograms`` is not finite or not strictly positive.
        """
        self.body(name).setMass(self._positive(kilograms))
        self.reinitialize()

    def body_mass(self, name: str) -> float:
        """Return a body's mass, in kilograms.

        Parameters
        ----------
        name : str
            OpenSim body name.
        """
        return float(self.body(name).getMass())

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

        This only writes the raw value into :attr:`state`; it does not
        propagate it to anything derived (body positions, muscle lengths,
        forces, ...). Call :meth:`update_state` before reading any derived
        quantity, or after a batch of several ``set_*``/direct ``state``
        edits, to bring the whole state up to date in one pass.

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

    def set_coordinate_speed_degrees(self, name: str, degrees_per_second: float) -> None:
        """Set a coordinate's speed (angular velocity) in degrees per second.

        This only writes the raw value into :attr:`state`; it does not
        propagate it to anything derived. Call :meth:`update_state` before
        reading any derived quantity, or after a batch of several
        ``set_*``/direct ``state`` edits, to bring the whole state up to
        date in one pass.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        degrees_per_second : float
            New coordinate speed in degrees per second.

        Raises
        ------
        ValueError
            If ``degrees_per_second`` is not finite.
        """
        if not np.isfinite(degrees_per_second):
            raise ValueError("degrees_per_second must be finite")
        coordinate = self.coordinate(name)
        if coordinate.get_locked():
            raise ValueError(f"OpenSim coordinate {name!r} is locked")
        coordinate.setSpeedValue(self.state, float(np.deg2rad(degrees_per_second)))

    def coordinate_speed_degrees(self, name: str) -> float:
        """Read a coordinate's speed converted from radians/s to degrees/s.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.

        Returns
        -------
        float
            Current coordinate speed in degrees per second.
        """
        coordinate = self.coordinate(name)
        return float(np.rad2deg(coordinate.getSpeedValue(self.state)))

    def update_state(self) -> None:
        """Propagate every raw value set on :attr:`state` to derived quantities.

        Coordinate/speed setters (and any direct edit of ``model.state``,
        e.g. through OpenSim's own API) only write the raw value; nothing
        derived -- body positions, muscle-tendon lengths and forces,
        muscle equilibrium -- is recomputed until this is realized.
        Calling this once after a batch of edits is both correct and more
        efficient than realizing after every single edit.

        This realizes through ``Stage::Dynamics`` (covering position,
        velocity and every force, including automatically re-solving
        muscle equilibrium via ``equilibrateMuscles`` -- see the muscle
        section of the README for why that step is needed after a posture
        change). It deliberately stops short of ``Stage::Acceleration``:
        that stage is not reliably safe to realize on every OpenSim model
        (it can crash the process outright on some muscle-driven models,
        a native failure Python cannot catch or prevent). If you need
        accelerations, call ``model.model.realizeAcceleration(model.state)``
        yourself, aware of that risk.

        Does *not* re-solve a :class:`opensim.CoordinateCouplerConstraint`
        (e.g. a patella coupled to knee flexion): realizing stages does not
        make Simbody re-derive a dependent coordinate's value from its
        independent one -- that requires ``Model.assemble()``, which is
        also not reliably safe to call on every model (same class of native
        crash risk). If a coupled coordinate must reflect a new posture for
        actual use (not just export), use :meth:`reinitialize` instead,
        which re-derives it through the coordinate's *default* value.
        """
        self.model.realizePosition(self.state)
        self.model.realizeVelocity(self.state)
        if self.muscles.getSize() > 0:
            self.model.equilibrateMuscles(self.state)
        self.model.realizeDynamics(self.state)

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

    def set_coordinate_range(
        self, name: str, min_degrees: float, max_degrees: float
    ) -> None:
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
            raise ValueError(
                "pennation angle must be a finite value in [0, 90) degrees"
            )
        self.muscle(name).setPennationAngleAtOptimalFiberLength(
            float(np.deg2rad(degrees))
        )

    def muscle_pennation_angle(self, name: str) -> float:
        """Return a muscle's pennation angle at optimal fiber length, in degrees."""
        return float(
            np.rad2deg(self.muscle(name).getPennationAngleAtOptimalFiberLength())
        )

    @staticmethod
    def _positive(value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"value must be strictly positive, got {value!r}")
        return float(value)

    def _sync_coordinate_defaults(self) -> None:
        """Copy every coordinate's current state value/speed into its default.

        OpenSim keeps posture and velocity in the ``State``, not in the
        model itself: any operation that rebuilds the system --
        :meth:`export`, :meth:`scale_bodies`, :meth:`add_model`,
        :meth:`reinitialize`, or a raw ``model.initSystem()`` call -- resets
        every coordinate to the value/speed baked into the model file
        otherwise. A coordinate driven by a
        :class:`opensim.CoordinateCouplerConstraint` (e.g. a patella coupled
        to knee flexion) is re-derived from its independent coordinates' new
        defaults through the constraint's function, since OpenSim does not
        do this automatically when a default value is set directly.
        """
        coordinates = self.coordinates
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            coordinate.setDefaultValue(coordinate.getValue(self.state))
            coordinate.setDefaultSpeedValue(coordinate.getSpeedValue(self.state))

        constraints = self.model.getConstraintSet()
        for index in range(constraints.getSize()):
            coupler = self.opensim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(index)
            )
            if coupler is None:
                continue
            independent_names = coupler.getIndependentCoordinateNames()
            independent_values = self.opensim.Vector(independent_names.getSize(), 0.0)
            for i in range(independent_names.getSize()):
                independent = coordinates.get(independent_names.get(i))
                independent_values.set(i, independent.getDefaultValue())
            dependent = coordinates.get(coupler.getDependentCoordinateName())
            dependent.setDefaultValue(coupler.getFunction().calcValue(independent_values))

    def reinitialize(self) -> None:
        """Rebuild the system state, preserving the current posture and velocity.

        Some changes -- e.g. a muscle's ``ignore_tendon_compliance`` -- do
        not touch the model's component tree, only a property, so
        :attr:`state` remains safely readable right up until this call;
        this reads the current posture/velocity, rebuilds the system, and
        restores them as the new defaults.

        This is *not* safe to call after adding or removing a component
        (see :mod:`opensim_models.operators`): doing so, even to only
        read/sync ``state``, crashes the process rather than raising a
        catchable error, because removing/adding a body/joint/force/...
        immediately invalidates ``state`` itself, not just its defaults.
        Use :meth:`structural_change` to wrap that kind of change instead.

        Calling ``model.model.initSystem()`` directly instead of this
        resets every coordinate to the model's original default.
        """
        self._sync_coordinate_defaults()
        self.model.finalizeConnections()
        self.state = self.model.initSystem()

    @contextlib.contextmanager
    def structural_change(self):
        """Context manager wrapping a batch of component additions/removals.

        Adding or removing a body, joint, force/muscle, marker, constraint,
        controller, contact geometry or probe (see
        :mod:`opensim_models.operators`) invalidates :attr:`state`
        immediately -- reading it afterward, even just to sync posture into
        defaults, crashes the process rather than raising a catchable
        error. This syncs the current posture/velocity into each
        coordinate's default *before* the block runs (while ``state`` is
        still valid), then rebuilds the system once the block exits,
        restoring that posture. Wrap any number of
        ``opensim_models.operators`` calls in one block; don't read
        :attr:`state` (directly, or through any coordinate/marker/muscle
        accessor) inside it.

        Example
        -------
        >>> with model.structural_change():
        ...     operators.remove_joint(model, "old_joint")
        ...     operators.remove_body(model, "old_body")
        """
        self._sync_coordinate_defaults()
        yield self
        self.model.finalizeConnections()
        self.state = self.model.initSystem()

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
        self._sync_coordinate_defaults()
        self.state = self.model.initSystem()

    def export(
        self, model_path: str | Path, *, geometry_dir_name: str = "Geometry"
    ) -> Path:
        """Save the current model, including scaling and posture, as ``.osim``.

        OpenSim keeps coordinate values in the ``State`` rather than in the
        model itself, so each coordinate's default value (and, for a
        coupled coordinate such as a patella, its dependent default) is
        synced from the current state before serializing; otherwise the
        exported file would reopen in the model's original, unposed
        configuration.

        Any mesh referenced by an attached body geometry is also copied next
        to the exported file, under ``geometry_dir_name`` -- the folder name
        OpenSim/Simbody search for automatically next to a ``.osim`` file --
        so the export is self-contained and portable on its own.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Destination path for the exported model file.
        geometry_dir_name : str, optional
            Name of the sibling folder receiving copies of the referenced
            mesh files. Defaults to ``"Geometry"``.

        Returns
        -------
        pathlib.Path
            Path to the written file.
        """
        self._sync_coordinate_defaults()
        destination = Path(model_path)
        self.model.printToXML(str(destination))
        self._export_mesh_files(destination.parent / geometry_dir_name)
        return destination

    def _referenced_mesh_filenames(self) -> set[str]:
        filenames: set[str] = set()
        for body in _iter_set(self.bodies):
            geometry_property = body.getPropertyByName("attached_geometry")
            for index in range(geometry_property.size()):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is not None:
                    filenames.add(mesh.get_mesh_file())
        return filenames

    def _resolve_geometry_file(self, filename: str) -> Path | None:
        search_dirs = list(self._geometry_dirs)
        if self.model_path is not None:
            search_dirs.append(self.model_path.parent)
        for directory in search_dirs:
            candidate = Path(directory) / filename
            if candidate.is_file():
                return candidate
        return None

    def _export_mesh_files(self, destination_dir: Path) -> None:
        sources = {
            filename: source
            for filename in self._referenced_mesh_filenames()
            if (source := self._resolve_geometry_file(filename)) is not None
        }
        if not sources:
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in sources.items():
            destination = destination_dir / filename
            if source.resolve() == destination.resolve():
                continue
            shutil.copy2(source, destination)

    def add_geometry_directory(self, directory: str | Path) -> None:
        """Register a directory to search for mesh geometry files.

        Registers ``directory`` both with OpenSim's own native mesh search
        (so a ``Mesh`` attached/loaded after this call resolves its file
        immediately, not just lazily on :meth:`show`) and in this
        instance's own directory list (used by :meth:`show` to extend the
        visualizer's search path, and returned by
        :attr:`geometry_directories`).

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory containing VTP/STL/OBJ (or other) geometry files.
        """
        _register_geometry_search_path(self.opensim, directory)
        self._geometry_dirs.append(Path(directory))

    @property
    def geometry_directories(self) -> tuple[Path, ...]:
        """Return the directories registered for mesh geometry search."""
        return tuple(self._geometry_dirs)

    @property
    def visualizer(self) -> Any | None:
        """Return the native OpenSim visualizer after :meth:`show` starts it."""
        return self._visualizer

    def show(self, geometry_path: str | Path | None = None) -> None:
        """Open a native Simbody window showing the current model state.

        The visualizer is initialized lazily. If enabling it requires a new
        state, the current coordinate values are copied over so the posture
        already set on this model is preserved.

        Parameters
        ----------
        geometry_path : str, pathlib.Path or None, optional
            Extra geometry search directory, in addition to any registered
            via :meth:`add_geometry_directory`.

        Raises
        ------
        RuntimeError
            If OpenSim cannot create or expose its visualizer.
        """
        coordinates = self._capture_coordinate_values()
        self.model.setUseVisualizer(True)
        try:
            self.state = self.model.initSystem()
            self._visualizer = self.model.getVisualizer()
            search_dirs = (
                *self._geometry_dirs,
                *([Path(geometry_path)] if geometry_path else []),
            )
            for directory in search_dirs:
                self._visualizer.addDirToGeometrySearchPaths(
                    str(Path(directory).resolve())
                )
            self._restore_coordinate_values(coordinates)
            self._visualizer.show(self.state)
        except Exception as error:
            raise RuntimeError(
                "Unable to open the OpenSim visualizer. Check the graphical "
                "backend and the model geometry files. On Windows, this can "
                "also happen if the active environment's 'Library/bin' "
                "directory (containing the Simbody/OpenGL DLLs) is missing "
                "from PATH."
            ) from error

    def _capture_coordinate_values(self) -> dict[str, float]:
        coordinates = self.model.getCoordinateSet()
        return {
            coordinates.get(index)
            .getName(): coordinates.get(index)
            .getValue(self.state)
            for index in range(coordinates.getSize())
        }

    def _restore_coordinate_values(self, values: dict[str, float]) -> None:
        if not values:
            return
        coordinates = self.model.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            value = values[coordinate.getName()]
            if not coordinate.get_locked():
                coordinate.setValue(self.state, value, False)
        self.model.realizePosition(self.state)

    def add_model(self, other: "OpenSimModel", *, name: str | None = None) -> None:
        """Merge another model's components into this one, in place.

        Colliding component names are renamed with an automatic prefix in
        the copy so both models coexist; ``other`` itself is never modified
        (its own coordinate defaults are updated to match its current
        posture/velocity, as a side effect of building a correct clone, but
        its ``State`` and every other component are left untouched). Joints
        anchored to ``other``'s ground stay anchored to this model's own
        shared ground, so both models keep their original default placement
        in the combined model.

        Parameters
        ----------
        other : OpenSimModel
            Model whose components are copied into this one.
        name : str or None, optional
            Prefix used to disambiguate colliding component names. Defaults
            to ``other``'s class name, lowercased.

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        """
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel, got {type(other).__name__!r}"
            )

        # Joints own their coordinates, so other's posture/velocity must be
        # baked into each coordinate's default *before* cloning below, or
        # the clones would revert to the original, unposed model defaults.
        other._sync_coordinate_defaults()
        self._sync_coordinate_defaults()

        prefix = name or type(other).__name__.lower()
        source = other.model

        ground_parent_joints: set[str] = set()
        ground_child_joints: set[str] = set()
        for joint in _iter_set(source.getJointSet()):
            if self.opensim.Ground.safeDownCast(joint.getParentFrame()) is not None:
                ground_parent_joints.add(joint.getName())
            if self.opensim.Ground.safeDownCast(joint.getChildFrame()) is not None:
                ground_child_joints.add(joint.getName())

        existing_names = {
            set_key: {
                item.getName() for item in _iter_set(getattr(self.model, getter)())
            }
            for set_key, getter in self._MERGE_SETS
        }

        clones: dict[str, list[Any]] = {set_key: [] for set_key, _ in self._MERGE_SETS}
        renamed_paths: dict[str, str] = {}
        joint_clones_by_original_name: dict[str, Any] = {}

        for set_key, getter in self._MERGE_SETS:
            for item in _iter_set(getattr(source, getter)()):
                clone = item.clone()
                original_name = clone.getName()
                final_name = original_name
                if final_name in existing_names[set_key]:
                    final_name = _unique_component_name(
                        prefix, original_name, existing_names[set_key]
                    )
                    renamed_paths[f"/{set_key}/{original_name}"] = (
                        f"/{set_key}/{final_name}"
                    )
                    clone.setName(final_name)
                existing_names[set_key].add(final_name)
                clones[set_key].append(clone)
                if set_key == "jointset":
                    joint_clones_by_original_name[original_name] = clone

        added: list[tuple[str, str]] = []
        for set_key, getter in self._MERGE_SETS:
            target_set = getattr(self.model, getter)()
            for clone in clones[set_key]:
                target_set.adoptAndAppend(clone)
                clone.thisown = False  # ownership now belongs to self.model; avoids a double-free crash on GC
                added.append((set_key, clone.getName()))

        self.model.finalizeFromProperties()
        if renamed_paths:
            _patch_sockets(self.model, renamed_paths)

        if ground_parent_joints or ground_child_joints:
            anchor_name = _unique_component_name(
                prefix, "ground_anchor", self._anchor_names
            )
            self._anchor_names.add(anchor_name)
            anchor = self.opensim.PhysicalOffsetFrame(
                anchor_name, self.model.getGround(), self.opensim.Transform()
            )
            self.model.addComponent(anchor)
            anchor.thisown = False
            anchor_path = f"/{anchor_name}"
            for original_name in ground_parent_joints:
                joint_clones_by_original_name[original_name].updSocket(
                    "parent_frame"
                ).setConnecteePath(anchor_path)
            for original_name in ground_child_joints:
                joint_clones_by_original_name[original_name].updSocket(
                    "child_frame"
                ).setConnecteePath(anchor_path)
            # The anchor frame is not tracked for removal: it is cheap to leave
            # orphaned after remove_model and OpenSim tolerates unused frames.

        self.model.finalizeConnections()
        self.state = self.model.initSystem()
        self._geometry_dirs.extend(other._geometry_dirs)
        self._merged[id(other)] = added

    def remove_model(self, other: "OpenSimModel") -> None:
        """Undo a previous :meth:`add_model` call for ``other``.

        Parameters
        ----------
        other : OpenSimModel
            Model previously merged into this one via :meth:`add_model`.

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        ValueError
            If ``other`` was never merged into this model.
        """
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel, got {type(other).__name__!r}"
            )
        manifest = self._merged.pop(id(other), None)
        if manifest is None:
            raise ValueError(
                "other was not previously merged into this model with add_model"
            )

        # Must run before any component is removed below: reading self.state
        # against a Model whose structure has already changed (components
        # removed) crashes natively instead of raising a catchable error.
        self._sync_coordinate_defaults()

        # Dependents (forces/markers/constraints/...) must be removed before the
        # joints and bodies they reference, otherwise OpenSim crashes natively
        # instead of raising a catchable error.
        for set_key, getter in reversed(self._MERGE_SETS):
            target_set = getattr(self.model, getter)()
            names = [
                component_name for key, component_name in manifest if key == set_key
            ]
            for component_name in names:
                index = target_set.getIndex(component_name)
                if index >= 0:
                    target_set.remove(index)

        self.model.finalizeConnections()
        self.state = self.model.initSystem()

    def __add__(self, other: "OpenSimModel") -> "OpenSimModel":
        """Return a new model containing the components of both operands.

        Neither operand is modified; see :meth:`add_model` for the merge
        semantics (automatic renaming on collision, shared ground).

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        """
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel, got {type(other).__name__!r}"
            )
        combined = OpenSimModel(model_path=None)
        combined.add_model(self)
        combined.add_model(other)
        return combined

    def __radd__(self, other: "OpenSimModel") -> "OpenSimModel":
        """Support ``other + self`` when ``other`` did not implement ``__add__``.

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        """
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel, got {type(other).__name__!r}"
            )
        return other.__add__(self)
