from __future__ import annotations

import contextlib
import os
import shutil
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["import_opensim", "OpenSimModel"]

# Bridges a bare opensim.Model (e.g. obtained from a component's own
# getModel()) back to the OpenSimModel wrapper that owns its `state` --
# needed because SWIG hands out a fresh Python proxy on every getModel()
# call, so identity can't be compared directly, only the underlying
# pointer address (`.this`) can; see _find_owner. Weak-valued so a
# garbage-collected OpenSimModel's entry disappears on its own.
_owners: "weakref.WeakValueDictionary[int, OpenSimModel]" = weakref.WeakValueDictionary()


def _register_owner(instance: "OpenSimModel") -> None:
    _owners[int(instance.model.this)] = instance


def _find_owner(raw_model: Any) -> "OpenSimModel | None":
    """Return the live :class:`OpenSimModel` wrapping ``raw_model``, if any.

    ``raw_model`` is a bare ``opensim.Model``, typically obtained from a
    component via its own ``getModel()``. Returns ``None`` if ``raw_model``
    is ``None`` or was never wrapped by a (still-alive) :class:`OpenSimModel`.
    """
    if raw_model is None:
        return None
    return _owners.get(int(raw_model.this))


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
    with the required native libraries reachable. Also raises OpenSim's
    console log threshold above ``Info``, so routine messages (e.g. "Loaded
    model ..." on every model load) stay silent while ``Warn``/``Error``
    messages still surface.

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
    opensim.Logger.setLevelString("Warn")  # Silence per-model "Loaded model ..." info logs
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
        self._player: Any | None = None
        self._player_window: Any | None = None
        _register_owner(self)

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
        clone._player = None
        clone._player_window = None
        _register_owner(clone)
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
    def bodies(self) -> dict[str, "components.Body"]:
        """Return every body in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Body(self, item)
            for item in _iter_set(self.model.getBodySet())
        }

    @property
    def joints(self) -> dict[str, "components.Joint"]:
        """Return every joint in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Joint(self, item)
            for item in _iter_set(self.model.getJointSet())
        }

    @property
    def muscles(self) -> dict[str, "components.Muscle"]:
        """Return every muscle in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Muscle(self, item)
            for item in _iter_set(self.model.getMuscles())
        }

    @property
    def markers(self) -> dict[str, "components.Marker"]:
        """Return every marker in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Marker(self, item)
            for item in _iter_set(self.model.getMarkerSet())
        }

    @property
    def coordinates(self) -> dict[str, "components.Coordinate"]:
        """Return every coordinate in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Coordinate(self, item)
            for item in _iter_set(self.model.getCoordinateSet())
        }

    @property
    def forces(self) -> dict[str, "components.Force"]:
        """Return every force in the model (including muscles), keyed by name."""
        from . import components

        return {
            item.getName(): components.Force(self, item)
            for item in _iter_set(self.model.getForceSet())
        }

    @property
    def constraints(self) -> dict[str, "components.Constraint"]:
        """Return every constraint in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Constraint(self, item)
            for item in _iter_set(self.model.getConstraintSet())
        }

    @property
    def controllers(self) -> dict[str, "components.Controller"]:
        """Return every controller in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Controller(self, item)
            for item in _iter_set(self.model.getControllerSet())
        }

    @property
    def contact_geometries(self) -> dict[str, "components.ContactGeometry"]:
        """Return every contact geometry in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.ContactGeometry(self, item)
            for item in _iter_set(self.model.getContactGeometrySet())
        }

    @property
    def probes(self) -> dict[str, "components.Probe"]:
        """Return every probe in the model, keyed by name."""
        from . import components

        return {
            item.getName(): components.Probe(self, item)
            for item in _iter_set(self.model.getProbeSet())
        }

    def body(self, name: str) -> "components.Body":
        """Return a body by its OpenSim name, wrapped as a :class:`~opensim_models.components.Body`.

        Parameters
        ----------
        name : str
            OpenSim body name.
        """
        from . import components

        return components.Body(self, self.model.getBodySet().get(name))

    def joint(self, name: str) -> "components.Joint":
        """Return a joint by its OpenSim name, wrapped as a :class:`~opensim_models.components.Joint`.

        Parameters
        ----------
        name : str
            OpenSim joint name.
        """
        from . import components

        return components.Joint(self, self.model.getJointSet().get(name))

    def muscle(self, name: str) -> "components.Muscle":
        """Return a muscle by its OpenSim name, wrapped as a :class:`~opensim_models.components.Muscle`.

        Parameters
        ----------
        name : str
            OpenSim muscle name.
        """
        from . import components

        return components.Muscle(self, self.model.getMuscles().get(name))

    def marker(self, name: str) -> "components.Marker":
        """Return a marker by its OpenSim name, wrapped as a :class:`~opensim_models.components.Marker`.

        Parameters
        ----------
        name : str
            OpenSim marker name.
        """
        from . import components

        return components.Marker(self, self.model.getMarkerSet().get(name))

    def coordinate(self, name: str) -> "components.Coordinate":
        """Return a coordinate by its OpenSim name, wrapped as a :class:`~opensim_models.components.Coordinate`.

        Parameters
        ----------
        name : str
            OpenSim coordinate name.
        """
        from . import components

        return components.Coordinate(self, self.model.getCoordinateSet().get(name))

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
        if len(self.muscles) > 0:
            self.model.equilibrateMuscles(self.state)
        self.model.realizeDynamics(self.state)

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
        coordinates = self.model.getCoordinateSet()
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

    def rotate(
        self,
        origin: Any,
        direction: tuple[float, float, float],
        angle_deg: float,
        inplace: bool = True,
    ) -> "tuple[float, float, float] | OpenSimModel":
        """Rotate this whole model rigidly about an axis through ``origin``.

        A thin convenience wrapper equivalent to
        ``operators.rotate_object(self, origin, direction, angle_deg,
        inplace=inplace)``: see :func:`~opensim_models.operators.rotate_object`
        for the full semantics (imported locally to avoid a circular
        import, since :mod:`opensim_models.operators` itself imports from
        this module).

        Parameters
        ----------
        origin : (x, y, z) coordinate, opensim.Marker, opensim.Joint, or opensim.Frame
            Pivot point for the rotation, in ground frame.
        direction : tuple[float, float, float]
            Direction of the rotation axis through ``origin``, in ground
            frame. Does not need to be a unit vector.
        angle_deg : float
            Rotation angle, in degrees.
        inplace : bool, optional
            When ``True`` (default), rotate this model itself and return
            its ground-attached joint's new position (or a tuple of them,
            if it has more than one). When ``False``, leave this model
            untouched and return an independent, rotated *copy* of the
            whole model instead (see :meth:`copy`).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            The new ground-frame position of this model's ground-attached
            joint (or a tuple of them, if it has more than one), when
            ``inplace=True``; otherwise the rotated copy of this model.
        """
        from .operators import rotate_object

        return rotate_object(self, origin, direction, angle_deg, inplace=inplace)

    def translate(
        self, direction: tuple[float, float, float], inplace: bool = True
    ) -> "tuple[float, float, float] | OpenSimModel":
        """Translate this whole model rigidly by ``direction``.

        A thin convenience wrapper equivalent to
        ``operators.translate_object(self, direction, inplace=inplace)``:
        see :func:`~opensim_models.operators.translate_object` for the full
        semantics (imported locally to avoid a circular import, since
        :mod:`opensim_models.operators` itself imports from this module).

        Parameters
        ----------
        direction : tuple[float, float, float]
            Displacement ``(dx, dy, dz)``, in ground frame, in metres.
        inplace : bool, optional
            When ``True`` (default), translate this model itself and
            return its ground-attached joint's new position (or a tuple of
            them, if it has more than one). When ``False``, leave this
            model untouched and return an independent, translated *copy*
            of the whole model instead (see :meth:`copy`).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            The new ground-frame position of this model's ground-attached
            joint (or a tuple of them, if it has more than one), when
            ``inplace=True``; otherwise the translated copy of this model.
        """
        from .operators import translate_object

        return translate_object(self, direction, inplace=inplace)

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
        body_set = self.model.getBodySet()
        scale_set = self.opensim.ScaleSet()
        for body_name, axes in factors.items():
            if not body_set.contains(body_name):
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
        for body in _iter_set(self.model.getBodySet()):
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

    def show(
        self,
        geometry_path: str | Path | None = None,
        *,
        motion: str | Path | Any | None = None,
        loop: bool = False,
        fps: float = 30.0,
    ) -> None:
        """Open an interactive VTK window showing the current model state.

        Each call closes and replaces any window(s) this instance already
        has open, then opens fresh ones -- so the current posture/motion is
        always exactly what's showing, but a reference to a previous
        :attr:`visualizer`/:attr:`player` is no longer live after a second
        ``show()`` call.

        Opens a single window: the 3D view in its upper area (on Windows,
        embedded directly into it; on other platforms the 3D view opens as
        its own separate window instead, since embedding relies on a
        Win32-specific reparenting call) with playback controls and a
        status bar below -- see :func:`~opensim_models._player.start_player`
        for exactly what it shows and its threading caveats. The status
        bar live-updates with ``"<model>-<component> (x, y, z)"`` for
        whatever the mouse is currently over in the 3D view, real
        OpenSim/ground-frame coordinates -- not available from a window
        showing only static geometry, which is the reason this package
        renders its own 3D view with VTK instead of delegating to
        OpenSim's native (Simbody) visualizer: that one runs as a separate
        process with no mouse-position, camera-transform, or picking API
        exposed to Python at all.

        To show several models together, merge them into one first
        (``user + screen``, or ``self.add_model(other)`` to merge in
        place) and call ``show()`` on the merged result -- a plain
        ``OpenSimModel`` only ever shows itself.

        Parameters
        ----------
        geometry_path : str, pathlib.Path or None, optional
            Extra geometry search directory, in addition to any registered
            via :meth:`add_geometry_directory` (registered the same way,
            so it also benefits :meth:`export`).
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, optional
            A motion to animate: a motion file path (``.mot``/``.sto``), or
            an already-loaded/built ``opensim.TimeSeriesTable`` (e.g. one
            written by a prior analysis). The playback window (see above)
            always opens either way; without a motion its playback controls
            (play/pause, stop, fast-forward/backward, cycle, the progress
            slider) are simply shown disabled, since there is nothing to
            play -- only the status bar is live. Defaults to ``None``.
        loop : bool, optional
            Initial state of the playback window's cycle/loop toggle.
            Ignored if ``motion`` is ``None``. Defaults to ``False``.
        fps : float, optional
            Target refresh rate for advancing playback, refreshing the
            status bar, and redrawing the visualizer, in frames per
            second. Defaults to ``30.0``.

        Raises
        ------
        RuntimeError
            If the 3D visualizer cannot be created (e.g. ``vtk`` is not
            installed, or no graphical backend is available).
        ValueError
            If ``motion`` resolves to fewer than 2 rows.
        """
        if geometry_path is not None:
            self.add_geometry_directory(geometry_path)

        coordinates = self._capture_coordinate_values()
        self.state = self.model.initSystem()
        self._restore_coordinate_values(coordinates)

        if self._player_window is not None:
            # also closes self._visualizer, on the background thread that
            # owns it: start_player's own tick() loop does this, once it
            # notices PlayerWindow.close()'s stop signal -- calling
            # self._visualizer.close() directly here instead, from this
            # (the caller's) thread, was tried and rejected, confirmed
            # directly to hang forever inside vtkRenderWindow.Finalize()
            # (see _player.py's tick() for the full explanation).
            self._player_window.close()
            self._player_window = None
            self._player = None
        self._visualizer = None

        from ._player import start_player

        # start_player itself constructs the VTKVisualizer, on the same
        # background thread that then owns/pumps its native window for
        # the rest of its life -- see that module's docstring for why.
        self._player_window = start_player(self, motion, loop=loop, fps=fps)
        self._player = self._player_window.motion_player

    @property
    def player(self) -> Any | None:
        """Return the playback state machine for the current ``show(motion=...)`` call.

        ``None`` if the last :meth:`show` call had no ``motion``.
        """
        return self._player

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

    def add_body(
        self,
        name: str,
        mass: float,
        *,
        mass_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        inertia: tuple[float, float, float, float, float, float] = (0.0,) * 6,
        mesh_files: str | Path | list[str | Path] | None = None,
        reinitialize: bool = False,
    ) -> "components.Body":
        """Construct an ``opensim.Body`` and add it to this model.

        A thin wrapper equivalent to ``operators.add_body(self, name, mass,
        ...)``: see :func:`~opensim_models.operators.add_body` for the full
        semantics (imported locally to avoid a circular import, since
        :mod:`opensim_models.operators` itself imports from this module).
        """
        from . import operators

        return operators.add_body(
            self,
            name,
            mass,
            mass_center=mass_center,
            inertia=inertia,
            mesh_files=mesh_files,
            reinitialize=reinitialize,
        )

    def add_free_joint(
        self,
        name: str,
        child_body: Any,
        *,
        parent_frame: Any = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Construct an ``opensim.FreeJoint`` (6 dof) and add it to this model.

        A thin wrapper equivalent to ``operators.add_free_joint(self, name,
        child_body, ...)``: see :func:`~opensim_models.operators.add_free_joint`
        for the full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_free_joint(
            self,
            name,
            child_body,
            parent_frame=parent_frame,
            position=position,
            orientation_deg=orientation_deg,
            child_position=child_position,
            child_orientation_deg=child_orientation_deg,
            reinitialize=reinitialize,
        )

    def add_pin_joint(
        self,
        name: str,
        child_body: Any,
        *,
        parent_frame: Any = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Construct an ``opensim.PinJoint`` (1 rotational dof) and add it to this model.

        A thin wrapper equivalent to ``operators.add_pin_joint(self, name,
        child_body, ...)``: see :func:`~opensim_models.operators.add_pin_joint`
        for the full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_pin_joint(
            self,
            name,
            child_body,
            parent_frame=parent_frame,
            position=position,
            orientation_deg=orientation_deg,
            child_position=child_position,
            child_orientation_deg=child_orientation_deg,
            reinitialize=reinitialize,
        )

    def add_ball_joint(
        self,
        name: str,
        child_body: Any,
        *,
        parent_frame: Any = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Construct an ``opensim.BallJoint`` (3 rotational dof) and add it to this model.

        A thin wrapper equivalent to ``operators.add_ball_joint(self, name,
        child_body, ...)``: see :func:`~opensim_models.operators.add_ball_joint`
        for the full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_ball_joint(
            self,
            name,
            child_body,
            parent_frame=parent_frame,
            position=position,
            orientation_deg=orientation_deg,
            child_position=child_position,
            child_orientation_deg=child_orientation_deg,
            reinitialize=reinitialize,
        )

    def add_slider_joint(
        self,
        name: str,
        child_body: Any,
        *,
        parent_frame: Any = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Construct an ``opensim.SliderJoint`` (1 translational dof) and add it to this model.

        A thin wrapper equivalent to ``operators.add_slider_joint(self,
        name, child_body, ...)``: see
        :func:`~opensim_models.operators.add_slider_joint` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_slider_joint(
            self,
            name,
            child_body,
            parent_frame=parent_frame,
            position=position,
            orientation_deg=orientation_deg,
            child_position=child_position,
            child_orientation_deg=child_orientation_deg,
            reinitialize=reinitialize,
        )

    def add_weld_joint(
        self,
        name: str,
        child_body: Any,
        *,
        parent_frame: Any = None,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Construct an ``opensim.WeldJoint`` (0 dof) and add it to this model.

        A thin wrapper equivalent to ``operators.add_weld_joint(self, name,
        child_body, ...)``: see :func:`~opensim_models.operators.add_weld_joint`
        for the full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_weld_joint(
            self,
            name,
            child_body,
            parent_frame=parent_frame,
            position=position,
            orientation_deg=orientation_deg,
            child_position=child_position,
            child_orientation_deg=child_orientation_deg,
            reinitialize=reinitialize,
        )

    def add_force(self, force: Any, *, reinitialize: bool = False) -> "components.Force":
        """Add an already-constructed force/actuator to this model.

        A thin wrapper equivalent to ``operators.add_force(self, force,
        ...)``: see :func:`~opensim_models.operators.add_force` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_force(self, force, reinitialize=reinitialize)

    def add_muscle(
        self,
        name: str,
        origin_component: Any,
        origin_position: tuple[float, float, float],
        insertion_component: Any,
        insertion_position: tuple[float, float, float],
        *,
        max_isometric_force: float = 1000.0,
        optimal_fiber_length: float = 0.1,
        tendon_slack_length: float = 0.2,
        pennation_angle_deg: float = 0.0,
        via_points: Any = (),
        muscle_class: str = "Millard2012EquilibriumMuscle",
        reinitialize: bool = False,
    ) -> "components.Muscle":
        """Build a muscle from its origin/insertion attachments and add it to this model.

        A thin wrapper equivalent to ``operators.add_muscle(self, name,
        origin_component, origin_position, insertion_component,
        insertion_position, ...)``: see
        :func:`~opensim_models.operators.add_muscle` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_muscle(
            self,
            name,
            origin_component,
            origin_position,
            insertion_component,
            insertion_position,
            max_isometric_force=max_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            tendon_slack_length=tendon_slack_length,
            pennation_angle_deg=pennation_angle_deg,
            via_points=via_points,
            muscle_class=muscle_class,
            reinitialize=reinitialize,
        )

    def add_marker(self, marker: Any, *, reinitialize: bool = False) -> "components.Marker":
        """Add an already-constructed marker to this model.

        A thin wrapper equivalent to ``operators.add_marker(self, marker,
        ...)``: see :func:`~opensim_models.operators.add_marker` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_marker(self, marker, reinitialize=reinitialize)

    def add_constraint(
        self, constraint: Any, *, reinitialize: bool = False
    ) -> "components.Constraint":
        """Add an already-constructed constraint to this model.

        A thin wrapper equivalent to ``operators.add_constraint(self,
        constraint, ...)``: see
        :func:`~opensim_models.operators.add_constraint` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_constraint(self, constraint, reinitialize=reinitialize)

    def add_weld_constraint(
        self,
        name: str,
        body1: Any,
        body2: Any,
        *,
        position1: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation1_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        position2: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation2_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        reinitialize: bool = False,
    ) -> "components.Constraint":
        """Rigidly weld ``body1`` and ``body2`` together (removes all 6 relative dof).

        A thin wrapper equivalent to ``operators.add_weld_constraint(self,
        name, body1, body2, ...)``: see
        :func:`~opensim_models.operators.add_weld_constraint` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_weld_constraint(
            self,
            name,
            body1,
            body2,
            position1=position1,
            orientation1_deg=orientation1_deg,
            position2=position2,
            orientation2_deg=orientation2_deg,
            reinitialize=reinitialize,
        )

    def add_point_constraint(
        self,
        name: str,
        body1: Any,
        position1: tuple[float, float, float],
        body2: Any,
        position2: tuple[float, float, float],
        *,
        reinitialize: bool = False,
    ) -> "components.Constraint":
        """Constrain a point fixed on ``body1`` to coincide with a point fixed on ``body2``.

        A thin wrapper equivalent to ``operators.add_point_constraint(self,
        name, body1, position1, body2, position2, ...)``: see
        :func:`~opensim_models.operators.add_point_constraint` for the full
        semantics (imported locally, see :meth:`add_body`) -- including a
        confirmed native-crash risk for a body-to-body (non-ground) pair;
        use :meth:`add_weld_constraint` for that case instead.
        """
        from . import operators

        return operators.add_point_constraint(
            self, name, body1, position1, body2, position2, reinitialize=reinitialize
        )

    def add_coordinate_coupler_constraint(
        self,
        name: str,
        independent_coordinates: Any,
        dependent_coordinate: Any,
        function: Any,
        *,
        reinitialize: bool = False,
    ) -> "components.Constraint":
        """Couple ``dependent_coordinate``'s value to ``independent_coordinates`` via ``function``.

        A thin wrapper equivalent to
        ``operators.add_coordinate_coupler_constraint(self, name,
        independent_coordinates, dependent_coordinate, function, ...)``:
        see :func:`~opensim_models.operators.add_coordinate_coupler_constraint`
        for the full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_coordinate_coupler_constraint(
            self,
            name,
            independent_coordinates,
            dependent_coordinate,
            function,
            reinitialize=reinitialize,
        )

    def add_controller(
        self, controller: Any, *, reinitialize: bool = False
    ) -> "components.Controller":
        """Add an already-constructed controller to this model.

        A thin wrapper equivalent to ``operators.add_controller(self,
        controller, ...)``: see
        :func:`~opensim_models.operators.add_controller` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_controller(self, controller, reinitialize=reinitialize)

    def add_contact_geometry(
        self, contact_geometry: Any, *, reinitialize: bool = False
    ) -> "components.ContactGeometry":
        """Add already-constructed contact geometry to this model.

        A thin wrapper equivalent to ``operators.add_contact_geometry(self,
        contact_geometry, ...)``: see
        :func:`~opensim_models.operators.add_contact_geometry` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_contact_geometry(
            self, contact_geometry, reinitialize=reinitialize
        )

    def add_probe(self, probe: Any, *, reinitialize: bool = False) -> "components.Probe":
        """Add an already-constructed probe to this model.

        A thin wrapper equivalent to ``operators.add_probe(self, probe,
        ...)``: see :func:`~opensim_models.operators.add_probe` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        return operators.add_probe(self, probe, reinitialize=reinitialize)

    def remove_body(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a body from this model, by name.

        A thin wrapper equivalent to ``operators.remove_body(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_body` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_body(self, name, reinitialize=reinitialize)

    def remove_joint(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a joint from this model, by name (any joint type).

        A thin wrapper equivalent to ``operators.remove_joint(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_joint` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_joint(self, name, reinitialize=reinitialize)

    def remove_force(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a force/actuator from this model, by name (including a muscle).

        A thin wrapper equivalent to ``operators.remove_force(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_force` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_force(self, name, reinitialize=reinitialize)

    def remove_muscle(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a muscle from this model, by name.

        A thin wrapper equivalent to ``operators.remove_muscle(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_muscle` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_muscle(self, name, reinitialize=reinitialize)

    def remove_marker(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a marker from this model, by name.

        A thin wrapper equivalent to ``operators.remove_marker(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_marker` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_marker(self, name, reinitialize=reinitialize)

    def remove_constraint(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a constraint from this model, by name (any constraint type).

        A thin wrapper equivalent to ``operators.remove_constraint(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_constraint` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_constraint(self, name, reinitialize=reinitialize)

    def remove_controller(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a controller from this model, by name.

        A thin wrapper equivalent to ``operators.remove_controller(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_controller` for the full
        semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_controller(self, name, reinitialize=reinitialize)

    def remove_contact_geometry(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a contact geometry from this model, by name.

        A thin wrapper equivalent to ``operators.remove_contact_geometry(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_contact_geometry` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_contact_geometry(self, name, reinitialize=reinitialize)

    def remove_probe(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a probe from this model, by name.

        A thin wrapper equivalent to ``operators.remove_probe(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_probe` for the
        full semantics (imported locally, see :meth:`add_body`).
        """
        from . import operators

        operators.remove_probe(self, name, reinitialize=reinitialize)

    def __add__(self, other: "OpenSimModel") -> "OpenSimModel":
        """Return a new, generic ``OpenSimModel`` with both operands merged in.

        ``other`` can be another ``OpenSimModel``, or a standalone
        component with its own private container (e.g. a
        :class:`~opensim_models.components.Box`/
        :class:`~opensim_models.components.Screen` -- detected via its
        ``_container`` attribute, so this works for any future standalone
        component the same way, with no per-class special-casing). Built
        via :meth:`add_model`, applied twice to a fresh ``OpenSimModel`` --
        neither ``self`` nor ``other`` is modified. The result is always a
        plain ``OpenSimModel``, never a subclass of either operand (e.g.
        ``user_a + user_b`` is not a ``User``): a merged model can't
        generally be relied on to satisfy a specific subclass's structural
        assumptions. Chaining (``a + b + c``) merges all three, regardless
        of grouping.

        Raises
        ------
        TypeError
            If ``other`` is neither an ``OpenSimModel`` nor a standalone
            component.
        """
        container = getattr(other, "_container", None)
        if isinstance(container, OpenSimModel):
            other = container
        elif not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel or standalone component, "
                f"got {type(other).__name__!r}"
            )
        result = OpenSimModel(model_path=None)
        result.add_model(self)
        result.add_model(other)
        return result

    def __radd__(self, other: "OpenSimModel") -> "OpenSimModel":
        """Support ``other + self`` when ``other`` did not implement ``__add__``.

        See :meth:`__add__` for what kinds of ``other`` are accepted (this
        is simply its mirror image).

        Raises
        ------
        TypeError
            If ``other`` is neither an ``OpenSimModel`` nor a standalone
            component.
        """
        container = getattr(other, "_container", None)
        if isinstance(container, OpenSimModel):
            other = container
        elif not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel or standalone component, "
                f"got {type(other).__name__!r}"
            )
        result = OpenSimModel(model_path=None)
        result.add_model(other)
        result.add_model(self)
        return result
