from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ._registry import _find_owner, _iter_set, _register_owner

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
        Path to an OpenSim ``.osim`` model file. When ``None`` (default),
        an empty in-memory model is created instead (used internally to
        build the result of :meth:`__add__`/:meth:`add_model`, and
        available directly for building a model up from scratch with
        :meth:`add_body`/:meth:`add_free_joint`/etc.).

    Attributes
    ----------
    model : opensim.Model
        The wrapped native OpenSim model. The escape hatch to OpenSim's
        full native API for anything this wrapper doesn't (yet) cover
        with its own property/method.
    state : opensim.State
        The current simulation state: posture, velocities, and whichever
        derived quantities have been realized so far (see
        :meth:`update_state`). Reading a derived quantity before it has
        been realized raises a native, catchable error; reading *any*
        quantity after a structural change (adding/removing a component)
        without going through :meth:`reinitialize`/:meth:`structural_change`
        first crashes the process instead. Replaced wholesale (a new
        ``opensim.State`` object) by :meth:`reinitialize`,
        :meth:`structural_change`, :meth:`scale_bodies`, :meth:`add_model`,
        :meth:`remove_model` and :meth:`show`.
    opensim : module
        The imported ``opensim`` Python bindings module (see
        :func:`import_opensim`), stored per instance so every
        ``opensim.<Class>(...)`` construction elsewhere in this package
        (and in caller code) uses the exact bindings this model was built
        from.
    model_path : pathlib.Path or None
        Path this model was loaded from, or ``None`` if it was created
        empty (``model_path=None``, or as the result of
        :meth:`__add__`/:meth:`__radd__`/:meth:`add_model`/:meth:`copy`
        starting from an empty model). Not updated by :meth:`export`: it
        always reflects where the model was *loaded* from, not the most
        recent save destination.

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
        """Load (or create) the model, unlock its coordinates, and initialize its state.

        Every coordinate the loaded model locks by default (e.g. subtalar,
        MTP or wrist coordinates in a gait-oriented base model) is unlocked
        here (see :meth:`_unlock_coordinates`), so any subclass can freely
        drive the full coordinate set -- those locks are not backed by
        constraints, so removing them is safe.

        Parameters
        ----------
        model_path : str, pathlib.Path or None, optional
            Path to an existing OpenSim ``.osim`` model file to load. When
            ``None`` (default), an empty in-memory ``opensim.Model`` is
            created instead, ready for :meth:`add_body`/:meth:`add_free_joint`/
            etc. to build up from scratch.

        Raises
        ------
        FileNotFoundError
            If ``model_path`` is given but does not point to an existing
            file.
        RuntimeError
            If the native OpenSim Python bindings cannot be imported (see
            :func:`import_opensim`).
        """
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
    def ground(self) -> Any:
        """This model's ground frame (``opensim.Ground``).

        A thin convenience over ``self.model.getGround()``, returned as the
        raw ``opensim`` object rather than wrapped in
        :class:`~opensim_models.components.Body` (``Ground`` is a
        ``PhysicalFrame``, not a ``Body`` -- it carries no mass/inertia, so
        most of that wrapper's API wouldn't apply to it). Use it directly
        as the fixed parent frame for a joint (e.g. ``operators.add_weld_joint``,
        a component's ``attach_component``/constructor ``to=``) wherever
        ``self.model.getGround()`` would otherwise be spelled out.

        Returns
        -------
        opensim.Ground
            This model's ground frame. Rebuilt fresh on every access (via
            ``self.model.getGround()``), so it always reflects the current
            ``self.model`` -- never a stale reference from before a
            structural change replaced it.
        """
        return self.model.getGround()

    @property
    def bodies(self) -> dict[str, "components.Body"]:
        """Return every body currently in the model.

        Returns
        -------
        dict[str, components.Body]
            Maps each body's OpenSim name to a
            :class:`~opensim_models.components.Body` wrapping it. Rebuilt
            fresh on every access (not cached), so it always reflects the
            model's current component tree; empty if the model has no
            bodies.
        """
        from . import components

        return {
            item.getName(): components.Body(self, item)
            for item in _iter_set(self.model.getBodySet())
        }

    @property
    def joints(self) -> dict[str, "components.Joint"]:
        """Return every joint currently in the model.

        Returns
        -------
        dict[str, components.Joint]
            Maps each joint's OpenSim name to a
            :class:`~opensim_models.components.Joint` wrapping it (of any
            joint type -- ``FreeJoint``, ``PinJoint``, ``WeldJoint``, ...).
            Rebuilt fresh on every access; empty if the model has no
            joints.
        """
        from . import components

        return {
            item.getName(): components.Joint(self, item)
            for item in _iter_set(self.model.getJointSet())
        }

    @property
    def muscles(self) -> dict[str, "components.Muscle"]:
        """Return every muscle currently in the model.

        Returns
        -------
        dict[str, components.Muscle]
            Maps each muscle's OpenSim name to a
            :class:`~opensim_models.components.Muscle` wrapping it. Unlike
            :attr:`forces`, this excludes non-muscle forces/actuators (e.g.
            a ``CoordinateActuator``). Rebuilt fresh on every access; empty
            if the model has no muscles.
        """
        from . import components

        return {
            item.getName(): components.Muscle(self, item)
            for item in _iter_set(self.model.getMuscles())
        }

    @property
    def markers(self) -> dict[str, "components.Marker"]:
        """Return every marker currently in the model.

        Returns
        -------
        dict[str, components.Marker]
            Maps each marker's OpenSim name to a
            :class:`~opensim_models.components.Marker` wrapping it. Rebuilt
            fresh on every access; empty if the model has no markers.
        """
        from . import components

        return {
            item.getName(): components.Marker(self, item)
            for item in _iter_set(self.model.getMarkerSet())
        }

    @property
    def coordinates(self) -> dict[str, "components.Coordinate"]:
        """Return every coordinate currently in the model.

        Includes every coordinate owned by every joint, locked or not
        (locks are not reflected here -- see
        :class:`~opensim_models.components.Coordinate`'s own ``locked``
        property), and any coordinate driven by a
        :class:`opensim.CoordinateCouplerConstraint`.

        Returns
        -------
        dict[str, components.Coordinate]
            Maps each coordinate's OpenSim name to a
            :class:`~opensim_models.components.Coordinate` wrapping it.
            Rebuilt fresh on every access; empty if the model has no
            coordinates.
        """
        from . import components

        return {
            item.getName(): components.Coordinate(self, item)
            for item in _iter_set(self.model.getCoordinateSet())
        }

    @property
    def forces(self) -> dict[str, "components.Force"]:
        """Return every force/actuator currently in the model, including muscles.

        A muscle is itself a ``Force`` subtype in OpenSim (there is no
        separate muscle set at the native level) -- see :attr:`muscles` to
        get only the muscles.

        Returns
        -------
        dict[str, components.Force]
            Maps each force's OpenSim name to a
            :class:`~opensim_models.components.Force` wrapping it. Rebuilt
            fresh on every access; empty if the model has no forces.
        """
        from . import components

        return {
            item.getName(): components.Force(self, item)
            for item in _iter_set(self.model.getForceSet())
        }

    @property
    def constraints(self) -> dict[str, "components.Constraint"]:
        """Return every constraint currently in the model.

        Returns
        -------
        dict[str, components.Constraint]
            Maps each constraint's OpenSim name to a
            :class:`~opensim_models.components.Constraint` wrapping it (of
            any constraint type -- ``WeldConstraint``, ``PointConstraint``,
            ``CoordinateCouplerConstraint``, ...). Rebuilt fresh on every
            access; empty if the model has no constraints.
        """
        from . import components

        return {
            item.getName(): components.Constraint(self, item)
            for item in _iter_set(self.model.getConstraintSet())
        }

    @property
    def controllers(self) -> dict[str, "components.Controller"]:
        """Return every controller currently in the model.

        Returns
        -------
        dict[str, components.Controller]
            Maps each controller's OpenSim name to a
            :class:`~opensim_models.components.Controller` wrapping it.
            Rebuilt fresh on every access; empty if the model has no
            controllers.
        """
        from . import components

        return {
            item.getName(): components.Controller(self, item)
            for item in _iter_set(self.model.getControllerSet())
        }

    @property
    def contact_geometries(self) -> dict[str, "components.ContactGeometry"]:
        """Return every contact geometry currently in the model.

        Returns
        -------
        dict[str, components.ContactGeometry]
            Maps each contact geometry's OpenSim name to a
            :class:`~opensim_models.components.ContactGeometry` wrapping it
            (e.g. an ``opensim.ContactSphere``). Rebuilt fresh on every
            access; empty if the model has no contact geometry.
        """
        from . import components

        return {
            item.getName(): components.ContactGeometry(self, item)
            for item in _iter_set(self.model.getContactGeometrySet())
        }

    @property
    def probes(self) -> dict[str, "components.Probe"]:
        """Return every probe currently in the model.

        Returns
        -------
        dict[str, components.Probe]
            Maps each probe's OpenSim name to a
            :class:`~opensim_models.components.Probe` wrapping it. Rebuilt
            fresh on every access; empty if the model has no probes.
        """
        from . import components

        return {
            item.getName(): components.Probe(self, item)
            for item in _iter_set(self.model.getProbeSet())
        }

    def body(self, name: str) -> "components.Body":
        """Look up a single body by its exact OpenSim name.

        Parameters
        ----------
        name : str
            Exact (case-sensitive) OpenSim name of the body to look up.

        Returns
        -------
        components.Body
            A :class:`~opensim_models.components.Body` wrapping the body
            named ``name``.

        Raises
        ------
        RuntimeError
            If no body named ``name`` exists in the model. This surfaces
            unchanged from OpenSim's own ``BodySet.get()`` call (a native
            ``std::exception`` crossing into Python as a ``RuntimeError``),
            not a friendlier Python exception.
        """
        from . import components

        return components.Body(self, self.model.getBodySet().get(name))

    def joint(self, name: str) -> "components.Joint":
        """Look up a single joint by its exact OpenSim name.

        Parameters
        ----------
        name : str
            Exact (case-sensitive) OpenSim name of the joint to look up.

        Returns
        -------
        components.Joint
            A :class:`~opensim_models.components.Joint` wrapping the joint
            named ``name`` (of any joint type).

        Raises
        ------
        RuntimeError
            If no joint named ``name`` exists in the model -- see
            :meth:`body` for why this is a ``RuntimeError`` rather than a
            ``KeyError``/``ValueError``.
        """
        from . import components

        return components.Joint(self, self.model.getJointSet().get(name))

    def muscle(self, name: str) -> "components.Muscle":
        """Look up a single muscle by its exact OpenSim name.

        Parameters
        ----------
        name : str
            Exact (case-sensitive) OpenSim name of the muscle to look up.
            Only muscles are searched (see :attr:`muscles`); a non-muscle
            force/actuator of the same name is not found here even though
            it exists in the same underlying ``ForceSet`` -- use
            :attr:`forces` for that.

        Returns
        -------
        components.Muscle
            A :class:`~opensim_models.components.Muscle` wrapping the
            muscle named ``name``.

        Raises
        ------
        RuntimeError
            If no muscle named ``name`` exists in the model -- see
            :meth:`body` for why this is a ``RuntimeError`` rather than a
            ``KeyError``/``ValueError``.
        """
        from . import components

        return components.Muscle(self, self.model.getMuscles().get(name))

    def marker(self, name: str) -> "components.Marker":
        """Look up a single marker by its exact OpenSim name.

        Parameters
        ----------
        name : str
            Exact (case-sensitive) OpenSim name of the marker to look up.

        Returns
        -------
        components.Marker
            A :class:`~opensim_models.components.Marker` wrapping the
            marker named ``name``.

        Raises
        ------
        RuntimeError
            If no marker named ``name`` exists in the model -- see
            :meth:`body` for why this is a ``RuntimeError`` rather than a
            ``KeyError``/``ValueError``.
        """
        from . import components

        return components.Marker(self, self.model.getMarkerSet().get(name))

    def coordinate(self, name: str) -> "components.Coordinate":
        """Look up a single coordinate by its exact OpenSim name.

        Parameters
        ----------
        name : str
            Exact (case-sensitive) OpenSim name of the coordinate to look
            up (e.g. ``"knee_angle_r"``).

        Returns
        -------
        components.Coordinate
            A :class:`~opensim_models.components.Coordinate` wrapping the
            coordinate named ``name``.

        Raises
        ------
        RuntimeError
            If no coordinate named ``name`` exists in the model -- see
            :meth:`body` for why this is a ``RuntimeError`` rather than a
            ``KeyError``/``ValueError``.
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

        Raises
        ------
        TypeError
            If ``origin`` cannot be resolved to a ground-frame position
            (it must be an ``(x, y, z)`` coordinate, an ``opensim.Marker``,
            ``opensim.Joint`` or ``opensim.Frame`` belonging to this
            model), or if this model has a joint attached directly to
            ground with no offset frame to rotate.
        ValueError
            If ``direction`` is a zero vector, ``origin`` is a coordinate
            without exactly 3 values, or this model has no joint attached
            to ground at all.
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

        Raises
        ------
        TypeError
            If this model has a joint attached directly to ground with no
            offset frame to move.
        ValueError
            If ``direction`` does not have exactly 3 values, or this model
            has no joint attached to ground at all.
        """
        from .operators import translate_object

        return translate_object(self, direction, inplace=inplace)

    def scale_bodies(self, factors: dict[str, tuple[float, float, float]]) -> None:
        """Scale one or more bodies through OpenSim's native ScaleSet pipeline.

        For each body, OpenSim scales its geometry, mass and inertia, and
        every attachment point referencing it (joints, muscle path points,
        markers, ...), consistently with the applied factors -- this is
        OpenSim's own ``Model.scale()``, not a naive geometry-only resize.
        After scaling, this preserves the current posture/velocity the same
        way :meth:`reinitialize` does (coordinate defaults, including any
        coupled coordinate, are synced from the current state) before the
        system is rebuilt.

        Parameters
        ----------
        factors : dict[str, tuple[float, float, float]]
            Maps body names to the ``(x, y, z)`` scale factors to apply to
            each, unitless multipliers along that body's own local axes
            (``1.0`` leaves that axis unchanged). A name not found in the
            model is silently skipped rather than raising. Bodies not
            listed are left unscaled.

        Raises
        ------
        ValueError
            If any scale factor in ``factors`` is non-finite or not
            strictly positive (checked only for body names that do exist
            in the model).
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
            ``model_path``, as a :class:`pathlib.Path` -- the path the
            ``.osim`` file was actually written to.

        Raises
        ------
        RuntimeError
            If the file cannot be written (e.g. ``model_path``'s parent
            directory does not exist, or the path is not writable). This
            surfaces unchanged from OpenSim's own ``printToXML()`` call (a
            native ``std::exception`` crossing into Python as a
            ``RuntimeError``), not a friendlier Python exception.
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
            Resolved to an absolute path before registering; not required
            to exist yet at call time (not checked here), but a ``Mesh``
            file that can't later be found in it will fail to resolve when
            that mesh's properties are finalized.

        Returns
        -------
        None
            Registers ``directory`` as a side effect; nothing is returned.
        """
        _register_geometry_search_path(self.opensim, directory)
        self._geometry_dirs.append(Path(directory))

    @property
    def geometry_directories(self) -> tuple[Path, ...]:
        """Return every directory registered for mesh geometry search.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Every directory passed to :meth:`add_geometry_directory` so
            far (directly, or via ``geometry_path=`` on :meth:`show`), in
            the order they were registered. Empty if none has been
            registered yet. May contain duplicates if the same directory
            was registered more than once.
        """
        return tuple(self._geometry_dirs)

    @property
    def visualizer(self) -> Any | None:
        """Return this instance's own VTK-based 3D visualizer, once :meth:`show` has started it.

        This is this package's own :class:`~opensim_models._gui.visualizer.VTKVisualizer`
        (not OpenSim's native Simbody visualizer -- see :meth:`show` for why
        this package renders its own 3D view instead). It is constructed on
        a background thread shortly after :meth:`show` starts, so it may
        still briefly be ``None`` immediately after :meth:`show` returns.
        Stays set (not reset to ``None``) after the user closes the window;
        a later :meth:`show` call resets it to ``None`` before constructing
        a fresh one.

        Returns
        -------
        Any or None
            The live :class:`~opensim_models._gui.visualizer.VTKVisualizer`
            instance, or ``None`` if :meth:`show` has never been called (or
            was just called and the background thread hasn't constructed it
            yet).
        """
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
        Win32-specific reparenting call) with Playback/View/Export controls
        below it -- see :func:`~opensim_models._gui.player.start_player`
        for exactly what it shows and its threading caveats. A floating
        tooltip next to the cursor shows ``"<model>-<component>"`` and its
        ``X``/``Y``/``Z`` ground-frame position for whatever the mouse is
        currently over in the 3D view -- not available from a window
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
            written by a prior analysis). The window always opens either
            way; without a motion, its Playback controls (play/pause,
            stop, fast-forward/backward, cycle, the progress slider) and
            its Export animation button are simply shown disabled, since
            there is nothing to play/animate -- the hover tooltip and the
            Export image button stay live regardless. Defaults to ``None``.
        loop : bool, optional
            Initial state of the playback window's cycle/loop toggle.
            Ignored if ``motion`` is ``None``. Defaults to ``False``.
        fps : float, optional
            Target refresh rate for advancing playback, refreshing the
            tooltip, and redrawing the visualizer, in frames per second;
            also the frame rate of an exported animation. Defaults to
            ``30.0``.

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
            # (see _gui/player.py's tick() for the full explanation).
            self._player_window.close()
            self._player_window = None
            self._player = None
        self._visualizer = None

        from ._gui.player import start_player

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

        Returns
        -------
        None
            Merges ``other``'s components into ``self`` in place; nothing
            is returned. The merge is recorded internally so a later
            :meth:`remove_model` call can undo it.

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

        Removes exactly the components that call added (tracked
        internally by :meth:`add_model`), in reverse order of the 8
        ``_MERGE_SETS`` categories so dependents (forces, markers,
        constraints, ...) are removed before the joints and bodies they
        reference -- removing them in the other order crashes OpenSim
        natively rather than raising a catchable error. The shared
        ground anchor frame created for a ground-attached joint (if any)
        is deliberately left behind as a harmless orphan.

        Parameters
        ----------
        other : OpenSimModel
            Model previously merged into this one via :meth:`add_model`.
            ``other`` itself is not modified or consulted beyond identity
            (``id(other)``) -- only the record of what was added on its
            behalf matters.

        Returns
        -------
        None
            Removes the previously merged components from ``self`` in
            place; nothing is returned.

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        ValueError
            If ``other`` was never merged into this model via
            :meth:`add_model`, or was already removed by a prior
            :meth:`remove_model` call.
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

        Parameters
        ----------
        name : str
            Name for the new body.
        mass : float
            Mass, in kilograms.
        mass_center : tuple[float, float, float], optional
            Centre of mass in the body's own frame, in metres. Defaults to
            the body's origin, ``(0.0, 0.0, 0.0)``.
        inertia : tuple[float, float, float, float, float, float], optional
            ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`` central inertia tensor, in
            kg*m^2. Defaults to all zeros.
        mesh_files : str, pathlib.Path, list thereof, or None, optional
            Path(s) to existing mesh file(s) (``.vtp``, ``.stl``, ``.obj``)
            to attach to the body as geometry; their containing
            directories are also registered for geometry search (see
            :meth:`add_geometry_directory`). ``None`` (default) attaches no
            mesh.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False`` -- leave it there and wrap this call together with
            the joint that connects the new body in
            :meth:`structural_change`, since the model cannot initialize a
            system while the body has no joint yet.

        Returns
        -------
        components.Body
            The newly created body, wrapped -- not yet connected to
            anything until a joint (e.g. :meth:`add_free_joint`) is added
            for it.
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

        Parameters
        ----------
        name : str
            Name for the new joint.
        child_body : opensim.Body or components.Body
            Body the joint connects (its origin becomes the joint's child
            frame, offset by ``child_position``/``child_orientation_deg``).
        parent_frame : opensim.PhysicalFrame, components wrapper, or None, optional
            Frame the joint connects ``child_body`` to. Defaults to this
            model's ground (``self.model.getGround()``) when ``None``.
        position : tuple[float, float, float], optional
            Joint location in ``parent_frame``, in metres. Defaults to
            ``(0.0, 0.0, 0.0)``; with the body's own ``mass_center`` left
            at the origin (see :meth:`add_body`), this is the body's
            centre of mass position in ``parent_frame``.
        orientation_deg : tuple[float, float, float], optional
            Joint orientation in ``parent_frame``, as X-Y-Z body-fixed
            Euler angles in degrees about ``parent_frame``'s own axes.
            Defaults to no tilt.
        child_position, child_orientation_deg : tuple[float, float, float], optional
            Same as ``position``/``orientation_deg``, but for the offset on
            ``child_body``'s side of the joint. Both default to no offset
            (the joint sits at the body's origin with no added tilt).
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created, 6-degree-of-freedom joint, wrapped.
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

        Parameters
        ----------
        name : str
            Name for the new joint.
        child_body : opensim.Body or components.Body
            Body the joint connects (its origin becomes the joint's child
            frame, offset by ``child_position``/``child_orientation_deg``).
        parent_frame : opensim.PhysicalFrame, components wrapper, or None, optional
            Frame the joint connects ``child_body`` to. Defaults to this
            model's ground when ``None``.
        position : tuple[float, float, float], optional
            Joint location in ``parent_frame``, in metres. Defaults to
            ``(0.0, 0.0, 0.0)``.
        orientation_deg : tuple[float, float, float], optional
            Joint orientation in ``parent_frame``, as X-Y-Z body-fixed
            Euler angles in degrees about ``parent_frame``'s own axes.
            The pin's rotation axis is the Z axis of its own joint frame,
            so use this to point that axis wherever the hinge should
            rotate about. Defaults to no tilt.
        child_position, child_orientation_deg : tuple[float, float, float], optional
            Same as ``position``/``orientation_deg``, but for the offset on
            ``child_body``'s side of the joint. Both default to no offset.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created, single-rotational-dof joint, wrapped.
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

        Parameters
        ----------
        name : str
            Name for the new joint.
        child_body : opensim.Body or components.Body
            Body the joint connects (its origin becomes the joint's child
            frame, offset by ``child_position``/``child_orientation_deg``).
        parent_frame : opensim.PhysicalFrame, components wrapper, or None, optional
            Frame the joint connects ``child_body`` to. Defaults to this
            model's ground when ``None``.
        position : tuple[float, float, float], optional
            Joint location in ``parent_frame``, in metres. Defaults to
            ``(0.0, 0.0, 0.0)``.
        orientation_deg : tuple[float, float, float], optional
            Joint orientation in ``parent_frame``, as X-Y-Z body-fixed
            Euler angles in degrees about ``parent_frame``'s own axes.
            Defaults to no tilt.
        child_position, child_orientation_deg : tuple[float, float, float], optional
            Same as ``position``/``orientation_deg``, but for the offset on
            ``child_body``'s side of the joint. Both default to no offset.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created, 3-rotational-dof joint, wrapped.
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

        Parameters
        ----------
        name : str
            Name for the new joint.
        child_body : opensim.Body or components.Body
            Body the joint connects (its origin becomes the joint's child
            frame, offset by ``child_position``/``child_orientation_deg``).
        parent_frame : opensim.PhysicalFrame, components wrapper, or None, optional
            Frame the joint connects ``child_body`` to. Defaults to this
            model's ground when ``None``.
        position : tuple[float, float, float], optional
            Joint location in ``parent_frame``, in metres. Defaults to
            ``(0.0, 0.0, 0.0)``.
        orientation_deg : tuple[float, float, float], optional
            Joint orientation in ``parent_frame``, as X-Y-Z body-fixed
            Euler angles in degrees about ``parent_frame``'s own axes.
            The slider's translation axis is the X axis of its own joint
            frame, so use this to point that axis wherever it should slide
            along. Defaults to no tilt.
        child_position, child_orientation_deg : tuple[float, float, float], optional
            Same as ``position``/``orientation_deg``, but for the offset on
            ``child_body``'s side of the joint. Both default to no offset.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created, single-translational-dof joint, wrapped.
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
        """Construct an ``opensim.WeldJoint`` (0 dof, rigid attachment) and add it to this model.

        A thin wrapper equivalent to ``operators.add_weld_joint(self, name,
        child_body, ...)``: see :func:`~opensim_models.operators.add_weld_joint`
        for the full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name for the new joint.
        child_body : opensim.Body or components.Body
            Body the joint connects (its origin becomes the joint's child
            frame, offset by ``child_position``/``child_orientation_deg``).
        parent_frame : opensim.PhysicalFrame, components wrapper, or None, optional
            Frame the joint connects ``child_body`` to. Defaults to this
            model's ground when ``None``.
        position : tuple[float, float, float], optional
            Joint location in ``parent_frame``, in metres. Defaults to
            ``(0.0, 0.0, 0.0)``.
        orientation_deg : tuple[float, float, float], optional
            Joint orientation in ``parent_frame``, as X-Y-Z body-fixed
            Euler angles in degrees about ``parent_frame``'s own axes.
            Defaults to no tilt.
        child_position, child_orientation_deg : tuple[float, float, float], optional
            Same as ``position``/``orientation_deg``, but for the offset on
            ``child_body``'s side of the joint. Both default to no offset.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created, zero-dof (rigid) joint, wrapped.
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

    def attach_component(
        self,
        child: Any,
        *,
        to: Any,
        child_point: Any = "com",
        parent_point: Any = "com",
        child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_type: str = "weld",
        name: str | None = None,
        reinitialize: bool = False,
    ) -> "components.Joint":
        """Re-attach ``child``'s existing joint so its parent becomes ``to``.

        A thin wrapper equivalent to ``operators.attach_component(self,
        child, to=to, ...)``: see
        :func:`~opensim_models.operators.attach_component` for the full
        semantics (imported locally, see :meth:`add_body`) -- including why
        ``child``/``to`` must already both be in this model.

        Parameters
        ----------
        child : opensim.PhysicalFrame or components wrapper
            The body (already in this model) to re-attach. Its current
            joint is removed and replaced by a new one; its placement
            before this call becomes irrelevant.
        to : opensim.PhysicalFrame or components wrapper
            The body (already in this model) ``child`` attaches to.
        child_point : ``"com"`` or tuple[float, float, float], optional
            Attachment point on ``child``, in its own local frame, in
            metres. ``"com"`` (default) uses ``child``'s centre of mass.
        parent_point : ``"com"`` or tuple[float, float, float], optional
            Attachment point on ``to``, in its own local frame, in metres.
            ``"com"`` (default) uses ``to``'s centre of mass.
        child_orientation_deg, parent_orientation_deg : tuple[float, float, float], optional
            Orientation of the new joint's frame on each side, as X-Y-Z
            body-fixed Euler degrees about that side's own local axes.
            Both default to no tilt.
        joint_type : str, optional
            One of ``"free"``, ``"pin"``, ``"ball"``, ``"slider"``, or
            ``"weld"`` (default) -- the same joint types
            :meth:`add_free_joint`/:meth:`add_pin_joint`/etc. build.
        name : str or None, optional
            Name for the new joint. Defaults (``None``) to
            ``"{child_name}_to_{to_name}"``.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after
            re-attaching, preserving the current posture/velocity.
            Defaults to ``False``.

        Returns
        -------
        components.Joint
            The newly created joint connecting ``child`` to ``to``,
            wrapped.

        Raises
        ------
        ValueError
            If ``child`` is not currently connected by any joint in this
            model (e.g. it was never merged in, or was already removed),
            or if ``joint_type`` is not one of ``"free"``, ``"pin"``,
            ``"ball"``, ``"slider"`` or ``"weld"``.
        """
        from . import operators

        return operators.attach_component(
            self,
            child,
            to=to,
            child_point=child_point,
            parent_point=parent_point,
            child_orientation_deg=child_orientation_deg,
            parent_orientation_deg=parent_orientation_deg,
            joint_type=joint_type,
            name=name,
            reinitialize=reinitialize,
        )

    def add_force(self, force: Any, *, reinitialize: bool = False) -> "components.Force":
        """Add an already-constructed force/actuator to this model.

        A thin wrapper equivalent to ``operators.add_force(self, force,
        ...)``: see :func:`~opensim_models.operators.add_force` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        force : opensim.Force
            The already-constructed force/actuator, e.g. a
            ``opensim.Millard2012EquilibriumMuscle`` or
            ``opensim.CoordinateActuator``. Muscles are a ``Force``
            subtype in OpenSim, so :meth:`add_muscle` is simply a named
            convenience for building one and adding it here.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Force
            ``force``, wrapped, for chaining.
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

        Parameters
        ----------
        name : str
            Name for the new muscle.
        origin_component, insertion_component : opensim.PhysicalFrame or components wrapper
            The body (or other physical frame) each end of the muscle's
            path attaches to.
        origin_position, insertion_position : tuple[float, float, float]
            Attachment point, in metres, in the local frame of
            ``origin_component``/``insertion_component`` respectively.
        max_isometric_force : float, optional
            Maximum isometric force, in newtons. Defaults to ``1000.0``.
        optimal_fiber_length : float, optional
            Optimal fiber length, in metres. Defaults to ``0.1``.
        tendon_slack_length : float, optional
            Tendon slack length, in metres. Defaults to ``0.2``.
        pennation_angle_deg : float, optional
            Pennation angle at optimal fiber length, in degrees (converted
            to radians for the raw constructor). Defaults to ``0.0``.
        via_points : sequence of (component, (x, y, z)), optional
            Extra path points inserted, in order, between the origin and
            insertion attachments, e.g. to wrap a muscle's path around a
            joint -- each ``(x, y, z)`` is in metres, in that point's own
            component's local frame. Empty by default (a straight
            origin-to-insertion path).
        muscle_class : str, optional
            Name of the ``opensim`` muscle class to instantiate (looked up
            as an attribute of ``self.opensim``). Must accept the
            ``(name, max_isometric_force, optimal_fiber_length,
            tendon_slack_length, pennation_angle)`` constructor signature.
            Defaults to ``"Millard2012EquilibriumMuscle"``.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Muscle
            The newly created muscle, wrapped.

        Raises
        ------
        AttributeError
            If ``muscle_class`` does not name an attribute of the
            ``opensim`` module.
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

        Parameters
        ----------
        marker : opensim.Marker
            The already-constructed marker (with its name, parent frame,
            and local-frame ``location`` already set).
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Marker
            ``marker``, wrapped, for chaining.
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

        Parameters
        ----------
        constraint : opensim.Constraint
            The already-constructed constraint, e.g. an
            ``opensim.CoordinateCouplerConstraint``. For the common
            constraint types, see the named convenience methods instead:
            :meth:`add_weld_constraint`, :meth:`add_point_constraint`,
            :meth:`add_coordinate_coupler_constraint`.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Constraint
            ``constraint``, wrapped, for chaining.
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
        semantics (imported locally, see :meth:`add_body`). Unlike
        :meth:`add_weld_joint`, this does not change the kinematic tree:
        both bodies keep their own joints, and the constraint just forces
        their two attachment frames to coincide.

        Parameters
        ----------
        name : str
            Name for the new constraint.
        body1, body2 : opensim.PhysicalFrame or components wrapper
            The two bodies (or other physical frames) to weld together.
        position1, orientation1_deg : tuple[float, float, float], optional
            Attachment point, in metres, and orientation (X-Y-Z body-fixed
            Euler degrees) on ``body1``'s own local frame. Both default to
            no offset.
        position2, orientation2_deg : tuple[float, float, float], optional
            Same as ``position1``/``orientation1_deg``, but for ``body2``.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Constraint
            The newly created constraint, wrapped.
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

        This removes 3 relative translational degrees of freedom; relative
        orientation stays free -- use :meth:`add_weld_constraint` instead
        if orientation should be locked too. **Only use this when one of
        ``body1``/``body2`` is this model's ground** (``self.model.getGround()``):
        a ``PointConstraint`` between two non-ground bodies has been
        confirmed (OpenSim 4.6) to crash the process natively during
        ``initSystem()``, not raise a catchable Python exception.

        Parameters
        ----------
        name : str
            Name for the new constraint.
        body1 : opensim.PhysicalFrame or components wrapper
            The first body (or other physical frame) whose point is
            constrained.
        position1 : tuple[float, float, float]
            Point, in metres, in ``body1``'s own local frame.
        body2 : opensim.PhysicalFrame or components wrapper
            The second body (or other physical frame) whose point is
            constrained to coincide with ``body1``'s.
        position2 : tuple[float, float, float]
            Point, in metres, in ``body2``'s own local frame.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Constraint
            The newly created constraint, wrapped.
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
        Whenever any independent coordinate changes, OpenSim re-solves
        ``function`` of their values and assigns the result to
        ``dependent_coordinate`` (e.g. a patella coupled to knee flexion).

        Parameters
        ----------
        name : str
            Name for the new constraint.
        independent_coordinates : str, components.Coordinate, or a sequence of either
            The coordinate(s) ``function`` is evaluated on, identified by
            name or by wrapper. A single coordinate is also accepted
            directly, not just a sequence of one.
        dependent_coordinate : str or components.Coordinate
            The coordinate whose value ``function``'s result is assigned
            to, identified by name or by wrapper.
        function : opensim.Function
            The already-built coupling function, e.g.
            ``opensim.LinearFunction(slope, intercept)``. Building the
            function itself is left to the caller.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Constraint
            The newly created constraint, wrapped.
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

        Parameters
        ----------
        controller : opensim.Controller
            The already-constructed controller.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Controller
            ``controller``, wrapped, for chaining.
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

        Parameters
        ----------
        contact_geometry : opensim.ContactGeometry
            The already-constructed contact geometry, e.g. an
            ``opensim.ContactSphere``.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.ContactGeometry
            ``contact_geometry``, wrapped, for chaining.
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

        Parameters
        ----------
        probe : opensim.Probe
            The already-constructed probe.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after adding,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        components.Probe
            ``probe``, wrapped, for chaining.
        """
        from . import operators

        return operators.add_probe(self, probe, reinitialize=reinitialize)

    def remove_body(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a body from this model, by name.

        A thin wrapper equivalent to ``operators.remove_body(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_body` for the
        full semantics (imported locally, see :meth:`add_body`). Remove the
        joint connecting this body (see :meth:`remove_joint`) first, and
        any force/constraint referencing it, otherwise OpenSim crashes
        natively rather than raising a catchable error -- wrap the body and
        its joint removal together in :meth:`structural_change`.

        Parameters
        ----------
        name : str
            Name of the body to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity -- only safe to use
            when this one call is the entire structural change (e.g. the
            body already has no joint/force/constraint referencing it).
            Defaults to ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no body named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_body(self, name, reinitialize=reinitialize)

    def remove_joint(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a joint from this model, by name (any joint type).

        A thin wrapper equivalent to ``operators.remove_joint(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_joint` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the joint to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity -- only safe when this
            call is the entire structural change (e.g. the orphaned body
            is removed too, in the same batch, or already has no body).
            Defaults to ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no joint named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_joint(self, name, reinitialize=reinitialize)

    def remove_force(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a force/actuator from this model, by name (including a muscle).

        A thin wrapper equivalent to ``operators.remove_force(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_force` for the
        full semantics (imported locally, see :meth:`add_body`). Muscles
        and other forces/actuators share the same underlying ``ForceSet``,
        so this removes either kind by name; :meth:`remove_muscle` is
        simply a named alias of this same method.

        Parameters
        ----------
        name : str
            Name of the force (or muscle) to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no force (or muscle) named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_force(self, name, reinitialize=reinitialize)

    def remove_muscle(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a muscle from this model, by name. A named alias of :meth:`remove_force`.

        A thin wrapper equivalent to ``operators.remove_muscle(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_muscle` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the muscle to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no muscle (force) named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_muscle(self, name, reinitialize=reinitialize)

    def remove_marker(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a marker from this model, by name.

        A thin wrapper equivalent to ``operators.remove_marker(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_marker` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the marker to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no marker named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_marker(self, name, reinitialize=reinitialize)

    def remove_constraint(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a constraint from this model, by name (any constraint type).

        A thin wrapper equivalent to ``operators.remove_constraint(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_constraint` for the full
        semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the constraint to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no constraint named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_constraint(self, name, reinitialize=reinitialize)

    def remove_controller(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a controller from this model, by name.

        A thin wrapper equivalent to ``operators.remove_controller(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_controller` for the full
        semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the controller to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no controller named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_controller(self, name, reinitialize=reinitialize)

    def remove_contact_geometry(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a contact geometry from this model, by name.

        A thin wrapper equivalent to ``operators.remove_contact_geometry(self,
        name, ...)``: see
        :func:`~opensim_models.operators.remove_contact_geometry` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the contact geometry to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no contact geometry named ``name`` exists in the model.
        """
        from . import operators

        operators.remove_contact_geometry(self, name, reinitialize=reinitialize)

    def remove_probe(self, name: str, *, reinitialize: bool = False) -> None:
        """Remove a probe from this model, by name.

        A thin wrapper equivalent to ``operators.remove_probe(self, name,
        ...)``: see :func:`~opensim_models.operators.remove_probe` for the
        full semantics (imported locally, see :meth:`add_body`).

        Parameters
        ----------
        name : str
            Name of the probe to remove.
        reinitialize : bool, optional
            When ``True``, rebuild the system immediately after removing,
            preserving the current posture/velocity. Defaults to
            ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no probe named ``name`` exists in the model.
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

        Parameters
        ----------
        other : OpenSimModel or a standalone component
            The model (or standalone component, e.g. a
            :class:`~opensim_models.components.Box`) to merge with
            ``self``. Left unmodified by this call either way.

        Returns
        -------
        OpenSimModel
            A brand-new, plain ``OpenSimModel`` containing every component
            of both ``self`` and ``other``.

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
        is simply its mirror image). Python calls this automatically for
        ``other + self`` whenever ``other``'s own ``__add__`` returns
        ``NotImplemented`` for an ``OpenSimModel`` right-hand side (or
        ``other`` has none) -- it is not meant to be called directly.

        Parameters
        ----------
        other : OpenSimModel or a standalone component
            The left-hand operand being added to ``self``. Left unmodified
            by this call either way.

        Returns
        -------
        OpenSimModel
            A brand-new, plain ``OpenSimModel`` containing every component
            of both ``other`` and ``self``.

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
