"""Generic add/remove/move operators for OpenSim model components.

Thin wrappers around OpenSim's own component ``Set`` API (``BodySet``,
``JointSet``, ``ForceSet``, ``MarkerSet``, ``ConstraintSet``,
``ControllerSet``, ``ContactGeometrySet``, ``ProbeSet``), covering every
component category an :class:`~opensim_models.model.OpenSimModel` exposes,
plus two repositioning operators -- :func:`rotate_object` and
:func:`translate_object` -- that work on a component or on a whole model.

None of the add/remove functions rebuild the model's system by default:
adding or removing a component is a structural change that leaves
``model.state`` temporarily invalid (not just its defaults, but the state
object itself -- reading it crashes the process rather than raising a
catchable error). Pass ``reinitialize=True`` for a single, self-contained
call; for a batch (e.g. a body and the joint connecting it, or several
removals), wrap the whole batch in
:meth:`~opensim_models.model.OpenSimModel.structural_change` instead and
leave every call's ``reinitialize`` at its default ``False``:

>>> with model.structural_change():
...     body = operators.add_body(model, "b1", mass=2.0)
...     operators.add_joint(model, model.opensim.FreeJoint("b1_to_ground", model.model.getGround(), body.raw))

:func:`rotate_object`/:func:`translate_object` are a different kind of
operator: they don't add or remove anything, only reposition what's
already there (a ``Marker``, a joint's ``PhysicalOffsetFrame``, or an
entire :class:`~opensim_models.model.OpenSimModel`), and call
:meth:`~opensim_models.model.OpenSimModel.reinitialize` themselves, so they
need no ``reinitialize=``/``structural_change()`` handling from the caller.
:meth:`OpenSimModel.rotate <opensim_models.model.OpenSimModel.rotate>` and
:meth:`OpenSimModel.translate <opensim_models.model.OpenSimModel.translate>`
are convenience methods for rotating/translating a whole model without
importing this module directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import components
from .model import OpenSimModel, _find_owner, _iter_set

__all__ = [
    "add_component",
    "remove_component",
    "add_body",
    "remove_body",
    "add_joint",
    "remove_joint",
    "add_free_joint",
    "add_pin_joint",
    "add_ball_joint",
    "add_slider_joint",
    "add_weld_joint",
    "add_box_body",
    "add_cylinder_body",
    "add_sphere_body",
    "add_force",
    "remove_force",
    "add_muscle",
    "remove_muscle",
    "add_marker",
    "remove_marker",
    "add_constraint",
    "remove_constraint",
    "add_weld_constraint",
    "add_point_constraint",
    "add_coordinate_coupler_constraint",
    "add_controller",
    "remove_controller",
    "add_contact_geometry",
    "remove_contact_geometry",
    "add_probe",
    "remove_probe",
    "rotate_object",
    "translate_object",
]

# Component category -> the OpenSim Model accessor for its (mutable) Set.
_COMPONENT_SET_GETTERS: dict[str, str] = {
    "body": "updBodySet",
    "joint": "updJointSet",
    "force": "updForceSet",  # Muscle is a Force subtype; see add_muscle/remove_muscle.
    "marker": "updMarkerSet",
    "constraint": "updConstraintSet",
    "controller": "updControllerSet",
    "contact_geometry": "updContactGeometrySet",
    "probe": "updProbeSet",
}


def _unwrap(thing: Any) -> Any:
    """Return ``thing.raw`` if ``thing`` is a :mod:`opensim_models.components`
    wrapper, else ``thing`` unchanged.

    Every function in this module that accepts a component (a body, a
    joint's ``parent_frame``, a marker/frame to rotate, ...) needs to work
    whether the caller passes the raw ``opensim`` object or the
    Python-friendly wrapper :class:`~opensim_models.model.OpenSimModel`'s
    own accessors now return (e.g. ``model.body(name)``) -- this is the
    one place that difference is absorbed, rather than teaching every
    ``isinstance``/``safeDownCast`` check here about the wrapper type.
    """
    return getattr(thing, "raw", thing)


def _component_set(model: "OpenSimModel", kind: str) -> Any:
    try:
        getter = _COMPONENT_SET_GETTERS[kind]
    except KeyError as error:
        valid = ", ".join(sorted(_COMPONENT_SET_GETTERS))
        raise ValueError(
            f"Unknown component kind {kind!r}; expected one of {valid}"
        ) from error
    return getattr(model.model, getter)()


def add_component(
    model: "OpenSimModel", kind: str, component: Any, *, reinitialize: bool = False
) -> Any:
    """Add a component to the model's matching set, by category.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the component to.
    kind : str
        Component category: ``"body"``, ``"joint"``, ``"force"`` (includes
        muscles), ``"marker"``, ``"constraint"``, ``"controller"``,
        ``"contact_geometry"`` or ``"probe"``.
    component : Any
        The already-constructed OpenSim component (e.g. an
        ``opensim.Body`` or ``opensim.Millard2012EquilibriumMuscle``).
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately
        after adding, preserving the current posture/velocity, so the
        model is ready to use -- equivalent to wrapping just this call in
        :meth:`OpenSimModel.structural_change`. Leave ``False`` and wrap a
        whole batch of calls in that context manager instead when adding
        several components together (e.g. a body and the joint connecting
        it): reinitializing after each one individually would fail, since
        the body has no joint yet.

    Returns
    -------
    Any
        ``component``, for chaining.

    Raises
    ------
    ValueError
        If ``kind`` is not a recognized component category.
    """
    component_set = _component_set(model, kind)
    if reinitialize:
        with model.structural_change():
            component_set.adoptAndAppend(component)
            component.thisown = False  # ownership now belongs to model.model; avoids a double-free on GC
    else:
        component_set.adoptAndAppend(component)
        component.thisown = False
    return component


def remove_component(
    model: "OpenSimModel", kind: str, name: str, *, reinitialize: bool = False
) -> None:
    """Remove a named component from the model's matching set, by category.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the component from.
    kind : str
        Component category -- see :func:`add_component`.
    name : str
        Name of the component to remove.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately
        after removing, preserving the current posture/velocity --
        equivalent to wrapping just this call in
        :meth:`OpenSimModel.structural_change`. Leave ``False`` and wrap a
        whole batch of removals in that context manager instead (e.g. a
        joint and the body it connects, which must be removed together).

    Raises
    ------
    ValueError
        If ``kind`` is not a recognized component category, or if no
        component named ``name`` exists in that category.
    """
    component_set = _component_set(model, kind)
    index = component_set.getIndex(name)
    if index < 0:
        raise ValueError(f"No {kind} named {name!r} in the model")
    if reinitialize:
        with model.structural_change():
            component_set.remove(index)
    else:
        component_set.remove(index)


def _attach_mesh_files(model: "OpenSimModel", body: Any, mesh_files: Any) -> None:
    if mesh_files is None:
        return
    if isinstance(mesh_files, (str, Path)):
        mesh_files = [mesh_files]
    for mesh_file in mesh_files:
        mesh_path = Path(mesh_file)
        # Registers the search path (needed immediately: a Mesh resolves
        # its file when finalizing properties, i.e. right in attachGeometry
        # below, not lazily when the model is later shown).
        model.add_geometry_directory(mesh_path.parent)
        body.attachGeometry(model.opensim.Mesh(mesh_path.name))


def add_body(
    model: "OpenSimModel",
    name: str,
    mass: float,
    *,
    mass_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: tuple[float, float, float, float, float, float] = (0.0,) * 6,
    mesh_files: str | Path | list[str | Path] | None = None,
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.Body`` and add it to the model.

    The body is not connected to anything yet: the model cannot
    initialize its system until a joint connects it (see :func:`add_joint`)
    -- leave ``reinitialize=False`` (the default) and wrap both this call
    and the one adding the joint in
    :meth:`OpenSimModel.structural_change`.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body.
    mass : float
        Mass, in kilograms.
    mass_center : tuple[float, float, float], optional
        Centre of mass in the body's own frame, in metres. Defaults to the
        body's origin.
    inertia : tuple[float, ...], optional
        ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`` central inertia tensor. Defaults
        to all zeros.
    mesh_files : str, pathlib.Path, list thereof, or None, optional
        Path(s) to existing mesh file(s) (``.vtp``, ``.stl``, ``.obj``) to
        attach to the body as geometry, and whose containing directories
        are registered for OpenSim's geometry search (see
        :meth:`OpenSimModel.add_geometry_directory`) and for :meth:`OpenSimModel.show`.
        For a body whose mesh is a primitive shape, see
        :func:`add_box_body`, :func:`add_cylinder_body` and
        :func:`add_sphere_body` instead, which can generate it for you.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Body
        The newly created body.
    """
    body = model.opensim.Body(
        name,
        float(mass),
        model.opensim.Vec3(*mass_center),
        model.opensim.Inertia(*inertia),
    )
    add_component(model, "body", body, reinitialize=False)
    _attach_mesh_files(model, body, mesh_files)
    if reinitialize:
        model.reinitialize()
    return components.Body(model, body)


def remove_body(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a body from the model.

    Remove the joint connecting it (see :func:`remove_joint`) first, and
    any force/constraint referencing it, otherwise OpenSim will crash
    natively rather than raising a catchable error -- the same ordering
    :meth:`OpenSimModel.remove_model` follows internally. Wrap the body and
    joint removal together in :meth:`OpenSimModel.structural_change`.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the body from.
    name : str
        Name of the body to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "body", name, reinitialize=reinitialize)


def add_joint(model: "OpenSimModel", joint: Any, *, reinitialize: bool = False) -> Any:
    """Add an already-constructed joint (e.g. ``opensim.FreeJoint``) to the model.

    OpenSim has many joint types with different constructor signatures
    (``FreeJoint``, ``PinJoint``, ``WeldJoint``, ``CustomJoint``, ...), so
    the joint is built by the caller with OpenSim's own API; this only
    attaches it to the model. For the common joint types, see the named,
    position/orientation-based convenience functions instead:
    :func:`add_free_joint`, :func:`add_pin_joint`, :func:`add_ball_joint`,
    :func:`add_slider_joint`, :func:`add_weld_joint`.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the joint to.
    joint : opensim.Joint
        The already-constructed joint, e.g.
        ``model.opensim.FreeJoint(name, model.model.getGround(), body)``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        ``joint``, wrapped, for chaining.
    """
    joint = _unwrap(joint)
    add_component(model, "joint", joint, reinitialize=reinitialize)
    return components.Joint(model, joint)


def remove_joint(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a joint from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the joint from.
    name : str
        Name of the joint to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "joint", name, reinitialize=reinitialize)


# Joint type -> the opensim.<Joint> class it's built from, for the named,
# position/orientation-based convenience functions below.
_JOINT_CLASSES: dict[str, str] = {
    "free": "FreeJoint",
    "pin": "PinJoint",
    "ball": "BallJoint",
    "slider": "SliderJoint",
    "weld": "WeldJoint",
}


def _add_offset_joint(
    model: "OpenSimModel",
    joint_type: str,
    name: str,
    child_body: Any,
    *,
    parent_frame: Any,
    position: tuple[float, float, float],
    orientation_deg: tuple[float, float, float],
    child_position: tuple[float, float, float],
    child_orientation_deg: tuple[float, float, float],
    reinitialize: bool,
) -> Any:
    child_body = _unwrap(child_body)
    parent_frame = _unwrap(parent_frame)
    joint_class = getattr(model.opensim, _JOINT_CLASSES[joint_type])
    joint = joint_class(
        name,
        parent_frame if parent_frame is not None else model.model.getGround(),
        model.opensim.Vec3(*position),
        model.opensim.Vec3(*np.deg2rad(orientation_deg)),
        child_body,
        model.opensim.Vec3(*child_position),
        model.opensim.Vec3(*np.deg2rad(child_orientation_deg)),
    )
    add_component(model, "joint", joint, reinitialize=reinitialize)
    return components.Joint(model, joint)


def add_free_joint(
    model: "OpenSimModel",
    name: str,
    child_body: Any,
    *,
    parent_frame: Any = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.FreeJoint`` (6 dof) and add it to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the joint to.
    name : str
        Name for the new joint.
    child_body : opensim.Body
        Body the joint connects (its origin becomes the joint's child
        frame, offset by ``child_position``/``child_orientation_deg``).
    parent_frame : opensim.PhysicalFrame or None, optional
        Frame the joint connects ``child_body`` to. Defaults to
        ``model.model.getGround()``.
    position : tuple[float, float, float], optional
        Joint location in ``parent_frame``, in metres. With the body's own
        ``mass_center`` left at the origin (see :func:`add_body`), this is
        the body's centre of mass position in ``parent_frame``.
    orientation_deg : tuple[float, float, float], optional
        Joint orientation in ``parent_frame``, as X-Y-Z body-fixed Euler
        angles in degrees about ``parent_frame``'s own axes.
    child_position, child_orientation_deg : tuple[float, float, float], optional
        Same as ``position``/``orientation_deg``, but for the offset on
        ``child_body``'s side of the joint. Defaults to no offset (the
        joint sits at the body's origin/mass centre with no added tilt).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.
    """
    return _add_offset_joint(
        model,
        "free",
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
    model: "OpenSimModel",
    name: str,
    child_body: Any,
    *,
    parent_frame: Any = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.PinJoint`` (1 rotational dof, about its Z axis) and add it.

    Parameters are the same as :func:`add_free_joint`; the pin's rotation
    axis is the Z axis of its own joint frame, so use ``orientation_deg``
    to point that axis wherever the hinge should rotate about.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.
    """
    return _add_offset_joint(
        model,
        "pin",
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
    model: "OpenSimModel",
    name: str,
    child_body: Any,
    *,
    parent_frame: Any = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.BallJoint`` (3 rotational dof) and add it.

    Parameters are the same as :func:`add_free_joint`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.
    """
    return _add_offset_joint(
        model,
        "ball",
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
    model: "OpenSimModel",
    name: str,
    child_body: Any,
    *,
    parent_frame: Any = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.SliderJoint`` (1 translational dof, along its X axis) and add it.

    Parameters are the same as :func:`add_free_joint`; the slider's
    translation axis is the X axis of its own joint frame, so use
    ``orientation_deg`` to point that axis wherever it should slide along.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.
    """
    return _add_offset_joint(
        model,
        "slider",
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
    model: "OpenSimModel",
    name: str,
    child_body: Any,
    *,
    parent_frame: Any = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.WeldJoint`` (0 dof, rigid attachment) and add it.

    Parameters are the same as :func:`add_free_joint`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.
    """
    return _add_offset_joint(
        model,
        "weld",
        name,
        child_body,
        parent_frame=parent_frame,
        position=position,
        orientation_deg=orientation_deg,
        child_position=child_position,
        child_orientation_deg=child_orientation_deg,
        reinitialize=reinitialize,
    )


# Joint type -> the add_*_joint function to connect a primitive body to
# ground with, for add_box_body/add_cylinder_body/add_sphere_body below.
_JOINT_ADDERS = {
    "free": add_free_joint,
    "pin": add_pin_joint,
    "ball": add_ball_joint,
    "slider": add_slider_joint,
    "weld": add_weld_joint,
}


def _add_primitive_body(
    model: "OpenSimModel",
    name: str,
    mass: float,
    inertia: tuple[float, float, float, float, float, float],
    *,
    position: tuple[float, float, float],
    orientation_deg: tuple[float, float, float],
    joint_type: str,
    write_mesh: Any,
    native_geometry_factory: Any,
    mesh: bool,
    mesh_dir: str | Path | None,
    mesh_filename: str,
    reinitialize: bool,
) -> tuple[Any, Any]:
    if joint_type not in _JOINT_ADDERS:
        valid = ", ".join(sorted(_JOINT_ADDERS))
        raise ValueError(f"Unknown joint_type {joint_type!r}; expected one of {valid}")
    if mesh and mesh_dir is None:
        raise ValueError("mesh_dir is required when mesh=True")

    mesh_files = None
    if mesh:
        mesh_path = Path(mesh_dir) / mesh_filename
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(mesh_path)
        mesh_files = mesh_path

    def build() -> tuple[Any, Any]:
        body = add_body(model, name, mass, inertia=inertia, mesh_files=mesh_files)
        if not mesh:
            body.raw.attachGeometry(native_geometry_factory())
        joint = _JOINT_ADDERS[joint_type](
            model,
            f"{name}_joint",
            body,
            position=position,
            orientation_deg=orientation_deg,
        )
        return body, joint

    if reinitialize:
        with model.structural_change():
            return build()
    return build()


def add_box_body(
    model: "OpenSimModel",
    name: str,
    size: tuple[float, float, float],
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a box-shaped body, positioned and oriented by its own joint.

    Mass and (central) inertia are derived analytically from ``size`` and
    ``density``, so they always match what is actually drawn.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    size : tuple[float, float, float]
        Full extents ``(size_x, size_y, size_z)`` of the box, in metres.
    density : float, optional
        Density, in kg/m^3, used to derive mass and inertia from ``size``.
        Defaults to ``1000.0`` (water), a generic placeholder.
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        Joint connecting the body to ground: one of ``"weld"`` (default,
        rigid), ``"free"``, ``"pin"``, ``"ball"`` or ``"slider"``.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Brick`` geometry (no file written). When ``True``,
        generate and attach an actual box mesh file instead (useful for
        exporting the model portably) -- requires ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately,
        preserving the current posture/velocity, so the model is ready to
        use. Leave ``False`` and wrap several primitive-body calls in one
        :meth:`OpenSimModel.structural_change` block to add them as a
        batch, rebuilding the system only once.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body and the joint connecting it to ground.

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    size_x, size_y, size_z = size
    volume = size_x * size_y * size_z
    mass = volume * density
    inertia = (
        mass * (size_y**2 + size_z**2) / 12.0,
        mass * (size_x**2 + size_z**2) / 12.0,
        mass * (size_x**2 + size_y**2) / 12.0,
        0.0,
        0.0,
        0.0,
    )

    def write_mesh(path: Path) -> None:
        from . import _primitives

        _primitives.write_box_mesh(path, size_x, size_y, size_z)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Brick(
            model.opensim.Vec3(size_x / 2.0, size_y / 2.0, size_z / 2.0)
        ),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


def add_cylinder_body(
    model: "OpenSimModel",
    name: str,
    radius: float,
    height: float,
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a cylinder-shaped body, positioned and oriented by its own joint.

    The cylinder's axis is its local Y axis (matching ``opensim.Cylinder``'s
    own convention); use ``orientation_deg`` to point it elsewhere. Mass
    and (central) inertia are derived analytically from ``radius``,
    ``height`` and ``density``.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    radius : float
        Base radius, in metres.
    height : float
        Height along the cylinder's own Y axis, in metres.
    density : float, optional
        Density, in kg/m^3. Defaults to ``1000.0`` (water).
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        See :func:`add_box_body`.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Cylinder`` geometry (no file written). When ``True``,
        generate and attach an actual mesh file instead -- requires
        ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        See :func:`add_box_body`.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body and the joint connecting it to ground.

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    volume = math.pi * radius**2 * height
    mass = volume * density
    inertia = (
        mass * (3.0 * radius**2 + height**2) / 12.0,
        mass * radius**2 / 2.0,
        mass * (3.0 * radius**2 + height**2) / 12.0,
        0.0,
        0.0,
        0.0,
    )

    def write_mesh(path: Path) -> None:
        from . import _primitives

        _primitives.write_cylinder_mesh(path, radius, height)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Cylinder(radius, height / 2.0),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


def add_sphere_body(
    model: "OpenSimModel",
    name: str,
    radius: float,
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a sphere-shaped body, positioned and oriented by its own joint.

    Mass and (central) inertia are derived analytically from ``radius`` and
    ``density``. ``orientation_deg`` has no visible effect on the sphere
    itself (a solid sphere is rotationally symmetric) but still applies to
    the joint, e.g. if ``joint_type`` gives it a non-symmetric-axis motion.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    radius : float
        Radius, in metres.
    density : float, optional
        Density, in kg/m^3. Defaults to ``1000.0`` (water).
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        See :func:`add_box_body`.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Sphere`` geometry (no file written). When ``True``,
        generate and attach an actual mesh file instead -- requires
        ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        See :func:`add_box_body`.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body and the joint connecting it to ground.

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    volume = 4.0 / 3.0 * math.pi * radius**3
    mass = volume * density
    moment = 2.0 / 5.0 * mass * radius**2
    inertia = (moment, moment, moment, 0.0, 0.0, 0.0)

    def write_mesh(path: Path) -> None:
        from . import _primitives

        _primitives.write_sphere_mesh(path, radius)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Sphere(radius),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


def add_force(model: "OpenSimModel", force: Any, *, reinitialize: bool = False) -> Any:
    """Add an already-constructed force/actuator to the model.

    Muscles are a ``Force`` subtype in OpenSim (there is no separate,
    directly-addable "muscle set") -- see :func:`add_muscle` for a named
    alias when the component is specifically a muscle.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the force to.
    force : opensim.Force
        The already-constructed force/actuator, e.g. a
        ``opensim.Millard2012EquilibriumMuscle`` or
        ``opensim.CoordinateActuator``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Force
        ``force``, wrapped, for chaining.
    """
    force = _unwrap(force)
    add_component(model, "force", force, reinitialize=reinitialize)
    return components.Force(model, force)


def remove_force(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a force/actuator (including a muscle) from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the force from.
    name : str
        Name of the force to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "force", name, reinitialize=reinitialize)


def add_muscle(
    model: "OpenSimModel",
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
) -> Any:
    """Build a muscle from its origin/insertion attachments and add it to the model.

    A ``Muscle`` is a ``Force`` subtype in OpenSim and is stored in (and
    removed from) the same ``ForceSet`` -- see :func:`add_force` for
    adding any other kind of force/actuator, or any muscle type whose
    constructor doesn't match ``muscle_class``'s assumed
    ``(name, max_isometric_force, optimal_fiber_length,
    tendon_slack_length, pennation_angle)`` shape.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the muscle to.
    name : str
        Name for the new muscle.
    origin_component, insertion_component : opensim.PhysicalFrame
        The body (or other physical frame) each end of the muscle's path
        attaches to.
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
        Pennation angle at optimal fiber length, in degrees (converted to
        radians for the raw constructor, which takes radians). Defaults
        to ``0.0``.
    via_points : sequence of (component, (x, y, z)), optional
        Extra path points inserted, in order, between the origin and
        insertion attachments -- e.g. to wrap a muscle's path around a
        joint. Empty by default (a straight origin-to-insertion path).
    muscle_class : str, optional
        Name of the ``opensim`` muscle class to instantiate (looked up on
        ``model.opensim``, same resolution as the joint type -> class
        lookup :func:`add_free_joint`/:func:`add_pin_joint`/etc. use).
        Defaults to ``"Millard2012EquilibriumMuscle"``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Muscle
        The newly created muscle, wrapped.
    """
    origin_component = _unwrap(origin_component)
    insertion_component = _unwrap(insertion_component)
    muscle_cls = getattr(model.opensim, muscle_class)
    muscle = muscle_cls(
        name,
        float(max_isometric_force),
        float(optimal_fiber_length),
        float(tendon_slack_length),
        float(np.deg2rad(pennation_angle_deg)),
    )
    muscle.addNewPathPoint(f"{name}-P1", origin_component, model.opensim.Vec3(*origin_position))
    index = 2
    for component, position in via_points:
        muscle.addNewPathPoint(f"{name}-P{index}", _unwrap(component), model.opensim.Vec3(*position))
        index += 1
    muscle.addNewPathPoint(f"{name}-P{index}", insertion_component, model.opensim.Vec3(*insertion_position))
    add_component(model, "force", muscle, reinitialize=reinitialize)
    return components.Muscle(model, muscle)


def remove_muscle(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a muscle from the model. A named alias of :func:`remove_force`.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the muscle from.
    name : str
        Name of the muscle to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_force(model, name, reinitialize=reinitialize)


def add_marker(
    model: "OpenSimModel", marker: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed marker to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the marker to.
    marker : opensim.Marker
        The already-constructed marker.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Marker
        ``marker``, wrapped, for chaining.
    """
    marker = _unwrap(marker)
    add_component(model, "marker", marker, reinitialize=reinitialize)
    return components.Marker(model, marker)


def remove_marker(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a marker from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the marker from.
    name : str
        Name of the marker to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "marker", name, reinitialize=reinitialize)


def add_constraint(
    model: "OpenSimModel", constraint: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed constraint to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    constraint : opensim.Constraint
        The already-constructed constraint, e.g. an
        ``opensim.CoordinateCouplerConstraint``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        ``constraint``, wrapped, for chaining.
    """
    constraint = _unwrap(constraint)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def remove_constraint(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a constraint from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the constraint from.
    name : str
        Name of the constraint to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "constraint", name, reinitialize=reinitialize)


def add_weld_constraint(
    model: "OpenSimModel",
    name: str,
    body1: Any,
    body2: Any,
    *,
    position1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation1_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position2: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation2_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Rigidly weld ``body1`` and ``body2`` together (removes all 6 relative dof).

    Builds an ``opensim.WeldConstraint`` from two attachment frames, one
    per body -- unlike a ``WeldJoint`` (see :func:`add_weld_joint`), this
    does not change the kinematic tree: both bodies keep their own
    joints, and the constraint just forces their two attachment frames to
    coincide.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    body1, body2 : opensim.PhysicalFrame
        The two bodies (or other physical frames) to weld together.
    position1, orientation1_deg : tuple[float, float, float], optional
        Attachment point/orientation (X-Y-Z body-fixed Euler degrees) on
        ``body1``'s own frame. Defaults to no offset.
    position2, orientation2_deg : tuple[float, float, float], optional
        Same as ``position1``/``orientation1_deg``, but for ``body2``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    body1 = _unwrap(body1)
    body2 = _unwrap(body2)
    constraint = model.opensim.WeldConstraint(
        name,
        body1,
        model.opensim.Vec3(*position1),
        model.opensim.Vec3(*np.deg2rad(orientation1_deg)),
        body2,
        model.opensim.Vec3(*position2),
        model.opensim.Vec3(*np.deg2rad(orientation2_deg)),
    )
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def add_point_constraint(
    model: "OpenSimModel",
    name: str,
    body1: Any,
    position1: tuple[float, float, float],
    body2: Any,
    position2: tuple[float, float, float],
    *,
    reinitialize: bool = False,
) -> Any:
    """Constrain a point fixed on ``body1`` to coincide with a point fixed on ``body2``.

    Builds an ``opensim.PointConstraint`` (removes 3 relative translational
    dof, orientation stays free) -- use :func:`add_weld_constraint` instead
    if orientation should be locked too.

    Confirmed directly (OpenSim 4.6): a ``PointConstraint`` between two
    *non-ground* bodies crashes the process natively during
    ``initSystem()`` (not a catchable Python exception) -- regardless of
    whether the two bodies are independent branches or parent/child in the
    same chain. The identical construction with ``model.model.getGround()``
    as one of the two bodies works correctly. Until this is resolved
    upstream, treat ``add_point_constraint`` as ground-to-body only; for
    body-to-body, use :func:`add_weld_constraint` instead (confirmed not to
    have this issue), accepting the extra 3 locked rotational dof.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    body1, body2 : opensim.PhysicalFrame
        The two bodies (or other physical frames) whose points are
        constrained to coincide.
    position1 : tuple[float, float, float]
        Point, in metres, in ``body1``'s own local frame.
    position2 : tuple[float, float, float]
        Point, in metres, in ``body2``'s own local frame.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    body1 = _unwrap(body1)
    body2 = _unwrap(body2)
    constraint = model.opensim.PointConstraint(
        body1, model.opensim.Vec3(*position1), body2, model.opensim.Vec3(*position2)
    )
    constraint.setName(name)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def add_coordinate_coupler_constraint(
    model: "OpenSimModel",
    name: str,
    independent_coordinates: Any,
    dependent_coordinate: Any,
    function: Any,
    *,
    reinitialize: bool = False,
) -> Any:
    """Couple ``dependent_coordinate``'s value to ``independent_coordinates`` via ``function``.

    Builds an ``opensim.CoordinateCouplerConstraint``: whenever any
    independent coordinate changes, OpenSim re-solves ``function`` of
    their values and assigns the result to ``dependent_coordinate`` (e.g.
    a patella coupled to knee flexion).

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    independent_coordinates : str, components.Coordinate, or a sequence of either
        The coordinate(s) ``function`` is evaluated on. A single
        coordinate (name or wrapper) is also accepted directly, not just
        a sequence of one.
    dependent_coordinate : str or components.Coordinate
        The coordinate whose value ``function``'s result is assigned to.
    function : opensim.Function
        The already-built coupling function, e.g.
        ``opensim.LinearFunction(slope, intercept)``. OpenSim has many
        ``Function`` subtypes (linear, spline, constant, ...); building
        one is left to the caller, same as :func:`add_force` leaves
        building the force/actuator itself to the caller for anything
        beyond a named muscle (:func:`add_muscle`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    if isinstance(independent_coordinates, (str, components.Coordinate)):
        independent_coordinates = [independent_coordinates]
    names = model.opensim.ArrayStr()
    for coordinate in independent_coordinates:
        names.append(
            coordinate.name if isinstance(coordinate, components.Coordinate) else coordinate
        )
    dependent_name = (
        dependent_coordinate.name
        if isinstance(dependent_coordinate, components.Coordinate)
        else dependent_coordinate
    )
    constraint = model.opensim.CoordinateCouplerConstraint()
    constraint.setName(name)
    constraint.setIndependentCoordinateNames(names)
    constraint.setDependentCoordinateName(dependent_name)
    constraint.setFunction(function)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def add_controller(
    model: "OpenSimModel", controller: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed controller to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the controller to.
    controller : opensim.Controller
        The already-constructed controller.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Controller
        ``controller``, wrapped, for chaining.
    """
    controller = _unwrap(controller)
    add_component(model, "controller", controller, reinitialize=reinitialize)
    return components.Controller(model, controller)


def remove_controller(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a controller from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the controller from.
    name : str
        Name of the controller to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "controller", name, reinitialize=reinitialize)


def add_contact_geometry(
    model: "OpenSimModel", contact_geometry: Any, *, reinitialize: bool = False
) -> Any:
    """Add already-constructed contact geometry to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the contact geometry to.
    contact_geometry : opensim.ContactGeometry
        The already-constructed contact geometry, e.g. an
        ``opensim.ContactSphere``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.ContactGeometry
        ``contact_geometry``, wrapped, for chaining.
    """
    contact_geometry = _unwrap(contact_geometry)
    add_component(model, "contact_geometry", contact_geometry, reinitialize=reinitialize)
    return components.ContactGeometry(model, contact_geometry)


def remove_contact_geometry(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove contact geometry from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the contact geometry from.
    name : str
        Name of the contact geometry to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "contact_geometry", name, reinitialize=reinitialize)


def add_probe(model: "OpenSimModel", probe: Any, *, reinitialize: bool = False) -> Any:
    """Add an already-constructed probe to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the probe to.
    probe : opensim.Probe
        The already-constructed probe.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Probe
        ``probe``, wrapped, for chaining.
    """
    probe = _unwrap(probe)
    add_component(model, "probe", probe, reinitialize=reinitialize)
    return components.Probe(model, probe)


def remove_probe(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a probe from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the probe from.
    name : str
        Name of the probe to remove.
    reinitialize : bool, optional
        See :func:`add_component`.
    """
    remove_component(model, "probe", name, reinitialize=reinitialize)


# ---------------------------------------------------------------------------
# Rotating an existing object about an arbitrary pivot
# ---------------------------------------------------------------------------


def _rodrigues_matrix(axis: tuple[float, float, float], angle_rad: float) -> Any:
    """Return the 3x3 rotation matrix for ``angle_rad`` about ``axis``.

    Plain-numpy Rodrigues formula, independent of OpenSim's own
    ``Rotation`` composition API: that API only exposes rotation
    composition/inversion through ``InverseRotation``, a read-only SWIG
    view without a usable ``multiply`` -- fine for the one-shot Euler
    conversion in :func:`_matrix_to_body_fixed_xyz`, but not for chaining
    the several compositions :func:`rotate_object` needs.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("direction must be a non-zero vector")
    x, y, z = axis / norm
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3)
        + np.sin(angle_rad) * skew
        + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )


def _matrix_to_body_fixed_xyz(model: "OpenSimModel", matrix: Any) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to body-fixed X-Y-Z Euler angles (radians).

    Matches the convention every ``orientation_deg`` parameter in this
    module already uses (e.g. :func:`add_weld_joint`), via OpenSim's own
    ``Rotation`` conversion rather than a hand-rolled Euler-angle formula.
    """
    rotation = model.opensim.Rotation()
    rotation.setRotationFromApproximateMat33(model.opensim.Mat33(*matrix.flatten()))
    euler = rotation.convertRotationToBodyFixedXYZ()
    return (euler.get(0), euler.get(1), euler.get(2))


def _rotation_matrix_in_ground(model: "OpenSimModel", frame: Any) -> Any:
    rotation = frame.getRotationInGround(model.state)
    matrix = rotation.asMat33()
    return np.array([[matrix.get(i, j) for j in range(3)] for i in range(3)])


def _owner_of_component(thing: Any) -> "OpenSimModel | None":
    """Return the :class:`OpenSimModel` that owns ``thing``, if resolvable.

    ``thing`` is expected to be an OpenSim component (``getModel()`` is how
    every ``ModelComponent`` -- ``Body``, ``Marker``, ``Joint``, ``Frame``,
    ...  -- reaches back to its owning ``opensim.Model``); anything else
    (a plain coordinate, an unattached component, ``None``, ...) resolves
    to ``None`` rather than raising, so callers can keep trying other
    candidates.
    """
    thing = _unwrap(thing)
    getter = getattr(thing, "getModel", None)
    if getter is None:
        return None
    try:
        raw_model = getter()
    except Exception:
        return None
    return _find_owner(raw_model)


def _resolve_model_for(obj: Any) -> "OpenSimModel | None":
    """Find the :class:`OpenSimModel` that ``obj`` itself belongs to.

    ``obj`` may be an :class:`OpenSimModel` (returned directly), an OpenSim
    component owned by one (see :func:`_owner_of_component`), or anything
    else (a plain coordinate, ``None``, ...), which resolves to ``None``.
    """
    if isinstance(obj, OpenSimModel):
        return obj
    return _owner_of_component(obj)


def _resolve_model(obj: Any, origin: Any) -> "OpenSimModel | None":
    """Find the :class:`OpenSimModel` that ``obj`` and/or ``origin`` belong to.

    ``obj`` takes priority (it is the thing actually being rotated); a
    plain ``(x, y, z)`` coordinate for either does not, by itself, require
    a model at all -- see :func:`rotate_object`.
    """
    model = _resolve_model_for(obj)
    if model is not None:
        return model
    if isinstance(origin, (tuple, list, np.ndarray)):
        return None
    return _resolve_model_for(origin)


def _resolve_ground_position(
    thing: Any, model: "OpenSimModel | None"
) -> tuple[float, float, float]:
    """Resolve ``thing`` to a ground-frame ``(x, y, z)`` position, in metres.

    ``thing`` may be a plain ``(x, y, z)`` coordinate (no ``model`` needed
    at all), an ``opensim.Marker``, an ``opensim.Joint`` (its child frame's
    position, matching the convention every joint-centre property on
    :class:`~opensim_models.models.User` already uses), or any
    ``opensim.Frame`` (a ``Body``, a ``PhysicalOffsetFrame``, ``Ground``,
    ...), in which case ``model`` must be the :class:`OpenSimModel` it
    belongs to.

    Raises
    ------
    TypeError
        If ``thing`` is none of the above, or is a component but ``model``
        is ``None``.
    """
    thing = _unwrap(thing)
    if isinstance(thing, (tuple, list, np.ndarray)):
        values = tuple(float(value) for value in thing)
        if len(values) != 3:
            raise ValueError("a coordinate must have exactly 3 values (x, y, z)")
        return values

    def unresolvable():
        raise TypeError(
            f"Cannot resolve a ground-frame position for {thing!r}: expected a "
            "(x, y, z) coordinate, or an opensim.Marker/Joint/Frame that "
            "belongs to a live OpenSimModel/User"
        )

    if model is None:
        unresolvable()

    opensim = model.opensim

    # safeDownCast raises a cryptic SWIG TypeError (rather than returning
    # None) when given something that isn't even an OpenSim object at all
    # (e.g. a plain string) -- guard with isinstance first for a clean error.
    if not isinstance(thing, opensim.OpenSimObject):
        unresolvable()

    model.model.realizePosition(model.state)

    if isinstance(thing, opensim.Marker):
        location = thing.getLocationInGround(model.state)
        return (location.get(0), location.get(1), location.get(2))

    joint = opensim.Joint.safeDownCast(thing)
    if joint is not None:
        thing = joint.getChildFrame()

    frame = opensim.Frame.safeDownCast(thing)
    if frame is None:
        unresolvable()
    position = frame.getPositionInGround(model.state)
    return (position.get(0), position.get(1), position.get(2))


def _corresponding_component(target_model: "OpenSimModel", obj: Any) -> Any:
    """Return the object in ``target_model`` at ``obj``'s absolute path.

    Used for ``inplace=False``: ``obj`` belongs to the model that was just
    cloned into ``target_model`` (:meth:`OpenSimModel.copy` keeps every
    component's name and path unchanged), so the path alone identifies the
    corresponding object in the copy. ``type(obj).safeDownCast(...)``
    returns it typed as whatever ``obj`` already was (``Marker``,
    ``PhysicalOffsetFrame``, ...), not the generic ``Component`` that
    ``getComponent`` itself returns.

    Stashes ``target_model`` on the returned component (SWIG proxies allow
    arbitrary attributes, same as the ``component.thisown = False`` idiom
    used elsewhere in this module): the component's underlying C++ memory
    is owned by ``target_model.model``, so once this function's caller
    (:func:`rotate_object`/:func:`translate_object`) returns just the
    component, nothing else would otherwise keep ``target_model`` (a local
    variable there) alive -- Python garbage-collecting it would leave the
    returned component a dangling pointer.
    """
    raw = target_model.model.getComponent(obj.getAbsolutePathString())
    found = type(obj).safeDownCast(raw)
    found._opensim_models_owner = target_model
    return found


def _rotate_marker(
    model: "OpenSimModel", obj: Any, pivot: Any, rotation_matrix: Any
) -> tuple[float, float, float]:
    """Rotate a ``Marker``'s location in place.

    Does not call :meth:`~opensim_models.model.OpenSimModel.reinitialize`:
    callers decide when (see :func:`rotate_object`).
    """
    opensim = model.opensim
    current_position = np.array(_resolve_ground_position(obj, model), dtype=float)
    new_position = pivot + rotation_matrix @ (current_position - pivot)

    parent = opensim.Frame.safeDownCast(obj.getParentFrame())
    parent_position = np.array(_resolve_ground_position(parent, model), dtype=float)
    parent_rotation = _rotation_matrix_in_ground(model, parent)
    local_position = parent_rotation.T @ (new_position - parent_position)

    obj.set_location(opensim.Vec3(*local_position))
    return tuple(float(value) for value in new_position)


def _rotate_offset_frame(
    model: "OpenSimModel", obj: Any, pivot: Any, rotation_matrix: Any
) -> tuple[float, float, float]:
    """Rotate a ``PhysicalOffsetFrame``'s translation and orientation in place.

    Does not call :meth:`~opensim_models.model.OpenSimModel.reinitialize`:
    callers rotating several frames together (see :func:`_rotate_whole_model`)
    do that once, after the whole batch.
    """
    opensim = model.opensim
    current_position = np.array(_resolve_ground_position(obj, model), dtype=float)
    current_rotation = _rotation_matrix_in_ground(model, obj)
    new_position = pivot + rotation_matrix @ (current_position - pivot)
    new_rotation = rotation_matrix @ current_rotation

    parent = opensim.PhysicalFrame.safeDownCast(obj.getParentFrame())
    parent_position = np.array(_resolve_ground_position(parent, model), dtype=float)
    parent_rotation = _rotation_matrix_in_ground(model, parent)

    local_position = parent_rotation.T @ (new_position - parent_position)
    local_rotation = parent_rotation.T @ new_rotation

    obj.set_translation(opensim.Vec3(*local_position))
    obj.set_orientation(opensim.Vec3(*_matrix_to_body_fixed_xyz(model, local_rotation)))
    return tuple(float(value) for value in new_position)


def _ground_attached_offset_frames(model: "OpenSimModel", verb: str) -> list[Any]:
    """Return the ``PhysicalOffsetFrame``\\ s of every joint of ``model`` attached to ground.

    Every body hangs, directly or transitively, off some joint; moving
    each joint that is itself attached to ground (there is usually exactly
    one, e.g. a ``User``'s ``ground_pelvis``) by the same amount moves the
    whole model as one rigid assembly, without touching any of its
    internal/relative joint coordinates. ``verb`` (``"rotate"`` or
    ``"move"``) only customizes the error messages below.
    """
    opensim = model.opensim
    root_frames = []
    for joint in _iter_set(model.model.getJointSet()):
        parent = joint.getParentFrame()
        if opensim.Ground.safeDownCast(parent) is not None:
            raise TypeError(
                f"Joint {joint.getName()!r} is attached directly to ground with "
                f"no offset frame to {verb}; rebuild it with an explicit "
                "position/orientation (e.g. via add_weld_joint) so it has one."
            )
        offset = opensim.PhysicalOffsetFrame.safeDownCast(parent)
        if offset is not None and opensim.Ground.safeDownCast(offset.getParentFrame()) is not None:
            root_frames.append(offset)

    if not root_frames:
        raise ValueError(f"model has no joint attached to ground to {verb}")
    return root_frames


def _rotate_whole_model(
    model: "OpenSimModel", pivot: Any, rotation_matrix: Any
) -> tuple[float, float, float] | tuple[tuple[float, float, float], ...]:
    """Rotate every one of ``model``'s own ground-attached joints rigidly."""
    root_frames = _ground_attached_offset_frames(model, "rotate")
    new_positions = tuple(
        _rotate_offset_frame(model, frame, pivot, rotation_matrix) for frame in root_frames
    )
    model.reinitialize()
    return new_positions[0] if len(new_positions) == 1 else new_positions


def rotate_object(
    obj: Any,
    origin: Any,
    direction: tuple[float, float, float],
    angle_deg: float,
    inplace: bool = True,
) -> Any:
    """Rotate ``obj`` by ``angle_deg`` about the axis through ``origin`` pointing along ``direction``.

    ``origin`` -- the pivot -- is resolved to a ground-frame point via
    :func:`_resolve_ground_position`: it can be a plain ``(x, y, z)``
    coordinate, or any component with a ground-frame position (an
    ``opensim.Marker``, an ``opensim.Joint``, or any ``opensim.Frame``,
    e.g. a ``Body`` or a joint's own ``PhysicalOffsetFrame``). The pivot
    does not have to coincide with ``obj``'s own origin: this rotates
    ``obj`` rigidly in place around that external point, same as rotating
    a point on a rigid body around an arbitrary axis elsewhere in space.

    ``obj`` can be:

    - An entire :class:`~opensim_models.model.OpenSimModel` (or a
      :class:`~opensim_models.models.User`): every joint of
      ``obj`` that is itself attached to ground is rotated by the same
      amount, which rigidly rotates the whole model (preserving every
      internal/relative joint angle) -- see :func:`_rotate_whole_model`.
    - An ``opensim.Marker``: its ``location`` (relative to its own parent
      frame) is updated so the marker ends up at the rotated position.
      A marker carries no orientation, so only its position changes.
    - An ``opensim.PhysicalOffsetFrame`` (e.g. a joint's parent/child
      offset frame, as built by every ``add_*_joint`` function in this
      module): both its ``translation`` and ``orientation`` properties are
      updated. This is how a joint's static placement -- set once via
      ``position=``/``orientation_deg=`` at construction time -- can be
      re-tilted afterward.
    - Anything else accepted by :func:`_resolve_ground_position` (a
      ``Body``, a ``Joint``, a generic ``Frame``, or a plain coordinate):
      there is no generic, unambiguous way to *move* these (a ``Body``'s
      placement is entirely derived from the joint connecting it, and a
      ``Joint`` is not itself a placeable thing), so only the rotated
      position is computed and returned, read-only, regardless of
      ``inplace`` -- useful to compute a ``position=``/``origin=``
      argument for another call (e.g. building a new joint with
      :func:`add_weld_joint`) without mutating anything. A plain
      coordinate needs no owning model at all: ``rotate_object((1, 0, 0),
      (0, 0, 0), (0, 0, 1), 90)`` works standalone.

    ``inplace`` controls what happens to the three cases above that are
    actually mutable (a whole model, a ``Marker``, or a
    ``PhysicalOffsetFrame``):

    - ``True`` (default): ``obj`` itself is mutated, and the return value
      is its new ground-frame position (or a tuple of them, for a whole
      model with more than one ground-attached joint) -- same as before
      this parameter existed.
    - ``False``: ``obj`` is left untouched; instead, the *model it belongs
      to* is cloned (:meth:`~opensim_models.model.OpenSimModel.copy`) and
      the rotation is applied to the corresponding object in that clone,
      located by matching ``obj``'s absolute path (stable across
      ``copy()``, see :func:`_corresponding_component`). The return value
      is then that rotated **object** -- the whole cloned model, if
      ``obj`` was a whole model; otherwise the corresponding ``Marker``/
      ``PhysicalOffsetFrame`` inside the (otherwise inaccessible unless
      you call ``.getModel()`` on it) cloned model.

    A mutated component needs its model's system rebuilt before the change
    is visible to subsequent position reads (the same reason
    :meth:`~opensim_models.model.OpenSimModel.set_body_mass` calls
    :meth:`~opensim_models.model.OpenSimModel.reinitialize` after changing
    a property): this calls it automatically, preserving the current
    posture/velocity.

    Parameters
    ----------
    obj : OpenSimModel, opensim.Marker, opensim.PhysicalOffsetFrame, or
        anything accepted by :func:`_resolve_ground_position`
        The object (or whole model) to rotate.
    origin : (x, y, z) coordinate, opensim.Marker, opensim.Joint, or opensim.Frame
        Pivot point for the rotation, in ground frame. A component must
        belong to the same model as ``obj`` (when ``obj`` is itself a
        component rather than a whole model).
    direction : tuple[float, float, float]
        Direction of the rotation axis through ``origin``, in ground frame.
        Does not need to be a unit vector.
    angle_deg : float
        Rotation angle, in degrees.
    inplace : bool, optional
        When ``True`` (default), mutate ``obj`` and return its new
        position. When ``False``, leave ``obj`` untouched and return a
        rotated copy of it instead (the whole model, for a whole-model
        ``obj``) -- see above. Has no effect on the read-only cases (a
        ``Body``, a ``Joint``, a generic ``Frame``, or a plain
        coordinate), which are never mutated either way.

    Returns
    -------
    tuple[float, float, float], OpenSimModel, opensim.Marker, or opensim.PhysicalOffsetFrame
        ``obj``'s new ground-frame position (or a tuple of them, for a
        whole model with more than one ground-attached joint) when
        ``inplace=True`` or ``obj`` is one of the read-only cases;
        otherwise (``inplace=False`` and ``obj`` is a whole model, a
        ``Marker``, or a ``PhysicalOffsetFrame``) the rotated copy itself.

    Raises
    ------
    TypeError
        If ``obj`` or ``origin`` cannot be resolved to a ground-frame
        position (see :func:`_resolve_ground_position`), or if ``obj`` is
        a model with a joint attached directly to ground with no offset
        frame to rotate.
    ValueError
        If ``direction`` is a zero vector, ``origin`` is a coordinate
        without exactly 3 values, or ``obj`` is a model with no joint
        attached to ground at all.
    """
    obj = _unwrap(obj)
    origin = _unwrap(origin)
    model = _resolve_model(obj, origin)
    pivot = np.array(_resolve_ground_position(origin, model), dtype=float)
    rotation_matrix = _rodrigues_matrix(direction, np.radians(angle_deg))

    if isinstance(obj, OpenSimModel):
        target_model = obj if inplace else obj.copy()
        new_position = _rotate_whole_model(target_model, pivot, rotation_matrix)
        return new_position if inplace else target_model

    target_model = model
    target_obj = obj
    if not inplace and model is not None:
        target_model = model.copy()
        target_obj = _corresponding_component(target_model, obj)

    if target_model is not None and isinstance(target_obj, target_model.opensim.Marker):
        new_position = _rotate_marker(target_model, target_obj, pivot, rotation_matrix)
        target_model.reinitialize()
        return new_position if inplace else target_obj

    if target_model is not None and isinstance(target_obj, target_model.opensim.PhysicalOffsetFrame):
        new_position = _rotate_offset_frame(target_model, target_obj, pivot, rotation_matrix)
        target_model.reinitialize()
        return new_position if inplace else target_obj

    current_position = np.array(_resolve_ground_position(obj, model), dtype=float)
    new_position = pivot + rotation_matrix @ (current_position - pivot)
    return tuple(float(value) for value in new_position)


# ---------------------------------------------------------------------------
# Translating an existing object
# ---------------------------------------------------------------------------


def _translate_marker(
    model: "OpenSimModel", obj: Any, direction_vector: Any
) -> tuple[float, float, float]:
    """Translate a ``Marker``'s location in place.

    Does not call :meth:`~opensim_models.model.OpenSimModel.reinitialize`:
    callers decide when (see :func:`translate_object`).
    """
    opensim = model.opensim
    parent = opensim.Frame.safeDownCast(obj.getParentFrame())
    parent_position = np.array(_resolve_ground_position(parent, model), dtype=float)
    parent_rotation = _rotation_matrix_in_ground(model, parent)

    local_delta = parent_rotation.T @ direction_vector
    current_local_location = np.array(obj.get_location().to_numpy(), dtype=float)
    new_local_location = current_local_location + local_delta
    obj.set_location(opensim.Vec3(*new_local_location))

    new_position = parent_position + parent_rotation @ new_local_location
    return tuple(float(value) for value in new_position)


def _translate_offset_frame(
    model: "OpenSimModel", obj: Any, direction_vector: Any
) -> tuple[float, float, float]:
    """Translate a ``PhysicalOffsetFrame``'s translation property in place.

    ``direction_vector`` is a displacement, in ground frame: unlike a
    position, it does not need converting by the parent's position, only
    by its orientation (:meth:`PhysicalOffsetFrame.set_translation` is
    relative to the parent frame's own axes).

    Does not call :meth:`~opensim_models.model.OpenSimModel.reinitialize`:
    callers moving several frames together (see
    :func:`_translate_whole_model`) do that once, after the whole batch.
    """
    opensim = model.opensim
    parent = opensim.PhysicalFrame.safeDownCast(obj.getParentFrame())
    parent_position = np.array(_resolve_ground_position(parent, model), dtype=float)
    parent_rotation = _rotation_matrix_in_ground(model, parent)

    local_delta = parent_rotation.T @ direction_vector
    current_local_translation = np.array(obj.get_translation().to_numpy(), dtype=float)
    new_local_translation = current_local_translation + local_delta
    obj.set_translation(opensim.Vec3(*new_local_translation))

    new_position = parent_position + parent_rotation @ new_local_translation
    return tuple(float(value) for value in new_position)


def _translate_whole_model(
    model: "OpenSimModel", direction_vector: Any
) -> tuple[float, float, float] | tuple[tuple[float, float, float], ...]:
    """Translate every one of ``model``'s own ground-attached joints rigidly."""
    root_frames = _ground_attached_offset_frames(model, "move")
    new_positions = tuple(
        _translate_offset_frame(model, frame, direction_vector) for frame in root_frames
    )
    model.reinitialize()
    return new_positions[0] if len(new_positions) == 1 else new_positions


def translate_object(
    obj: Any, direction: tuple[float, float, float], inplace: bool = True
) -> Any:
    """Translate ``obj`` by ``direction`` (``dx, dy, dz``), in ground frame.

    A pure rigid displacement -- unlike :func:`rotate_object`, there is no
    pivot to specify: every point of a rigid body shifts by the exact same
    vector under a translation. ``obj`` can be:

    - An entire :class:`~opensim_models.model.OpenSimModel` (or a
      :class:`~opensim_models.models.User`): every joint of
      ``obj`` that is itself attached to ground is shifted by the same
      amount, which rigidly translates the whole model (preserving every
      internal/relative joint angle) -- see :func:`_translate_whole_model`.
    - An ``opensim.Marker``: its ``location`` (relative to its own parent
      frame) is updated so the marker ends up at the shifted position.
    - An ``opensim.PhysicalOffsetFrame`` (e.g. a joint's parent/child
      offset frame, as built by every ``add_*_joint`` function in this
      module): its ``translation`` property is updated; ``orientation`` is
      untouched, since a translation does not rotate anything.
    - Anything else accepted by :func:`_resolve_ground_position` (a
      ``Body``, a ``Joint``, a generic ``Frame``, or a plain coordinate):
      there is no generic, unambiguous way to *move* these, so only the
      shifted position is computed and returned, read-only, regardless of
      ``inplace``. A plain coordinate needs no owning model at all:
      ``translate_object((1, 0, 0), (0, 1, 0))`` works standalone.

    ``inplace`` controls what happens to the three cases above that are
    actually mutable (a whole model, a ``Marker``, or a
    ``PhysicalOffsetFrame``) -- see :func:`rotate_object` for the full
    explanation, identical here:

    - ``True`` (default): ``obj`` itself is mutated, and the return value
      is its new ground-frame position (or a tuple of them, for a whole
      model with more than one ground-attached joint).
    - ``False``: ``obj`` is left untouched; the model it belongs to is
      cloned instead (:meth:`~opensim_models.model.OpenSimModel.copy`),
      the translation is applied to the corresponding object in that
      clone (see :func:`_corresponding_component`), and that translated
      **object** is returned -- the whole cloned model, if ``obj`` was a
      whole model; otherwise the corresponding ``Marker``/
      ``PhysicalOffsetFrame`` inside it.

    A mutated component needs its model's system rebuilt before the change
    is visible to subsequent position reads: this calls
    :meth:`~opensim_models.model.OpenSimModel.reinitialize` automatically,
    preserving the current posture/velocity (see :func:`rotate_object` for
    why).

    Parameters
    ----------
    obj : OpenSimModel, opensim.Marker, opensim.PhysicalOffsetFrame, or
        anything accepted by :func:`_resolve_ground_position`
        The object (or whole model) to translate.
    direction : tuple[float, float, float]
        Displacement ``(dx, dy, dz)``, in ground frame, in metres.
    inplace : bool, optional
        When ``True`` (default), mutate ``obj`` and return its new
        position. When ``False``, leave ``obj`` untouched and return a
        translated copy of it instead (the whole model, for a whole-model
        ``obj``) -- see above. Has no effect on the read-only cases (a
        ``Body``, a ``Joint``, a generic ``Frame``, or a plain
        coordinate), which are never mutated either way.

    Returns
    -------
    tuple[float, float, float], OpenSimModel, opensim.Marker, or opensim.PhysicalOffsetFrame
        ``obj``'s new ground-frame position (or a tuple of them, for a
        whole model with more than one ground-attached joint) when
        ``inplace=True`` or ``obj`` is one of the read-only cases;
        otherwise (``inplace=False`` and ``obj`` is a whole model, a
        ``Marker``, or a ``PhysicalOffsetFrame``) the translated copy
        itself.

    Raises
    ------
    TypeError
        If ``obj`` cannot be resolved to a ground-frame position (see
        :func:`_resolve_ground_position`), or if ``obj`` is a model with a
        joint attached directly to ground with no offset frame to move.
    ValueError
        If ``direction`` does not have exactly 3 values, or ``obj`` is a
        model with no joint attached to ground at all.
    """
    direction_vector = np.asarray(direction, dtype=float)
    if direction_vector.shape != (3,):
        raise ValueError("direction must have exactly 3 values (dx, dy, dz)")

    obj = _unwrap(obj)
    model = _resolve_model_for(obj)

    if isinstance(obj, OpenSimModel):
        target_model = obj if inplace else obj.copy()
        new_position = _translate_whole_model(target_model, direction_vector)
        return new_position if inplace else target_model

    target_model = model
    target_obj = obj
    if not inplace and model is not None:
        target_model = model.copy()
        target_obj = _corresponding_component(target_model, obj)

    if target_model is not None and isinstance(target_obj, target_model.opensim.Marker):
        new_position = _translate_marker(target_model, target_obj, direction_vector)
        target_model.reinitialize()
        return new_position if inplace else target_obj

    if target_model is not None and isinstance(target_obj, target_model.opensim.PhysicalOffsetFrame):
        new_position = _translate_offset_frame(target_model, target_obj, direction_vector)
        target_model.reinitialize()
        return new_position if inplace else target_obj

    current_position = np.array(_resolve_ground_position(obj, model), dtype=float)
    new_position = current_position + direction_vector
    return tuple(float(value) for value in new_position)
