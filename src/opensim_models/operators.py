"""Generic add/remove operators for OpenSim model components.

Thin wrappers around OpenSim's own component ``Set`` API (``BodySet``,
``JointSet``, ``ForceSet``, ``MarkerSet``, ``ConstraintSet``,
``ControllerSet``, ``ContactGeometrySet``, ``ProbeSet``), covering every
component category an :class:`~opensim_models.model.OpenSimModel` exposes.

None of these functions rebuild the model's system by default: adding or
removing a component is a structural change that leaves ``model.state``
temporarily invalid (not just its defaults, but the state object itself --
reading it crashes the process rather than raising a catchable error).
Pass ``reinitialize=True`` for a single, self-contained call; for a batch
(e.g. a body and the joint connecting it, or several removals), wrap the
whole batch in :meth:`~opensim_models.model.OpenSimModel.structural_change`
instead and leave every call's ``reinitialize`` at its default ``False``:

>>> with model.structural_change():
...     body = operators.add_body(model, "b1", mass=2.0)
...     operators.add_joint(model, model.opensim.FreeJoint("b1_to_ground", model.model.getGround(), body))
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .model import OpenSimModel

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
    "add_controller",
    "remove_controller",
    "add_contact_geometry",
    "remove_contact_geometry",
    "add_probe",
    "remove_probe",
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
    opensim.Body
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
    return body


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
    opensim.Joint
        ``joint``, for chaining.
    """
    return add_component(model, "joint", joint, reinitialize=reinitialize)


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
    return add_component(model, "joint", joint, reinitialize=reinitialize)


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
    opensim.FreeJoint
        The newly created joint.
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
    opensim.PinJoint
        The newly created joint.
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
    opensim.BallJoint
        The newly created joint.
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
    opensim.SliderJoint
        The newly created joint.
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
    opensim.WeldJoint
        The newly created joint.
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
            body.attachGeometry(native_geometry_factory())
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
    tuple[opensim.Body, opensim.Joint]
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
    tuple[opensim.Body, opensim.Joint]
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
    tuple[opensim.Body, opensim.Joint]
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
    opensim.Force
        ``force``, for chaining.
    """
    return add_component(model, "force", force, reinitialize=reinitialize)


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
    model: "OpenSimModel", muscle: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed muscle to the model.

    A named alias of :func:`add_force`: a ``Muscle`` is a ``Force``
    subtype in OpenSim and is stored in (and removed from) the same
    ``ForceSet``.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the muscle to.
    muscle : opensim.Muscle
        The already-constructed muscle, e.g. a
        ``opensim.Millard2012EquilibriumMuscle`` with its path points
        already set.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    opensim.Muscle
        ``muscle``, for chaining.
    """
    return add_force(model, muscle, reinitialize=reinitialize)


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
    opensim.Marker
        ``marker``, for chaining.
    """
    return add_component(model, "marker", marker, reinitialize=reinitialize)


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
    opensim.Constraint
        ``constraint``, for chaining.
    """
    return add_component(model, "constraint", constraint, reinitialize=reinitialize)


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
    opensim.Controller
        ``controller``, for chaining.
    """
    return add_component(model, "controller", controller, reinitialize=reinitialize)


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
    opensim.ContactGeometry
        ``contact_geometry``, for chaining.
    """
    return add_component(
        model, "contact_geometry", contact_geometry, reinitialize=reinitialize
    )


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
    opensim.Probe
        ``probe``, for chaining.
    """
    return add_component(model, "probe", probe, reinitialize=reinitialize)


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
