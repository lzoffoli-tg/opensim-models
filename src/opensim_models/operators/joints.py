"""Add/remove a joint, plus the named, position/orientation-based joint-type
convenience functions (``add_free_joint``, ``add_pin_joint``, ``add_ball_joint``,
``add_slider_joint``, ``add_weld_joint``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = [
    "add_joint",
    "remove_joint",
    "add_free_joint",
    "add_pin_joint",
    "add_ball_joint",
    "add_slider_joint",
    "add_weld_joint",
]

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
    joint : opensim.Joint or components.Joint
        The already-constructed joint, e.g.
        ``model.opensim.FreeJoint(name, model.model.getGround(), body)``.
        A :class:`~opensim_models.components.Joint` wrapper is also
        accepted and unwrapped automatically (see :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        ``joint``, wrapped in :class:`~opensim_models.components.Joint`,
        which exposes ``.coordinates`` -- a ``{name: Coordinate}`` dict for
        every degree of freedom this joint owns -- on top of the common
        ``.name``/``.raw``/``.set_name()``.
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

    Raises
    ------
    ValueError
        If no joint named ``name`` exists in the model.
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
    try:
        joint_class_name = _JOINT_CLASSES[joint_type]
    except KeyError as error:
        valid = ", ".join(sorted(_JOINT_CLASSES))
        raise ValueError(
            f"Unknown joint_type {joint_type!r}; expected one of {valid}"
        ) from error
    joint_class = getattr(model.opensim, joint_class_name)
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
        The newly created joint, wrapped in
        :class:`~opensim_models.components.Joint`, which exposes
        ``.coordinates`` -- a ``{name: Coordinate}`` dict with this
        joint's 6 degrees of freedom (3 rotational, 3 translational).
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

    Same attachment-point/orientation shape as :func:`add_free_joint`; the
    pin's rotation axis is the Z axis of its own joint frame, so use
    ``orientation_deg`` to point that axis wherever the hinge should
    rotate about.

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
        angles in degrees about ``parent_frame``'s own axes -- this is
        what points the pin's Z (rotation) axis in ``parent_frame``.
    child_position, child_orientation_deg : tuple[float, float, float], optional
        Same as ``position``/``orientation_deg``, but for the offset on
        ``child_body``'s side of the joint. Defaults to no offset (the
        joint sits at the body's origin/mass centre with no added tilt).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped in
        :class:`~opensim_models.components.Joint`, whose ``.coordinates``
        dict has exactly one entry (the pin's single rotational dof,
        in radians, about its own Z axis).
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

    Same attachment-point/orientation shape as :func:`add_free_joint`,
    minus the 3 translational dof (a ``BallJoint`` only rotates, like a
    shoulder).

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
        The newly created joint, wrapped in
        :class:`~opensim_models.components.Joint`, whose ``.coordinates``
        dict has 3 entries (the ball's 3 rotational dof, in radians).
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

    Same attachment-point/orientation shape as :func:`add_free_joint`; the
    slider's translation axis is the X axis of its own joint frame, so use
    ``orientation_deg`` to point that axis wherever it should slide along.

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
        angles in degrees about ``parent_frame``'s own axes -- this is
        what points the slider's X (sliding) axis in ``parent_frame``.
    child_position, child_orientation_deg : tuple[float, float, float], optional
        Same as ``position``/``orientation_deg``, but for the offset on
        ``child_body``'s side of the joint. Defaults to no offset (the
        joint sits at the body's origin/mass centre with no added tilt).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped in
        :class:`~opensim_models.components.Joint`, whose ``.coordinates``
        dict has exactly one entry (the slider's single translational dof,
        in metres, along its own X axis).
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

    Same attachment-point/orientation shape as :func:`add_free_joint`, but
    with no degrees of freedom: once placed, ``child_body`` cannot move
    relative to ``parent_frame`` at all.

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
        The newly created joint, wrapped in
        :class:`~opensim_models.components.Joint`; its ``.coordinates``
        dict is empty, since a weld joint has no degrees of freedom.
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


