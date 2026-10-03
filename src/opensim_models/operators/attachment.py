"""Re-attach an existing component's joint to a new parent (``attach_component``)."""

from __future__ import annotations

from typing import Any

from .._registry import _iter_set
from ._shared import _unwrap
from .joints import _JOINT_CLASSES, _add_offset_joint, remove_joint

__all__ = ["attach_component"]

def _joint_owning_body(model: "OpenSimModel", body: Any) -> Any | None:
    """Return the joint whose child is ``body`` (comparing through any offset frame), or ``None``."""
    for joint in _iter_set(model.model.getJointSet()):
        child_frame = joint.getChildFrame()
        offset = model.opensim.PhysicalOffsetFrame.safeDownCast(child_frame)
        actual = offset.getParentFrame() if offset is not None else child_frame
        if int(actual.this) == int(body.this):
            return joint
    return None


def _resolve_attachment_point(body: Any, point: Any) -> tuple[float, float, float]:
    """Resolve ``point`` (``"com"`` or an explicit ``(x, y, z)``) against ``body``'s own local frame."""
    if point == "com":
        center = body.get_mass_center()
        return (center.get(0), center.get(1), center.get(2))
    return tuple(float(value) for value in point)


def attach_component(
    model: "OpenSimModel",
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
) -> Any:
    """Re-attach ``child``'s existing joint so its parent becomes ``to``.

    ``child`` must already be in ``model`` (e.g. merged in via ``model +
    box``), together with ``to`` -- an OpenSim ``Joint`` can only ever
    connect two frames of the *same* ``opensim.Model``, so a standalone
    component's own private container can never be the target of this; it
    has to be fused into the real container first (see "Comporre più
    modelli"/:class:`~opensim_models.components.Box`'s docstring). This
    replaces whatever joint currently connects ``child`` to its current
    parent (initially, for a fresh ``Box``/``Screen``, its own ground) --
    the component's placement before this call becomes irrelevant, since
    the new joint is defined purely by ``to``/``child_point``/
    ``parent_point``.

    Parameters
    ----------
    model : OpenSimModel
        Model containing both ``child`` and ``to``.
    child : opensim.PhysicalFrame
        The body (already in ``model``) to re-attach.
    to : opensim.PhysicalFrame
        The body (already in ``model``) ``child`` attaches to.
    child_point : ``"com"`` or tuple[float, float, float], optional
        Attachment point on ``child``, in its own local frame, in metres.
        ``"com"`` (default) uses ``child``'s centre of mass.
    parent_point : ``"com"`` or tuple[float, float, float], optional
        Attachment point on ``to``, in its own local frame, in metres.
        ``"com"`` (default) uses ``to``'s centre of mass.
    child_orientation_deg, parent_orientation_deg : tuple[float, float, float], optional
        Orientation of the new joint's frame on each side, as X-Y-Z
        body-fixed Euler degrees about that side's own local axes.
        Defaults to no tilt. For a joint type with an axis that isn't
        symmetric (e.g. a ``"slider"``, which slides along its own local
        X) this is how to point that axis anywhere other than straight
        along the parent's own X -- e.g. ``parent_orientation_deg=(0, 0,
        30)`` tilts a slider's sliding direction by 30 degrees in the
        parent's own XY plane.
    joint_type : str, optional
        One of ``"free"``, ``"pin"``, ``"ball"``, ``"slider"``, ``"weld"``
        (default) -- same joint types :func:`add_free_joint`/etc. build.
    name : str or None, optional
        Name for the new joint. Defaults to ``"{child_name}_to_{to_name}"``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Joint
        The newly created joint, wrapped.

    Raises
    ------
    ValueError
        If ``child`` is not currently connected by any joint in ``model``
        (e.g. it was never merged in, or was already removed), or if
        ``joint_type`` is not one of the five listed above.
    """
    if joint_type not in _JOINT_CLASSES:
        valid = ", ".join(sorted(_JOINT_CLASSES))
        raise ValueError(f"Unknown joint_type {joint_type!r}; expected one of {valid}")
    child = _unwrap(child)
    to = _unwrap(to)
    current_joint = _joint_owning_body(model, child)
    if current_joint is None:
        raise ValueError(
            f"{child.getName()!r} is not connected by any joint in this model "
            "-- merge it in first (e.g. `model + component`)"
        )
    # Both static body properties (mass_center), safe to read before
    # touching anything structural below.
    resolved_child_point = _resolve_attachment_point(child, child_point)
    resolved_parent_point = _resolve_attachment_point(to, parent_point)
    joint_name = name or f"{child.getName()}_to_{to.getName()}"

    def _reattach() -> Any:
        remove_joint(model, current_joint.getName())
        return _add_offset_joint(
            model,
            joint_type,
            joint_name,
            child,
            parent_frame=to,
            position=resolved_parent_point,
            orientation_deg=parent_orientation_deg,
            child_position=resolved_child_point,
            child_orientation_deg=child_orientation_deg,
            reinitialize=False,
        )

    # The removal and the new joint must land as *one* structural change:
    # reinitializing in between would mean rebuilding the system with
    # child's joint already gone (no body may be jointless, even
    # momentarily) -- same reasoning as remove_body's own docstring.
    if reinitialize:
        with model.structural_change():
            return _reattach()
    return _reattach()


# Joint type -> the add_*_joint function to connect a primitive body to
# ground with, for add_box_body/add_cylinder_body/add_sphere_body below.
