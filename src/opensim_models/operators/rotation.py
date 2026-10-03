"""Rotate a component or a whole model about an axis (``rotate_object``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model import OpenSimModel
from ._shared import _unwrap
from ._spatial import (
    _resolve_model,
    _resolve_ground_position,
    _rotation_matrix_in_ground,
    _corresponding_component,
    _ground_attached_offset_frames,
)

__all__ = ["rotate_object"]

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


