"""Translate a component or a whole model by a displacement (``translate_object``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ..model import OpenSimModel
from ._shared import _unwrap
from ._spatial import (
    _resolve_model_for,
    _resolve_ground_position,
    _rotation_matrix_in_ground,
    _corresponding_component,
    _ground_attached_offset_frames,
)

__all__ = ["translate_object"]

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
    by its orientation (``PhysicalOffsetFrame``'s own ``translation`` is
    relative to the parent frame's own axes). Reads/writes ``obj``'s own
    translation through :class:`~opensim_models.components.OffsetFrame`
    (its ``translation``/``set_translation``) rather than raw SWIG calls;
    the parent side still goes through the generic, type-agnostic
    :func:`_resolve_ground_position`/:func:`_rotation_matrix_in_ground`
    helpers, since ``parent`` can be any kind of frame, not necessarily a
    ``PhysicalOffsetFrame`` itself.

    Does not call :meth:`~opensim_models.model.OpenSimModel.reinitialize`:
    callers moving several frames together (see
    :func:`_translate_whole_model`) do that once, after the whole batch.
    """
    opensim = model.opensim
    offset_frame = components.OffsetFrame(model, obj)
    parent = opensim.PhysicalFrame.safeDownCast(obj.getParentFrame())
    parent_position = np.array(_resolve_ground_position(parent, model), dtype=float)
    parent_rotation = _rotation_matrix_in_ground(model, parent)

    local_delta = parent_rotation.T @ direction_vector
    new_local_translation = np.array(offset_frame.translation, dtype=float) + local_delta
    offset_frame.set_translation(tuple(new_local_translation))

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
