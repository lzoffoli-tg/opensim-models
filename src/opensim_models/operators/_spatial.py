"""Helpers shared by both :mod:`.rotation` and :mod:`.translation`: resolving
which :class:`~opensim_models.model.OpenSimModel` a component/coordinate
belongs to, finding its ground-frame position, and finding the matching
component in another (e.g. a cloned) model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import _find_owner, _iter_set
from ..model import OpenSimModel
from ._shared import _unwrap

__all__: list[str] = []


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


