"""Geometry operators: a pure one (``euclidean_distance``) plus two that read
an object's current pose (``from_global_to_local``/``from_local_to_global``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import _unwrap
from ._spatial import _resolve_model_for

__all__ = ["euclidean_distance", "from_global_to_local", "from_local_to_global"]


def euclidean_distance(position1: Any, position2: Any) -> float:
    """Return the straight-line distance between two points, in whatever unit they're given in.

    A pure geometry operator, unlike almost everything else in
    :mod:`opensim_models.operators`: it takes no ``OpenSimModel`` and reads
    no state, so it works standalone -- ``euclidean_distance((0, 0, 0),
    (3, 4, 0))`` needs nothing else set up first. ``position1``/``position2``
    are plain ``(x, y, z)`` coordinates: the kind returned by
    :attr:`~opensim_models.components.Body.position_global`/
    :attr:`~opensim_models.components.Body.position_local` (and the
    matching properties on :class:`~opensim_models.components.Marker`,
    :class:`~opensim_models.components.Joint`,
    :class:`~opensim_models.components.ContactGeometry`), or any other
    3-tuple this package hands back as a ground-frame/local point --
    e.g. ``operators.rotate_object``/``operators.translate_object``'s own
    return value.

    Parameters
    ----------
    position1, position2 : sequence of 3 floats
        The two points to measure between, in the same unit and the same
        reference frame -- this function has no notion of "frame" at all,
        so it is the caller's responsibility that both points are already
        expressed consistently (e.g. both in the ground frame, or both in
        the same body's local frame).

    Returns
    -------
    float
        The Euclidean (straight-line) distance between ``position1`` and
        ``position2``, in the same unit as the inputs.

    Raises
    ------
    ValueError
        If ``position1`` or ``position2`` does not have exactly 3 values,
        or either contains a non-finite value.
    """
    point1 = np.asarray(position1, dtype=float)
    point2 = np.asarray(position2, dtype=float)
    if point1.shape != (3,):
        raise ValueError(
            f"position1 must have exactly 3 values (x, y, z), got shape {point1.shape}"
        )
    if point2.shape != (3,):
        raise ValueError(
            f"position2 must have exactly 3 values (x, y, z), got shape {point2.shape}"
        )
    if not np.all(np.isfinite(point1)):
        raise ValueError(f"position1 must be finite, got {tuple(position1)!r}")
    if not np.all(np.isfinite(point2)):
        raise ValueError(f"position2 must be finite, got {tuple(position2)!r}")
    return float(np.linalg.norm(point1 - point2))


def _validated_point(coordinates: Any) -> Any:
    point = np.asarray(coordinates, dtype=float)
    if point.shape != (3,):
        raise ValueError(
            f"coordinates must have exactly 3 values (x, y, z), got shape {point.shape}"
        )
    if not np.all(np.isfinite(point)):
        raise ValueError(f"coordinates must be finite, got {tuple(coordinates)!r}")
    return point


def _resolve_frame(obj: Any) -> tuple[Any, Any]:
    """Resolve ``obj`` to a ``(frame, owner)`` pair usable with
    :func:`~opensim_models.components._position_and_rotation_in_ground`.

    A :mod:`opensim_models.components` wrapper (a ``Body``, ``Box``,
    ``Screen``, ``OffsetFrame``, ...) already carries its own owning model
    internally -- same as every ``position_global``/``position_local``
    property on it -- so its ``owner`` is read directly off it rather than
    re-derived; this is also the only reliable way to find it for a
    ``Box``/``Screen`` specifically, whose private container is rebuilt
    (a fresh ``opensim.Model``) on every dimension/pose change, which
    leaves this package's owner registry pointing at a stale, no-longer
    current ``opensim.Model`` instance. A raw, unwrapped ``opensim``
    object instead has its owner looked up the same way
    :func:`~opensim_models.operators.rotate_object`/
    :func:`~opensim_models.operators.translate_object` already do for one
    (:func:`_resolve_model_for`, via the component's own ``getModel()``
    and this package's owner registry) -- no explicit ``model=`` argument
    is needed either way.

    Raises
    ------
    TypeError
        If ``obj``'s owning model cannot be resolved, or if the resolved
        object has no ground-frame orientation of its own (e.g. a
        ``Marker`` or a ``Joint``, neither of which is a frame) --
        :func:`~opensim_models.components.Marker.parents`/
        :attr:`~opensim_models.components.Joint.parent_frame`/
        :attr:`~opensim_models.components.Joint.child_frame` give the
        actual frame to pass instead.
    """
    if isinstance(obj, components._ComponentWrapper):
        raw = obj.raw
        owner = obj._owner
    else:
        raw = _unwrap(obj)
        owner = _resolve_model_for(raw)
        if owner is None:
            raise TypeError(
                f"Cannot resolve the owning OpenSimModel for {obj!r}: it "
                "must belong to a live OpenSimModel/User (pass either this "
                "package's wrapper around it, e.g. model.body(name), or "
                "the raw opensim object itself, as long as it is already "
                "attached to that model)."
            )

    if not (hasattr(raw, "getPositionInGround") and hasattr(raw, "getRotationInGround")):
        raise TypeError(
            f"{obj!r} has no ground-frame orientation of its own (no "
            "getPositionInGround/getRotationInGround) -- pass a Body, "
            "OffsetFrame, Box, Screen, or other frame-like object instead. "
            "A Marker has no orientation of its own: pass its parent frame "
            "(marker.parents[0]) instead; a Joint is not itself a frame "
            "either: pass its parent_frame/child_frame instead."
        )
    return raw, owner


def from_global_to_local(coordinates: Any, obj: Any) -> tuple[float, float, float]:
    """Express a ground-frame point in ``obj``'s own local frame.

    The exact inverse of :func:`from_local_to_global`:
    ``from_local_to_global(from_global_to_local(p, obj), obj) == p`` for any
    ground-frame point ``p`` and any ``obj`` accepted here, modulo
    floating-point error. Reuses this package's one shared
    ground-frame-transform reader
    (:func:`~opensim_models.components._position_and_rotation_in_ground`) --
    the same helper every ``position_global``/``position_local`` pair in
    :mod:`opensim_models.components` is built on -- rather than re-deriving
    ``getRotationInGround()``/``asMat33()`` by hand.

    Parameters
    ----------
    coordinates : sequence of 3 floats
        Point to transform, in metres, in the ground frame.
    obj : opensim_models.components wrapper or raw opensim object
        The object whose current local frame ``coordinates`` is expressed
        into. Accepts anything with a resolvable ground-frame position
        *and* orientation -- at minimum a
        :class:`~opensim_models.components.Body`,
        :class:`~opensim_models.components.Box`,
        :class:`~opensim_models.components.Screen`,
        :class:`~opensim_models.components.OffsetFrame`, or the matching
        raw ``opensim`` object (e.g. an ``opensim.Body`` or
        ``opensim.PhysicalOffsetFrame``), as long as it already belongs to
        a live :class:`~opensim_models.model.OpenSimModel`/
        :class:`~opensim_models.models.User`. A
        :class:`~opensim_models.components.Marker` or
        :class:`~opensim_models.components.Joint` is not itself a frame
        (see :func:`_resolve_frame`) -- pass the marker's parent frame, or
        the joint's ``parent_frame``/``child_frame``, instead.

    Returns
    -------
    tuple[float, float, float]
        ``coordinates``, re-expressed in ``obj``'s own local axes, in
        metres.

    Raises
    ------
    ValueError
        If ``coordinates`` does not have exactly 3 values, or contains a
        non-finite value.
    TypeError
        If ``obj`` cannot be resolved to a frame belonging to a live model
        -- see :func:`_resolve_frame`.
    """
    point = _validated_point(coordinates)
    frame, owner = _resolve_frame(obj)
    position, rotation = components._position_and_rotation_in_ground(owner, frame)
    local = rotation.T @ (point - position)
    return tuple(float(value) for value in local)


def from_local_to_global(coordinates: Any, obj: Any) -> tuple[float, float, float]:
    """Express a point given in ``obj``'s own local frame in the ground frame.

    The exact inverse of :func:`from_global_to_local`:
    ``from_global_to_local(from_local_to_global(p, obj), obj) == p`` for any
    local point ``p`` (in ``obj``'s own axes) and any ``obj`` accepted
    here, modulo floating-point error. Reuses this package's one shared
    ground-frame-transform reader
    (:func:`~opensim_models.components._position_and_rotation_in_ground`) --
    the same helper every ``position_global``/``position_local`` pair in
    :mod:`opensim_models.components` is built on -- rather than re-deriving
    ``getRotationInGround()``/``asMat33()`` by hand.

    Parameters
    ----------
    coordinates : sequence of 3 floats
        Point to transform, in metres, in ``obj``'s own local axes.
    obj : opensim_models.components wrapper or raw opensim object
        The object whose current local frame ``coordinates`` is expressed
        from. Accepts anything with a resolvable ground-frame position
        *and* orientation -- at minimum a
        :class:`~opensim_models.components.Body`,
        :class:`~opensim_models.components.Box`,
        :class:`~opensim_models.components.Screen`,
        :class:`~opensim_models.components.OffsetFrame`, or the matching
        raw ``opensim`` object (e.g. an ``opensim.Body`` or
        ``opensim.PhysicalOffsetFrame``), as long as it already belongs to
        a live :class:`~opensim_models.model.OpenSimModel`/
        :class:`~opensim_models.models.User`. A
        :class:`~opensim_models.components.Marker` or
        :class:`~opensim_models.components.Joint` is not itself a frame
        (see :func:`_resolve_frame`) -- pass the marker's parent frame, or
        the joint's ``parent_frame``/``child_frame``, instead.

    Returns
    -------
    tuple[float, float, float]
        ``coordinates``, re-expressed in the ground frame, in metres.

    Raises
    ------
    ValueError
        If ``coordinates`` does not have exactly 3 values, or contains a
        non-finite value.
    TypeError
        If ``obj`` cannot be resolved to a frame belonging to a live model
        -- see :func:`_resolve_frame`.
    """
    point = _validated_point(coordinates)
    frame, owner = _resolve_frame(obj)
    position, rotation = components._position_and_rotation_in_ground(owner, frame)
    global_point = position + rotation @ point
    return tuple(float(value) for value in global_point)
