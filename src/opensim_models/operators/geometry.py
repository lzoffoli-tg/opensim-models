"""Pure-geometry operators that touch no ``OpenSimModel`` at all (``euclidean_distance``)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["euclidean_distance"]


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
