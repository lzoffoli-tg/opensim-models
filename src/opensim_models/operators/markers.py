"""Add/remove a marker."""

from __future__ import annotations

from typing import Any

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = ["add_marker", "remove_marker"]

def add_marker(
    model: "OpenSimModel", marker: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed marker to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the marker to.
    marker : opensim.Marker or components.Marker
        The already-constructed marker, e.g. ``opensim.Marker(name,
        parent_frame, opensim.Vec3(x, y, z))``. A
        :class:`~opensim_models.components.Marker` wrapper is also
        accepted and unwrapped automatically (see :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Marker
        ``marker``, wrapped in :class:`~opensim_models.components.Marker`,
        which exposes ``.location``/``.set_location()`` (the marker's
        ``(x, y, z)`` offset within its parent frame, in metres) on top of
        the common ``.name``/``.raw``/``.set_name()``.
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

    Raises
    ------
    ValueError
        If no marker named ``name`` exists in the model.
    """
    remove_component(model, "marker", name, reinitialize=reinitialize)


