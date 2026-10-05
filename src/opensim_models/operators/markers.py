"""Add/remove a marker."""

from __future__ import annotations

from typing import Any

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = ["add_marker", "remove_marker"]

def add_marker(
    model: "OpenSimModel",
    marker_name: Any,
    body: Any = None,
    coordinates: tuple[float, float, float] | None = None,
    *,
    coordinates_are_global: bool = False,
    reinitialize: bool = False,
) -> components.Marker:
    """Create and add a marker, or add an existing OpenSim marker.

    For direct construction, provide the model, marker name, target body
    (a body name or :class:`~opensim_models.components.Body` wrapper),
    and 3D coordinates in metres::

        marker = add_marker(
            model,
            marker_name="tool_tip",
            body="tool_body",
            coordinates=(0.0, 0.2, 0.0),
            coordinates_are_global=False,
            reinitialize=True,
        )

    By default, ``coordinates`` are local to the body. Set
    ``coordinates_are_global=True`` to provide a point in the ground
    frame; it is converted to the body's local frame from its current
    pose. The body must belong to ``model``. ``reinitialize=True``
    rebuilds the OpenSim system and state after insertion, so the new
    marker can be queried immediately. For batches, call with
    ``reinitialize=False`` inside ``model.structural_change()`` instead.

    For backwards compatibility, an already-constructed
    ``opensim.Marker`` or ``components.Marker`` may still be passed as
    ``marker_name`` with ``body`` and ``coordinates`` omitted.

    Parameters
    ----------
    model : OpenSimModel
        Model that will own the newly created marker.
    marker_name : str, opensim.Marker or components.Marker
        Name for the new marker, or a preconstructed marker using the
        backwards-compatible form.
    body : str or components.Body, optional
        Name of the target body in ``model`` or its wrapper. Required when
        creating a marker; omitted for a preconstructed marker.
    coordinates : sequence of 3 floats, optional
        Marker point in metres. Interpreted in the target body's local
        frame by default, or in ground coordinates when
        ``coordinates_are_global=True``.
    coordinates_are_global : bool, optional
        Whether ``coordinates`` are expressed in the ground frame instead
        of the body's local frame. Defaults to ``False``.
    reinitialize : bool, optional
        When ``True``, rebuild the system immediately after insertion,
        preserving posture and velocity. Defaults to ``False``.

    Returns
    -------
    components.Marker
        The newly added marker, wrapped with its name, local ``location``,
        ``position_global`` and parent body accessors.

    Raises
    ------
    TypeError
        If direct-construction arguments are incomplete or have invalid
        types.
    ValueError
        If the body does not belong to ``model`` or the coordinates are
        malformed/non-finite.
    RuntimeError
        If ``body`` is a name that does not exist in ``model``.
    """
    if isinstance(marker_name, str):
        if body is None:
            raise TypeError("body is required when creating a marker")
        if coordinates is None:
            raise TypeError("coordinates are required when creating a marker")
        if isinstance(body, str):
            body = model.body(body)
        if not isinstance(body, components.Body):
            raise TypeError("body must be a body name or components.Body wrapper")
        if body._owner is not model:
            raise ValueError("body must belong to the model receiving the marker")
        marker = components.Marker(
            marker_name,
            body,
            coordinates,
            coordinates_are_global=coordinates_are_global,
        )
    else:
        if body is not None or coordinates is not None or coordinates_are_global:
            raise TypeError(
                "body, coordinates and coordinates_are_global are only used "
                "when creating a marker by name"
            )
        marker = marker_name
        if isinstance(marker, components.Marker) and marker._owner is not model:
            raise ValueError(
                "a body-attached Marker must be added to the model that owns its body"
            )

    marker = _unwrap(marker)
    add_component(model, "marker", marker, reinitialize=reinitialize)
    return model.marker(marker.getName())


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
