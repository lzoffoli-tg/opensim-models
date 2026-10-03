"""Add/remove a controller, contact geometry, or probe -- three distinct,
small component categories grouped in one file (none big enough alone to
justify its own module)."""

from __future__ import annotations

from typing import Any

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = [
    "add_controller",
    "remove_controller",
    "add_contact_geometry",
    "remove_contact_geometry",
    "add_probe",
    "remove_probe",
]

def add_controller(
    model: "OpenSimModel", controller: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed controller to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the controller to.
    controller : opensim.Controller or components.Controller
        The already-constructed controller, e.g. an
        ``opensim.PrescribedController``. A
        :class:`~opensim_models.components.Controller` wrapper is also
        accepted and unwrapped automatically (see :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Controller
        ``controller``, wrapped in the thin, generic
        :class:`~opensim_models.components.Controller` (``.name``/
        ``.raw``/``.set_name()`` only).
    """
    controller = _unwrap(controller)
    add_component(model, "controller", controller, reinitialize=reinitialize)
    return components.Controller(model, controller)


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

    Raises
    ------
    ValueError
        If no controller named ``name`` exists in the model.
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
    contact_geometry : opensim.ContactGeometry or components.ContactGeometry
        The already-constructed contact geometry, e.g. an
        ``opensim.ContactSphere``. A
        :class:`~opensim_models.components.ContactGeometry` wrapper is
        also accepted and unwrapped automatically (see :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.ContactGeometry
        ``contact_geometry``, wrapped in the thin, generic
        :class:`~opensim_models.components.ContactGeometry` (``.name``/
        ``.raw``/``.set_name()`` only).
    """
    contact_geometry = _unwrap(contact_geometry)
    add_component(model, "contact_geometry", contact_geometry, reinitialize=reinitialize)
    return components.ContactGeometry(model, contact_geometry)


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

    Raises
    ------
    ValueError
        If no contact geometry named ``name`` exists in the model (the
        underlying error reports the category as ``"contact_geometry"``).
    """
    remove_component(model, "contact_geometry", name, reinitialize=reinitialize)


def add_probe(model: "OpenSimModel", probe: Any, *, reinitialize: bool = False) -> Any:
    """Add an already-constructed probe to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the probe to.
    probe : opensim.Probe or components.Probe
        The already-constructed probe, e.g. an ``opensim.Umberger2010MuscleMetabolicsProbe``.
        A :class:`~opensim_models.components.Probe` wrapper is also
        accepted and unwrapped automatically (see :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Probe
        ``probe``, wrapped in the thin, generic
        :class:`~opensim_models.components.Probe` (``.name``/``.raw``/
        ``.set_name()`` only).
    """
    probe = _unwrap(probe)
    add_component(model, "probe", probe, reinitialize=reinitialize)
    return components.Probe(model, probe)


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

    Raises
    ------
    ValueError
        If no probe named ``name`` exists in the model.
    """
    remove_component(model, "probe", name, reinitialize=reinitialize)


# ---------------------------------------------------------------------------
# Rotating an existing object about an arbitrary pivot
# ---------------------------------------------------------------------------


