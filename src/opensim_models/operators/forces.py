"""Add/remove a force (``add_force``/``remove_force``) and build a muscle
from its origin/insertion attachments (``add_muscle``/``remove_muscle``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = ["add_force", "remove_force", "add_muscle", "remove_muscle"]

def add_force(model: "OpenSimModel", force: Any, *, reinitialize: bool = False) -> Any:
    """Add an already-constructed force/actuator to the model.

    Muscles are a ``Force`` subtype in OpenSim (there is no separate,
    directly-addable "muscle set") -- see :func:`add_muscle` for a named
    alias when the component is specifically a muscle.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the force to.
    force : opensim.Force or components.Force
        The already-constructed force/actuator, e.g. a
        ``opensim.Millard2012EquilibriumMuscle`` or
        ``opensim.CoordinateActuator``. A
        :class:`~opensim_models.components.Force` (or other component)
        wrapper is also accepted and unwrapped automatically (see
        :func:`_unwrap`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Force
        ``force``, wrapped in the thin, generic
        :class:`~opensim_models.components.Force` (``.name``/``.raw``/
        ``.set_name()`` only). This is true even if ``force`` is actually
        a muscle: use :func:`add_muscle` instead of this function to get
        back a :class:`~opensim_models.components.Muscle`, with its extra
        ``max_isometric_force``/``optimal_fiber_length``/
        ``tendon_slack_length``/``pennation_angle`` properties.
    """
    force = _unwrap(force)
    add_component(model, "force", force, reinitialize=reinitialize)
    return components.Force(model, force)


def remove_force(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a force/actuator (including a muscle) from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the force from.
    name : str
        Name of the force to remove.
    reinitialize : bool, optional
        See :func:`add_component`.

    Raises
    ------
    ValueError
        If no force (muscle or otherwise) named ``name`` exists in the
        model.
    """
    remove_component(model, "force", name, reinitialize=reinitialize)


def add_muscle(
    model: "OpenSimModel",
    name: str,
    origin_component: Any,
    origin_position: tuple[float, float, float],
    insertion_component: Any,
    insertion_position: tuple[float, float, float],
    *,
    max_isometric_force: float = 1000.0,
    optimal_fiber_length: float = 0.1,
    tendon_slack_length: float = 0.2,
    pennation_angle_deg: float = 0.0,
    via_points: Any = (),
    muscle_class: str = "Millard2012EquilibriumMuscle",
    reinitialize: bool = False,
) -> Any:
    """Build a muscle from its origin/insertion attachments and add it to the model.

    A ``Muscle`` is a ``Force`` subtype in OpenSim and is stored in (and
    removed from) the same ``ForceSet`` -- see :func:`add_force` for
    adding any other kind of force/actuator, or any muscle type whose
    constructor doesn't match ``muscle_class``'s assumed
    ``(name, max_isometric_force, optimal_fiber_length,
    tendon_slack_length, pennation_angle)`` shape.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the muscle to.
    name : str
        Name for the new muscle.
    origin_component, insertion_component : opensim.PhysicalFrame
        The body (or other physical frame) each end of the muscle's path
        attaches to.
    origin_position, insertion_position : tuple[float, float, float]
        Attachment point, in metres, in the local frame of
        ``origin_component``/``insertion_component`` respectively.
    max_isometric_force : float, optional
        Maximum isometric force, in newtons. Defaults to ``1000.0``.
    optimal_fiber_length : float, optional
        Optimal fiber length, in metres. Defaults to ``0.1``.
    tendon_slack_length : float, optional
        Tendon slack length, in metres. Defaults to ``0.2``.
    pennation_angle_deg : float, optional
        Pennation angle at optimal fiber length, in degrees (converted to
        radians for the raw constructor, which takes radians). Defaults
        to ``0.0``.
    via_points : sequence of (component, (x, y, z)), optional
        Extra path points inserted, in order, between the origin and
        insertion attachments -- e.g. to wrap a muscle's path around a
        joint. Empty by default (a straight origin-to-insertion path).
    muscle_class : str, optional
        Name of the ``opensim`` muscle class to instantiate (looked up on
        ``model.opensim``, same resolution as the joint type -> class
        lookup :func:`add_free_joint`/:func:`add_pin_joint`/etc. use).
        Defaults to ``"Millard2012EquilibriumMuscle"``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Muscle
        The newly created muscle, wrapped in
        :class:`~opensim_models.components.Muscle`, which exposes
        ``.max_isometric_force``/``.optimal_fiber_length``/
        ``.tendon_slack_length``/``.pennation_angle`` (each with a
        matching ``set_*()``) on top of the common ``.name``/``.raw``.

    Raises
    ------
    AttributeError
        If ``muscle_class`` does not name an attribute of
        ``model.opensim`` (e.g. a typo, or a muscle type not present in
        this build of OpenSim).
    """
    origin_component = _unwrap(origin_component)
    insertion_component = _unwrap(insertion_component)
    muscle_cls = getattr(model.opensim, muscle_class)
    muscle = muscle_cls(
        name,
        float(max_isometric_force),
        float(optimal_fiber_length),
        float(tendon_slack_length),
        float(np.deg2rad(pennation_angle_deg)),
    )
    muscle.addNewPathPoint(f"{name}-P1", origin_component, model.opensim.Vec3(*origin_position))
    index = 2
    for component, position in via_points:
        muscle.addNewPathPoint(f"{name}-P{index}", _unwrap(component), model.opensim.Vec3(*position))
        index += 1
    muscle.addNewPathPoint(f"{name}-P{index}", insertion_component, model.opensim.Vec3(*insertion_position))
    add_component(model, "force", muscle, reinitialize=reinitialize)
    return components.Muscle(model, muscle)


def remove_muscle(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a muscle from the model. A named alias of :func:`remove_force`.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the muscle from.
    name : str
        Name of the muscle to remove.
    reinitialize : bool, optional
        See :func:`add_component`.

    Raises
    ------
    ValueError
        If no force named ``name`` exists in the model (note the error
        message says "force", not "muscle" -- see :func:`remove_force`,
        which this delegates to: muscles live in the same ``ForceSet``).
    """
    remove_force(model, name, reinitialize=reinitialize)


