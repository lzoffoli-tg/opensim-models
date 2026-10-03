"""Add/remove a constraint, plus the named constraint-type convenience
functions (``add_weld_constraint``, ``add_point_constraint``,
``add_coordinate_coupler_constraint``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import add_component, remove_component, _unwrap

__all__ = [
    "add_constraint",
    "remove_constraint",
    "add_weld_constraint",
    "add_point_constraint",
    "add_coordinate_coupler_constraint",
]

def add_constraint(
    model: "OpenSimModel", constraint: Any, *, reinitialize: bool = False
) -> Any:
    """Add an already-constructed constraint to the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    constraint : opensim.Constraint or components.Constraint
        The already-constructed constraint, e.g. an
        ``opensim.CoordinateCouplerConstraint``. A
        :class:`~opensim_models.components.Constraint` wrapper is also
        accepted and unwrapped automatically (see :func:`_unwrap`). For
        the common constraint types, see the named, position-based
        convenience functions instead: :func:`add_weld_constraint`,
        :func:`add_point_constraint`, :func:`add_coordinate_coupler_constraint`.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        ``constraint``, wrapped in the thin, generic
        :class:`~opensim_models.components.Constraint` (``.name``/
        ``.raw``/``.set_name()`` only).
    """
    constraint = _unwrap(constraint)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def remove_constraint(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a constraint from the model.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the constraint from.
    name : str
        Name of the constraint to remove.
    reinitialize : bool, optional
        See :func:`add_component`.

    Raises
    ------
    ValueError
        If no constraint named ``name`` exists in the model.
    """
    remove_component(model, "constraint", name, reinitialize=reinitialize)


def add_weld_constraint(
    model: "OpenSimModel",
    name: str,
    body1: Any,
    body2: Any,
    *,
    position1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation1_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position2: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation2_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reinitialize: bool = False,
) -> Any:
    """Rigidly weld ``body1`` and ``body2`` together (removes all 6 relative dof).

    Builds an ``opensim.WeldConstraint`` from two attachment frames, one
    per body -- unlike a ``WeldJoint`` (see :func:`add_weld_joint`), this
    does not change the kinematic tree: both bodies keep their own
    joints, and the constraint just forces their two attachment frames to
    coincide.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    body1, body2 : opensim.PhysicalFrame
        The two bodies (or other physical frames) to weld together.
    position1, orientation1_deg : tuple[float, float, float], optional
        Attachment point/orientation (X-Y-Z body-fixed Euler degrees) on
        ``body1``'s own frame. Defaults to no offset.
    position2, orientation2_deg : tuple[float, float, float], optional
        Same as ``position1``/``orientation1_deg``, but for ``body2``.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    body1 = _unwrap(body1)
    body2 = _unwrap(body2)
    constraint = model.opensim.WeldConstraint(
        name,
        body1,
        model.opensim.Vec3(*position1),
        model.opensim.Vec3(*np.deg2rad(orientation1_deg)),
        body2,
        model.opensim.Vec3(*position2),
        model.opensim.Vec3(*np.deg2rad(orientation2_deg)),
    )
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def add_point_constraint(
    model: "OpenSimModel",
    name: str,
    body1: Any,
    position1: tuple[float, float, float],
    body2: Any,
    position2: tuple[float, float, float],
    *,
    reinitialize: bool = False,
) -> Any:
    """Constrain a point fixed on ``body1`` to coincide with a point fixed on ``body2``.

    Builds an ``opensim.PointConstraint`` (removes 3 relative translational
    dof, orientation stays free) -- use :func:`add_weld_constraint` instead
    if orientation should be locked too.

    Confirmed directly (OpenSim 4.6): a ``PointConstraint`` between two
    *non-ground* bodies crashes the process natively during
    ``initSystem()`` (not a catchable Python exception) -- regardless of
    whether the two bodies are independent branches or parent/child in the
    same chain. The identical construction with ``model.model.getGround()``
    as one of the two bodies works correctly. Until this is resolved
    upstream, treat ``add_point_constraint`` as ground-to-body only; for
    body-to-body, use :func:`add_weld_constraint` instead (confirmed not to
    have this issue), accepting the extra 3 locked rotational dof.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    body1, body2 : opensim.PhysicalFrame
        The two bodies (or other physical frames) whose points are
        constrained to coincide.
    position1 : tuple[float, float, float]
        Point, in metres, in ``body1``'s own local frame.
    position2 : tuple[float, float, float]
        Point, in metres, in ``body2``'s own local frame.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    body1 = _unwrap(body1)
    body2 = _unwrap(body2)
    constraint = model.opensim.PointConstraint(
        body1, model.opensim.Vec3(*position1), body2, model.opensim.Vec3(*position2)
    )
    constraint.setName(name)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


def add_coordinate_coupler_constraint(
    model: "OpenSimModel",
    name: str,
    independent_coordinates: Any,
    dependent_coordinate: Any,
    function: Any,
    *,
    reinitialize: bool = False,
) -> Any:
    """Couple ``dependent_coordinate``'s value to ``independent_coordinates`` via ``function``.

    Builds an ``opensim.CoordinateCouplerConstraint``: whenever any
    independent coordinate changes, OpenSim re-solves ``function`` of
    their values and assigns the result to ``dependent_coordinate`` (e.g.
    a patella coupled to knee flexion).

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to.
    name : str
        Name for the new constraint.
    independent_coordinates : str, components.Coordinate, or a sequence of either
        The coordinate(s) ``function`` is evaluated on. A single
        coordinate (name or wrapper) is also accepted directly, not just
        a sequence of one.
    dependent_coordinate : str or components.Coordinate
        The coordinate whose value ``function``'s result is assigned to.
    function : opensim.Function
        The already-built coupling function, e.g.
        ``opensim.LinearFunction(slope, intercept)``. OpenSim has many
        ``Function`` subtypes (linear, spline, constant, ...); building
        one is left to the caller, same as :func:`add_force` leaves
        building the force/actuator itself to the caller for anything
        beyond a named muscle (:func:`add_muscle`).
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Constraint
        The newly created constraint, wrapped.
    """
    if isinstance(independent_coordinates, (str, components.Coordinate)):
        independent_coordinates = [independent_coordinates]
    names = model.opensim.ArrayStr()
    for coordinate in independent_coordinates:
        names.append(
            coordinate.name if isinstance(coordinate, components.Coordinate) else coordinate
        )
    dependent_name = (
        dependent_coordinate.name
        if isinstance(dependent_coordinate, components.Coordinate)
        else dependent_coordinate
    )
    constraint = model.opensim.CoordinateCouplerConstraint()
    constraint.setName(name)
    constraint.setIndependentCoordinateNames(names)
    constraint.setDependentCoordinateName(dependent_name)
    constraint.setFunction(function)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components.Constraint(model, constraint)


