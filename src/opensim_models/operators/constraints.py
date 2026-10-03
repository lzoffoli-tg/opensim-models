"""Add/remove a constraint, plus the named constraint-type convenience
functions (``add_weld_constraint``, ``add_point_constraint``,
``add_coordinate_coupler_constraint``, ``add_point_on_plane_constraint``)."""

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
    "add_point_on_plane_constraint",
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
        ``constraint``, wrapped in the most specific wrapper available for
        its concrete type (:class:`~opensim_models.components.WeldConstraint`,
        :class:`~opensim_models.components.PointConstraint`,
        :class:`~opensim_models.components.ConstantDistanceConstraint`), or
        the thin, generic :class:`~opensim_models.components.Constraint`
        (``.name``/``.raw``/``.set_name()`` only) for anything else -- see
        :func:`~opensim_models.components._wrap_constraint`.
    """
    constraint = _unwrap(constraint)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components._wrap_constraint(model, constraint)


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
    components.WeldConstraint
        The newly created constraint, wrapped in
        :class:`~opensim_models.components.WeldConstraint`, which exposes
        each side's attachment frame (``frame1``/``frame2``, each an
        :class:`~opensim_models.components.OffsetFrame`) and the matching
        ``point1_global``/``point1_local``/``point2_global``/``point2_local``.
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
    return components._wrap_constraint(model, constraint)


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
    components.PointConstraint
        The newly created constraint, wrapped in
        :class:`~opensim_models.components.PointConstraint`, which exposes
        ``point1_global``/``point1_local``/``point2_global``/``point2_local``.
    """
    body1 = _unwrap(body1)
    body2 = _unwrap(body2)
    constraint = model.opensim.PointConstraint(
        body1, model.opensim.Vec3(*position1), body2, model.opensim.Vec3(*position2)
    )
    constraint.setName(name)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components._wrap_constraint(model, constraint)


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
    return components._wrap_constraint(model, constraint)


def _local_to_ground(frame: Any, state: Any, local_point: Any) -> Any:
    """Transform a body-fixed point (numpy array, metres) into the ground frame."""
    position = np.array([frame.getPositionInGround(state).get(i) for i in range(3)])
    matrix = frame.getRotationInGround(state).asMat33()
    rotation = np.array([[matrix.get(i, j) for j in range(3)] for i in range(3)])
    return position + rotation @ local_point


def add_point_on_plane_constraint(
    model: "OpenSimModel",
    name: str,
    body: Any,
    point: tuple[float, float, float],
    plane_body: Any,
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
    *,
    anchor_distance: float = 50.0,
    reinitialize: bool = False,
) -> Any:
    """Approximately constrain a point on ``body`` to a plane fixed on ``plane_body``.

    Removes (approximately -- see "Accuracy" below) the single relative
    degree of freedom along the plane's normal, leaving ``body`` free to
    slide in the other two, in-plane directions -- e.g. a pelvis/torso/
    humerus landmark resting on, and free to slide along, a backrest or
    pad surface, without welding the two bodies together or re-routing
    either one's own anatomical joint.

    **Why this is an approximation, and why it exists as a new function
    here at all.** OpenSim's own constraint catalogue -- confirmed by
    enumerating every ``*Constraint`` class in both ``opensim`` and
    ``opensim.simbody`` in this installation -- is exactly
    ``WeldConstraint`` (6 dof removed), ``PointConstraint`` (3 dof),
    ``PointOnLineConstraint`` (2 dof), ``ConstantDistanceConstraint`` (1
    dof, but to a *sphere*, not a plane) and ``CoordinateCouplerConstraint``
    (couples generalized coordinates to each other, not spatial points).
    There is no built-in, single-equation "point stays on a plane" OpenSim
    constraint, and Simbody's own lower-level ``Constraint::PointInPlane``
    (which *would* be exact) is not exposed by this project's bound
    ``opensim``/``opensim.simbody`` modules at all -- not reachable even by
    dropping to :meth:`~opensim_models.model.OpenSimModel.model`'s
    ``updMatterSubsystem()`` without writing and compiling a new SWIG
    binding, well outside this package's scope. This function instead
    builds an ``opensim.ConstantDistanceConstraint`` whose anchor point is
    placed ``anchor_distance`` metres *behind* ``plane_point`` along the
    negative ``plane_normal`` direction: for any point at the correct
    signed distance from the plane, the sphere of radius ``anchor_distance
    + signed_distance`` centred on that anchor is tangent to the plane at
    ``plane_point`` -- a classic long-rod approximation of a planar guide,
    exact only at zero in-plane (tangential) displacement.

    Accuracy: the constraint's radius is computed (see implementation
    note below) so it is satisfied *exactly*, with zero error, at
    ``body``'s and ``plane_body``'s current pose when this function is
    called -- regardless of any tangential offset ``point`` already has
    from ``plane_point`` at that moment (an earlier, buggy version of this
    function computed the radius from ``anchor_distance`` plus only the
    normal-direction component, which is only correct when that tangential
    offset is exactly zero at construction time; otherwise it reproduces
    the native-crash gotcha below, confirmed by triggering it). Error only
    appears for *further* tangential displacement away from that
    construction-time configuration: for a point subsequently displaced by
    an additional tangential distance ``t``, the sphere pulls it off the
    true (parallel) plane through its starting point by approximately
    ``t**2 / (2 * anchor_distance)`` (a second-order effect, derived from
    the sphere/plane geometry -- not a Simbody approximation, an exact
    property of this constraint choice). E.g. with the default
    ``anchor_distance=50.0`` m, a further 0.5 m slide along the surface
    pulls the point off-plane by roughly 0.5**2 / 100 = 2.5 mm. Increase
    ``anchor_distance`` for a tighter approximation over a larger sliding
    range, at the cost of a larger (but still well-conditioned in double
    precision) constraint radius.

    **A native-crash gotcha this function exists specifically to avoid**:
    confirmed directly in this environment (minimal repro, both for
    ``PointConstraint`` and for ``ConstantDistanceConstraint``) that
    building one of these distance-type constraints between two bodies
    that are each an independent descendant of ground (e.g. one hanging
    off a ``SliderJoint``, the other off a ``FreeJoint`` -- exactly the
    backrest/pelvis shape) segfaults ``Model.initSystem()`` natively (not a
    catchable Python exception) *if the model's current default posture
    does not already satisfy the constraint*; the identical construction
    with the constraint already exactly satisfied at the default posture
    initializes without incident. This is a narrower, more precise
    trigger than "two non-ground bodies" as such (see
    :func:`add_point_constraint`'s own warning) -- it is specifically about
    the assembler's handling of an *unsatisfied* distance constraint at
    initialization, not about non-ground bodies as a category. This
    function sidesteps it entirely by reading ``model``'s *current* posture
    (``model.state``, via ``realizePosition`` -- call this only once the
    model is already posed the way you want the constraint's reference
    configuration to be) and computing the one ``distance`` value that
    makes the new constraint exactly satisfied there, rather than ever
    asking the caller for a distance that could mismatch the model's
    current geometry.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the constraint to. Must already be in the posture
        this constraint should take as its reference configuration (see
        above).
    name : str
        Name for the new constraint.
    body : opensim.PhysicalFrame
        The body whose point is constrained to the plane (e.g. a pelvis,
        torso or humerus).
    point : tuple[float, float, float]
        The constrained point, in metres, in ``body``'s own local frame.
    plane_body : opensim.PhysicalFrame
        The body the contact plane is fixed to (e.g. a backrest or pad).
    plane_point : tuple[float, float, float]
        A point on the plane, in metres, in ``plane_body``'s own local
        frame (e.g. the centre of the pad's contact face).
    plane_normal : tuple[float, float, float]
        The plane's normal direction, in ``plane_body``'s own local frame.
        Need not be a unit vector (normalized internally); the sign only
        matters together with ``anchor_distance`` (see ``Raises`` below).
    anchor_distance : float, optional
        Distance, in metres, from ``plane_point`` to the sphere's anchor,
        measured along the *negative* ``plane_normal`` direction. Defaults
        to ``50.0`` -- see "Accuracy" above for the tradeoff involved in
        changing it.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.ConstantDistanceConstraint
        The newly created ``ConstantDistanceConstraint``, wrapped in
        :class:`~opensim_models.components.ConstantDistanceConstraint`,
        which exposes ``point1_global``/``point1_local`` (``body``/
        ``point``) and ``point2_global``/``point2_local`` (``plane_body``/
        the computed ``anchor_point`` -- *not* ``plane_point`` itself, see
        that class's own docstring) plus ``distance``.

    Raises
    ------
    ValueError
        If ``plane_normal`` is the zero vector, ``anchor_distance`` is not
        finite or not strictly positive, ``body``'s current point
        coincides with the computed anchor point (zero/undefined radius),
        or ``body``'s current point is at a signed distance from the plane
        beyond the anchor on the far side (``signed_distance <=
        -anchor_distance``) -- flip the sign of ``plane_normal`` in that
        last case.
    """
    body = _unwrap(body)
    plane_body = _unwrap(plane_body)
    point = np.asarray(point, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    normal_length = np.linalg.norm(normal)
    if normal_length == 0.0:
        raise ValueError("plane_normal must not be the zero vector")
    normal = normal / normal_length
    if not np.isfinite(anchor_distance) or anchor_distance <= 0:
        raise ValueError(
            f"anchor_distance must be finite and strictly positive, got {anchor_distance!r}"
        )

    # Il raggio della sfera deve corrispondere ESATTAMENTE alla distanza
    # 3D attuale fra `point` e l'ancora nella posa corrente del modello
    # (vedi la docstring sopra: altrimenti initSystem() va in segmentation
    # fault, non un'eccezione catturabile) -- quindi lo leggo dal vivo con
    # realizePosition invece di chiederlo al chiamante o ricavarlo solo
    # dalla componente lungo la normale: `point` puo' gia' avere un offset
    # TANGENZIALE non nullo rispetto a `plane_point` nella posa corrente
    # (il caso normale: i due corpi sono gia' stati posizionati/posati
    # prima di chiamare questa funzione), e la distanza 3D dall'ancora
    # dipende da ENTRAMBE le componenti (normale e tangenziale), non solo
    # da quella normale -- usare "anchor_distance + componente lungo la
    # normale" come raggio (un errore fatto e corretto durante lo sviluppo
    # di questa funzione) e' sbagliato ogniqualvolta l'offset tangenziale
    # all'istante della costruzione non e' zero, e riproduce esattamente
    # il segfault documentato sopra (raggio leggermente disallineato
    # rispetto alla geometria reale).
    model.model.realizePosition(model.state)
    point_ground = _local_to_ground(body, model.state, point)
    plane_point_ground = _local_to_ground(plane_body, model.state, plane_point)
    plane_rotation_matrix = plane_body.getRotationInGround(model.state).asMat33()
    plane_rotation = np.array(
        [[plane_rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
    )
    normal_ground = plane_rotation @ normal
    anchor_ground = plane_point_ground - anchor_distance * normal_ground

    radius = float(np.linalg.norm(point_ground - anchor_ground))
    if radius <= 1e-9:
        raise ValueError(
            f"required constraint radius is {radius!r}, not usably positive -- "
            "body's point coincides with the anchor point; check point/plane_point/"
            "plane_normal/anchor_distance"
        )
    signed_distance = float(np.dot(point_ground - plane_point_ground, normal_ground))
    if signed_distance <= -anchor_distance:
        raise ValueError(
            f"body's point is at signed distance {signed_distance!r} from the "
            f"plane, beyond the anchor (anchor_distance={anchor_distance!r}) on "
            "the far side -- try flipping the sign of plane_normal"
        )
    anchor_point = plane_point - anchor_distance * normal

    constraint = model.opensim.ConstantDistanceConstraint(
        body,
        model.opensim.Vec3(*point),
        plane_body,
        model.opensim.Vec3(*anchor_point),
        radius,
    )
    constraint.setName(name)
    add_component(model, "constraint", constraint, reinitialize=reinitialize)
    return components._wrap_constraint(model, constraint)


