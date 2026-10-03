"""Add a sliding point-contact force between a point on one body and a
flat plane fixed on another -- a real, physical (force-based) alternative
to :func:`~opensim_models.operators.constraints.add_point_on_plane_constraint`
for the same "point resting on, and free to slide along, a flat surface"
scenario (e.g. an acromion landmark resting on a shoulder pad's face, or a
pelvis/torso landmark resting on a backrest)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import add_component, _unwrap
from .constraints import _local_to_ground

__all__ = ["add_sliding_point_contact"]


def add_sliding_point_contact(
    model: "OpenSimModel",
    name: str,
    body: Any,
    point: tuple[float, float, float],
    plane_body: Any,
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
    *,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.2,
    settle_velocity: float = 0.01,
    reinitialize: bool = False,
) -> Any:
    """Add a real (force-based) sliding contact between a point on ``body`` and a plane on ``plane_body``.

    Builds an ``opensim.ExponentialContactForce``: a normal force that
    repels ``point`` from the plane (elastic, exponential in penetration
    depth, with damping) plus an in-plane friction force (static below
    ``settle_velocity``, kinetic above it, both with a smooth blend) --
    exactly the "point/marker resting on, and sliding along, a flat
    surface" physics the colleague's suggestion of ``ContactSphere``/
    ``ContactHalfSpace`` + ``HuntCrossleyForce``/``ElasticFoundationForce``
    was reaching for, but built on a different, newer (OpenSim 4.5+)
    OpenSim class that turned out to be both simpler and the only one that
    actually works in this installation -- see "Why ``ExponentialContactForce``,
    not ``HuntCrossleyForce``" below.

    Unlike :func:`add_point_on_plane_constraint` (a kinematic, holonomic
    approximation with no notion of force, friction, or settling -- see its
    own docstring), this is a genuine force: ``body`` is free to penetrate
    the plane slightly (the elastic restoring force grows exponentially
    with penetration depth, by design, so in practice the penetration stays
    small for a reasonable mass/gravity), free to bounce, and free to
    actually settle onto the plane under gravity over time (see
    :meth:`~opensim_models.model.OpenSimModel.settle_under_gravity`) instead
    of being pinned there from the first instant. It also does not require
    ``body``'s point to already be touching the plane when this function is
    called (a hard requirement of ``add_point_on_plane_constraint``,
    documented there): any starting position works, including one well
    above the plane, since the normal force simply drops to a negligible
    value ``d1*exp(-d2*(depth - d0))`` for a point far outside the surface
    (default shape constants tuned so this falls below 0.01 N by about
    1 cm away -- see ``opensim.ExponentialContactForce.setExponentialShapeParameters``
    for changing them, not exposed as a parameter of this simple wrapper).

    **Why `ExponentialContactForce`, not `HuntCrossleyForce`/`ElasticFoundationForce`.**
    Enumerating every ``Contact*``/``*Force`` class in this installation's
    bound ``opensim`` module (OpenSim 4.6) turns up the catalogue the
    colleague's suggestion assumed: ``ContactSphere``, ``ContactHalfSpace``,
    ``ContactMesh``, paired with ``HuntCrossleyForce`` or
    ``ElasticFoundationForce``. Both of the latter two delegate their
    per-geometry-pair stiffness/dissipation/friction to a nested
    ``ContactParameters`` object (``force.getContactParametersSet()``,
    ``force.addContactParameters(params)``) -- but that nested class is not
    exposed anywhere in this project's bound ``opensim`` module under any
    name tried (confirmed directly: no ``HuntCrossleyForce_ContactParameters``,
    no top-level alias, ``getContactParametersSet()`` returns an opaque
    ``SwigPyObject`` with no usable methods), so it cannot be constructed
    from Python at all in this installation. Both classes additionally
    expose a *flattened* set of setters directly on the force itself
    (``setStiffness``/``setDissipation``/``setStaticFriction``/...,
    ``addGeometry(name)``) that look like a usable substitute -- they accept
    values without error -- but confirmed directly (minimal repro, both
    force types, both paired with a real ``ContactSphere``/``ContactHalfSpace``):
    this flattened path is a trap. ``initSystem()`` succeeds and
    ``realizeDynamics()`` succeeds, but ``model.realizeAcceleration(state)``
    (and therefore any forward-dynamics integration, i.e. the entire point
    of a contact force) segfaults the process natively, not a catchable
    exception -- because the flattened setters never actually populate a
    valid ``ContactParameters`` entry the force's own acceleration-stage
    computation can find. ``ExponentialContactForce`` (added in OpenSim 4.5)
    sidesteps this whole class hierarchy: it needs no ``ContactGeometry``
    objects and no ``ContactParameters`` at all, just a body, a local point
    on it (the "station"), and a plane transform -- confirmed directly to
    build, ``initSystem()``, ``realizeDynamics()``, and forward-integrate
    through a full settle (see
    :meth:`~opensim_models.model.OpenSimModel.settle_under_gravity`) without
    incident, on a non-``FreeJoint`` body (see that method's docstring for
    why ``FreeJoint`` itself is excluded).

    **A real limitation this approach does carry, as a direct consequence
    of `ExponentialContactForce`'s own design**: the contact plane's pose is
    a plain ``Transform`` *property* of the force (baked in once, here, from
    ``plane_body``'s pose in ``model``'s *current* state -- same
    read-the-live-posture approach as :func:`add_point_on_plane_constraint`),
    not a live attachment to ``plane_body`` through a socket/frame
    connection. If ``plane_body`` is itself not fixed to ground (e.g. it
    rides on a ``SliderJoint``, like an adjustable pad) and moves *after*
    this function is called, the contact plane does **not** follow it --
    it stays at ``plane_body``'s pose at construction time. This is fine
    for a plane that is itself fixed (e.g. welded to ground, or simply not
    moved again before settling), which covers the common case (a backrest/
    pad whose position is being evaluated at one fixed setting at a time);
    it is not a substitute for a plane that must track a moving body
    through a whole simulation. Re-add (remove and re-create) the force if
    ``plane_body`` needs to move and the contact must track it.

    **A second real limitation, also a direct consequence of
    `ExponentialContactForce`'s own design, confirmed empirically**: the
    contact plane is a true mathematical plane, infinite in extent -- *not*
    clipped to ``plane_body``'s actual visible footprint (e.g. a ``Box``'s
    finite face). The normal force depends only on ``point``'s signed
    perpendicular distance to that infinite plane, so ``body`` is repelled
    from it everywhere, including well outside the area a real pad/backrest
    would actually cover -- confirmed directly by settling a point with no
    sideways degree of freedom at all onto a small, offset, tilted box:
    it still landed exactly on the plane passing through the box's face
    (at the height/tilt that infinite plane implies at the point's fixed
    (x, z)), even though that (x, z) location is nowhere near the box
    itself. If contact must be limited to a bounded area (e.g. "on the pad,
    not past its edge"), that check is left to the caller (e.g. compare
    ``point``'s in-plane position against ``plane_body``'s known
    half-width/half-depth after settling) -- ``ContactMesh`` (not wrapped
    here, see the module docstring) would be the natural OpenSim class for
    a truly bounded surface, at the cost of the mesh-file complexity this
    function exists to avoid.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the force to. Must already be in the posture this
        force's contact plane should take as its reference configuration
        (``plane_body``'s current pose in ``model.state``, read via
        ``realizePosition`` -- see "A real limitation" above).
    name : str
        Name for the new force.
    body : opensim.PhysicalFrame or components wrapper
        The body whose point is in contact with the plane (e.g. a torso,
        pelvis or humerus). Must not be attached to ground by a plain
        ``opensim.FreeJoint`` -- see
        :meth:`~opensim_models.model.OpenSimModel.settle_under_gravity` for
        why, and for safe alternatives (``BallJoint``/``SliderJoint``,
        a ``CustomJoint`` built like ``User``'s own 6-dof root joint, ...).
    point : tuple[float, float, float]
        The contact point, in metres, in ``body``'s own local frame.
    plane_body : opensim.PhysicalFrame or components wrapper
        The body the contact plane is fixed to (e.g. a backrest or pad).
    plane_point : tuple[float, float, float]
        A point on the plane, in metres, in ``plane_body``'s own local
        frame (e.g. the centre of the pad's contact face).
    plane_normal : tuple[float, float, float]
        The plane's normal direction, in ``plane_body``'s own local frame
        (the direction ``point`` is repelled towards). Need not be a unit
        vector (normalized internally).
    static_friction : float, optional
        Coefficient of static (non-sliding) friction, applied once the
        point's sliding speed drops below ``settle_velocity``. Defaults to
        ``0.5`` (``ExponentialContactForce``'s own built-in default).
    dynamic_friction : float, optional
        Coefficient of kinetic (sliding) friction, applied once the
        point's sliding speed exceeds ``settle_velocity``. Must not exceed
        ``static_friction`` (OpenSim enforces ``dynamic_friction <=
        static_friction`` internally regardless; this is checked here
        first for a clear error instead of a silently clamped value).
        Defaults to ``0.2`` (``ExponentialContactForce``'s own built-in
        default).
    settle_velocity : float, optional
        Sliding speed, in m/s, below which the point is considered
        "settled" (static friction applies) rather than "sliding" (kinetic
        friction applies), with a smooth blend around the threshold.
        Must be finite and strictly positive. Defaults to ``0.01``
        (``ExponentialContactForce``'s own built-in default). The
        remaining ``ExponentialContactForce`` parameters (the normal
        force's exponential shape, its viscosity, the friction spring's
        elasticity/viscosity, the maximum normal force) are left at
        OpenSim's own built-in defaults, tuned for typical contact
        interactions in SI units -- reachable, if needed, through the
        returned wrapper's ``.raw`` (e.g.
        ``result.raw.setFrictionElasticity(...)``).
    reinitialize : bool, optional
        See :func:`~opensim_models.operators._shared.add_component`.

    Returns
    -------
    components.Force
        The newly created ``ExponentialContactForce``, wrapped in the thin,
        generic :class:`~opensim_models.components.Force`.

    Raises
    ------
    ValueError
        If ``plane_normal`` is the zero vector, ``dynamic_friction`` is
        negative or exceeds ``static_friction``, or ``settle_velocity`` is
        not finite or not strictly positive.
    """
    body = _unwrap(body)
    plane_body = _unwrap(plane_body)
    point = np.asarray(point, dtype=float)
    plane_point_local = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    normal_length = np.linalg.norm(normal)
    if normal_length == 0.0:
        raise ValueError("plane_normal must not be the zero vector")
    normal = normal / normal_length

    if dynamic_friction < 0.0 or dynamic_friction > static_friction:
        raise ValueError(
            "must have 0 <= dynamic_friction <= static_friction, got "
            f"dynamic_friction={dynamic_friction!r}, static_friction={static_friction!r}"
        )
    if not np.isfinite(settle_velocity) or settle_velocity <= 0:
        raise ValueError(
            f"settle_velocity must be finite and strictly positive, got {settle_velocity!r}"
        )

    # Stessa logica di add_point_on_plane_constraint: leggo la posa ATTUALE
    # di plane_body (realizePosition esplicito, non ci si affida a uno stato
    # gia' propagato) e la "congelo" dentro la Transform -- vedi il docstring
    # sopra ("A real limitation") per il motivo (nessuna connessione via
    # socket a plane_body, solo una proprieta' statica).
    model.model.realizePosition(model.state)
    plane_point_ground = _local_to_ground(plane_body, model.state, plane_point_local)
    plane_rotation_matrix = plane_body.getRotationInGround(model.state).asMat33()
    plane_rotation = np.array(
        [[plane_rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
    )
    normal_ground = plane_rotation @ normal
    normal_ground = normal_ground / np.linalg.norm(normal_ground)

    # ExponentialContactForce vuole una Transform il cui asse Z locale e' la
    # normale della superficie (convenzione confermata leggendo il docstring
    # della classe e verificata costruendo un modello reale: la forza
    # normale repulsiva agisce lungo il +Z della Transform). X/Y (il piano
    # di scorrimento/attrito) sono completati arbitrariamente da Simbody --
    # va bene, l'attrito e' isotropo in questa funzione (un solo coefficiente
    # per direzione, non direzionale).
    rotation = model.opensim.Rotation()
    rotation.setRotationFromOneAxis(
        model.opensim.UnitVec3(model.opensim.Vec3(*[float(v) for v in normal_ground])),
        model.opensim.CoordinateAxis(2),
    )
    transform = model.opensim.Transform(
        rotation, model.opensim.Vec3(*[float(v) for v in plane_point_ground])
    )

    force = model.opensim.ExponentialContactForce(
        transform, body, model.opensim.Vec3(*[float(v) for v in point])
    )
    force.setName(name)
    force.setInitialMuStatic(float(static_friction))
    force.setInitialMuKinetic(float(dynamic_friction))
    force.setSettleVelocity(float(settle_velocity))

    add_component(model, "force", force, reinitialize=reinitialize)
    return components.Force(model, force)
