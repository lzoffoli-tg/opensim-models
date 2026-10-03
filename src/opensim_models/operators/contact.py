"""Build-and-attach functions for contact geometry (``add_contact_sphere``,
``add_contact_half_space``, ``add_contact_mesh``) and a sliding point-contact
force between a point on one body and a flat plane fixed on another
(``add_sliding_point_contact``) -- a real, physical (force-based)
alternative to
:func:`~opensim_models.operators.constraints.add_point_on_plane_constraint`
for the same "point resting on, and free to slide along, a flat surface"
scenario (e.g. an acromion landmark resting on a shoulder pad's face, or a
pelvis/torso landmark resting on a backrest).

**The contact geometry functions and ``add_sliding_point_contact`` solve
different problems -- read this before reaching for ``ContactSphere``/
``ContactHalfSpace`` expecting a working force-based contact.** A
``ContactGeometry`` (sphere/half-space/mesh) is pure, inert shape: adding
one to a body is always safe and has no dynamics implications by itself
(see :func:`add_contact_sphere`/:func:`add_contact_half_space`/
:func:`add_contact_mesh` below). Pairing one with ``HuntCrossleyForce``/
``ElasticFoundationForce`` to turn it into an actual contact force is the
one path this package does **not** support: see
:func:`add_sliding_point_contact`'s own docstring ("Why
``ExponentialContactForce``, not ``HuntCrossleyForce``") for the confirmed
native-crash reason -- that nested ``ContactParameters`` class is not
reachable from Python in this installation, and the flattened setters that
look like a substitute segfault at the acceleration stage.
``ExponentialContactForce`` (built by :func:`add_sliding_point_contact`,
needing no ``ContactGeometry`` at all) remains the only confirmed-working
force-based contact in this installation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import components
from ._shared import add_component, _unwrap
from .constraints import _local_to_ground

__all__ = [
    "add_contact_sphere",
    "add_contact_half_space",
    "add_contact_mesh",
    "add_sliding_point_contact",
]


def _positive(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"value must be strictly positive, got {value!r}")
    return float(value)


def add_contact_sphere(
    model: "OpenSimModel",
    name: str,
    body: Any,
    radius: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    reinitialize: bool = False,
) -> Any:
    """Build an ``opensim.ContactSphere`` and attach it to ``body`` in one call.

    Pure geometry, not a force: adding this (or any ``ContactGeometry``)
    never, by itself, makes ``body`` actually collide with anything -- see
    this module's own docstring ("The contact geometry functions and
    ``add_sliding_point_contact`` solve different problems") before
    reaching for ``HuntCrossleyForce``/``ElasticFoundationForce`` to pair
    with it; that combination is confirmed not to work in this
    installation. Use :func:`add_sliding_point_contact` instead for an
    actual, working contact force.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the contact sphere to.
    name : str
        Name for the new contact geometry.
    body : opensim.PhysicalFrame or components wrapper
        The body (or other physical frame) this sphere is attached to.
    radius : float
        Sphere radius, in metres. Must be finite and strictly positive.
    location : tuple[float, float, float], optional
        Sphere centre, in metres, in ``body``'s own local frame. Defaults
        to ``(0.0, 0.0, 0.0)``.
    reinitialize : bool, optional
        See :func:`~opensim_models.operators._shared.add_component`.

    Returns
    -------
    components.ContactSphere
        The newly created contact geometry, wrapped in
        :class:`~opensim_models.components.ContactSphere`.

    Raises
    ------
    ValueError
        If ``radius`` is not finite or not strictly positive.
    """
    body = _unwrap(body)
    sphere = model.opensim.ContactSphere(
        _positive(radius), model.opensim.Vec3(*location), body, name
    )
    add_component(model, "contact_geometry", sphere, reinitialize=reinitialize)
    return components._wrap_contact_geometry(model, sphere)


def add_contact_half_space(
    model: "OpenSimModel",
    name: str,
    body: Any,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    reinitialize: bool = False,
) -> Any:
    """Build an ``opensim.ContactHalfSpace`` and attach it to ``body`` in one call.

    Pure geometry, not a force -- see :func:`add_contact_sphere`'s own
    warning (identical here) about pairing this with
    ``HuntCrossleyForce``/``ElasticFoundationForce``; use
    :func:`add_sliding_point_contact` for an actual, working contact force
    instead.

    Every point with a *positive* local X coordinate (relative to this
    geometry's own frame, i.e. ``body``'s frame tilted by
    ``orientation_deg``) is OpenSim's own documented convention for
    "solid/inside" -- so the open, contactable half-space extends along
    the local **negative** X direction from ``location``. Point
    ``orientation_deg`` so local -X faces the side contact should occur on
    (e.g. tilt by 90 degrees about Z to make a half-space whose open side
    faces along what was the local +Y axis).

    Parameters
    ----------
    model : OpenSimModel
        Model to add the contact half-space to.
    name : str
        Name for the new contact geometry.
    body : opensim.PhysicalFrame or components wrapper
        The body (or other physical frame) this half-space is attached to.
    location : tuple[float, float, float], optional
        A point on the dividing plane, in metres, in ``body``'s own local
        frame. Defaults to ``(0.0, 0.0, 0.0)``.
    orientation_deg : tuple[float, float, float], optional
        Orientation of this geometry's own frame relative to ``body``, as
        X-Y-Z body-fixed Euler angles in degrees -- see above for how this
        determines which side is open/contactable. Defaults to no tilt.
    reinitialize : bool, optional
        See :func:`~opensim_models.operators._shared.add_component`.

    Returns
    -------
    components.ContactHalfSpace
        The newly created contact geometry, wrapped in
        :class:`~opensim_models.components.ContactHalfSpace`.
    """
    body = _unwrap(body)
    half_space = model.opensim.ContactHalfSpace(
        model.opensim.Vec3(*location),
        model.opensim.Vec3(*np.deg2rad(orientation_deg)),
        body,
        name,
    )
    add_component(model, "contact_geometry", half_space, reinitialize=reinitialize)
    return components._wrap_contact_geometry(model, half_space)


def add_contact_mesh(
    model: "OpenSimModel",
    name: str,
    body: Any,
    mesh_file: str,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    reinitialize: bool = False,
) -> Any:
    """Build an ``opensim.ContactMesh`` and attach it to ``body`` in one call.

    Pure geometry, not a force -- see :func:`add_contact_sphere`'s own
    warning (identical here) about pairing this with
    ``HuntCrossleyForce``/``ElasticFoundationForce``; use
    :func:`add_sliding_point_contact` for an actual, working contact force
    instead (that function takes a flat plane, not an arbitrary mesh, as
    its contact surface -- there is no mesh-shaped equivalent in this
    package yet).

    **A confirmed native-crash gotcha this function exists specifically to
    avoid.** ``opensim.ContactMesh``'s own convenience constructor --
    ``ContactMesh(filename, location, orientation, frame, name)``, the
    obvious one-call way to build it, used by every other
    ``add_contact_*``/``add_*_body`` function in this package for its own
    primitive's equivalent constructor -- segfaults the process (not a
    catchable exception) the moment it is called, confirmed directly and
    reproduced with *both* an ``.stl`` file (written by this package's own
    mesh writer) and a bundled ``.vtp`` file (one of ``User``'s own,
    confirmed to otherwise load fine as body-attached geometry), on an
    otherwise completely ordinary one-body model. The crash is specific to
    that all-at-once constructor overload: building the same
    ``ContactMesh`` piecemeal instead -- default-construct, then
    ``set_filename``/``set_location``/``set_orientation``/``setName``/
    ``connectSocket_frame`` individually -- is confirmed to work
    correctly, with identical resulting geometry (same ``location``/
    ``filename``/ground-frame position read back afterward), and is what
    this function actually does under the hood.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the contact mesh to.
    name : str
        Name for the new contact geometry.
    body : opensim.PhysicalFrame or components wrapper
        The body (or other physical frame) this mesh is attached to.
    mesh_file : str
        Path to the mesh file (``.obj``/``.vtp``/``.stl``) to load the
        surface from -- not validated to exist by this function (confirmed
        directly: even ``initSystem()`` succeeds with a nonexistent path,
        via the piecemeal construction this function uses -- same lazy
        resolution as an attached body ``Mesh``).
    location : tuple[float, float, float], optional
        Mesh origin, in metres, in ``body``'s own local frame. Defaults to
        ``(0.0, 0.0, 0.0)``.
    orientation_deg : tuple[float, float, float], optional
        Mesh orientation relative to ``body``, as X-Y-Z body-fixed Euler
        angles in degrees. Defaults to no tilt.
    reinitialize : bool, optional
        See :func:`~opensim_models.operators._shared.add_component`.

    Returns
    -------
    components.ContactMesh
        The newly created contact geometry, wrapped in
        :class:`~opensim_models.components.ContactMesh`.
    """
    body = _unwrap(body)
    opensim = model.opensim
    # Built piecemeal (empty constructor + individual setters), NOT via
    # ContactMesh(filename, location, orientation, frame, name) -- see the
    # docstring above for the confirmed segfault that all-at-once
    # constructor overload triggers in this installation.
    mesh = opensim.ContactMesh()
    mesh.set_filename(str(mesh_file))
    mesh.set_location(opensim.Vec3(*location))
    mesh.set_orientation(opensim.Vec3(*np.deg2rad(orientation_deg)))
    mesh.setName(name)
    mesh.connectSocket_frame(body)
    add_component(model, "contact_geometry", mesh, reinitialize=reinitialize)
    return components._wrap_contact_geometry(model, mesh)


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
    components.ExponentialContactForce
        The newly created ``ExponentialContactForce``, wrapped in
        :class:`~opensim_models.components.ExponentialContactForce`, which
        exposes ``point_global`` (the station on ``body``) and
        ``plane_point_global`` (a point on the plane) -- see that class's
        own docstring for why neither has a ``..._local`` counterpart.

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
    return components._wrap_force(model, force)
