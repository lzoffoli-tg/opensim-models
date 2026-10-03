"""Python-friendly wrappers around individual OpenSim components.

``OpenSimModel`` is a *container* (one ``opensim.Model``); the parts inside
it -- a ``Body``, a ``Joint``, a ``Marker``, ... -- used to come back from
its accessors (``.body(name)``, ``.bodies``, ...) as the raw SWIG-bound
``opensim.*`` object, with the full (and large) C++-mirroring API attached
and nothing about it documented in a way an IDE could show. Every class
here wraps one such raw object instead, exposing only a small,
consistently-named set of properties/``set_x()`` method pairs (never an
``@x.setter`` -- matching every other model class in this package, e.g.
:class:`~opensim_models.components.Box`), with real docstrings.

This wraps the categories :class:`~opensim_models.model.OpenSimModel`
already directly touches -- the 8 ``_MERGE_SETS`` categories (body, joint,
force, marker, constraint, controller, contact geometry, probe) plus
coordinate -- not OpenSim's entire ~800-class surface. ``Force``/
``Constraint``/``Controller``/``ContactGeometry``/``Probe`` are thin,
intentionally minimal wrappers for now (OpenSim has many concrete subtypes
of each, with very different APIs); the pattern here -- a
:class:`_ComponentWrapper` subclass plus property/``set_x()`` pairs -- is
meant to be extended the same way later for any subtype that needs its
own dedicated properties.

Every wrapper's :attr:`~_ComponentWrapper.raw` is the escape hatch back to
the underlying ``opensim`` object, for anything not (yet) wrapped here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _geometry

__all__ = [
    "Body",
    "Coordinate",
    "Marker",
    "Muscle",
    "Joint",
    "Force",
    "Constraint",
    "Controller",
    "ContactGeometry",
    "Probe",
]


def _positive(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"value must be strictly positive, got {value!r}")
    return float(value)


def _position_and_rotation_in_ground(owner: "OpenSimModel", frame: Any) -> tuple[Any, Any]:
    """Return ``frame``'s current ``(position, rotation)`` in the ground frame.

    ``position`` is a shape-``(3,)`` numpy array, in metres; ``rotation`` is
    the matching shape-``(3, 3)`` rotation matrix (ground axes expressed in
    terms of ``frame``'s own axes, i.e. ``ground_vector = rotation @
    frame_vector``). Realizes ``owner.state`` through ``Stage::Position``
    first (``owner.model.realizePosition``), so this is always safe to call
    regardless of what has or hasn't been realized since the last posture
    change.

    This is the one piece of raw-SWIG plumbing -- ``getPositionInGround()``/
    ``getRotationInGround().asMat33()``/rebuilding a plain 3x3 numpy array
    from it element by element -- every ground-frame position/orientation
    property in this module is now built on top of
    (:attr:`Body.position_global`, :attr:`Body.com`, :attr:`Body.inclination`,
    :attr:`Body.corners`, :class:`Marker`/:class:`ContactGeometry`/
    :class:`Joint`'s own ``position_global``, :class:`Box`/:class:`Screen`'s
    ``origin``/``angle_deg``/``corners``), instead of re-deriving it inline
    in each one -- the same hand-duplicated dance that previously showed up
    independently in at least four different places across this package.

    Parameters
    ----------
    owner : OpenSimModel
        The model ``frame`` belongs to -- needed to realize/read its state.
    frame : opensim.Frame (or anything with matching getPositionInGround/getRotationInGround methods)
        The frame to query (e.g. an ``opensim.Body``, a
        ``PhysicalOffsetFrame``, or ``opensim.Ground`` itself).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(position, rotation)`` -- ``position`` has shape ``(3,)``,
        ``rotation`` has shape ``(3, 3)``.
    """
    owner.model.realizePosition(owner.state)
    position = np.asarray(frame.getPositionInGround(owner.state).to_numpy())
    rotation_matrix = frame.getRotationInGround(owner.state).asMat33()
    rotation = np.array(
        [[rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
    )
    return position, rotation


def _point_in_ground(
    owner: "OpenSimModel", frame: Any, local_point: Any
) -> tuple[float, float, float]:
    """Transform ``local_point`` (metres, in ``frame``'s own axes) into the ground frame.

    Built on :func:`_position_and_rotation_in_ground`; shared by every
    property that must express an offset *within* some frame (a body's own
    ``mass_center``, a marker's or contact geometry's ``location``) as a
    ground-frame point.

    Parameters
    ----------
    owner : OpenSimModel
        The model ``frame`` belongs to.
    frame : opensim.Frame
        The frame ``local_point`` is expressed in.
    local_point : sequence of 3 floats
        Point, in metres, in ``frame``'s own local axes.

    Returns
    -------
    tuple[float, float, float]
        The equivalent point in the ground frame, in metres.
    """
    position, rotation = _position_and_rotation_in_ground(owner, frame)
    local = np.asarray(local_point, dtype=float)
    return tuple(float(value) for value in (position + rotation @ local))


class _ComponentWrapper:
    """Base for every Python-friendly wrapper around a raw opensim component.

    Parameters
    ----------
    owner : OpenSimModel
        The model this component belongs to -- needed for anything that
        reads/writes ``owner.state`` or calls back into OpenSim through
        ``owner.model``/``owner.opensim``.
    raw : Any
        The underlying raw ``opensim`` object (e.g. an ``opensim.Body``).
    """

    def __init__(self, owner: "OpenSimModel", raw: Any) -> None:
        self._owner = owner
        self._raw = raw

    @property
    def name(self) -> str:
        """OpenSim name of this component."""
        return self._raw.getName()

    def set_name(self, name: str) -> None:
        """Rename this component.

        Also re-finalizes ``owner.model``'s connections, so every other
        component's socket that refers to this one (e.g. a joint's
        offset frame naming its parent/child body) gets its stored
        connectee path refreshed to the new name. Skipping this would
        leave those paths pointing at the old name -- harmless for the
        live, already-resolved connection (it keeps working off the
        direct object reference), but fatal the next time the model is
        cloned or merged elsewhere (:meth:`~opensim_models.model.OpenSimModel.copy`/
        :meth:`~opensim_models.model.OpenSimModel.add_model`/``+``), since
        that re-resolves every connection from its stored path string: a
        stale one raises a native ``RuntimeError`` from
        ``finalizeConnections()`` (``Component ... could not find
        '/bodyset/<old name>'``) instead of finding the renamed component.
        This is purely a bookkeeping refresh (no structural change), so
        ``owner.state`` stays valid -- no :meth:`~opensim_models.model.OpenSimModel.reinitialize`
        needed.

        Parameters
        ----------
        name : str
            New OpenSim name.
        """
        self._raw.setName(name)
        self._owner.model.finalizeConnections()

    @property
    def raw(self) -> Any:
        """The underlying raw ``opensim`` object, for anything this wrapper doesn't cover."""
        return self._raw

    def __eq__(self, other: object) -> bool:
        # SWIG hands out a fresh Python proxy on every access (same reason
        # model.py keeps a pointer-identity _owners map, see its
        # _register_owner/_find_owner) -- plain `is`/default `==` would
        # wrongly treat two wrappers around the *same* underlying component
        # as different objects.
        if isinstance(other, _ComponentWrapper):
            return int(self._raw.this) == int(other._raw.this)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(int(self._raw.this))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Body(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Body``."""

    @property
    def mass(self) -> float:
        """Mass, in kilograms."""
        return float(self._raw.getMass())

    def set_mass(self, kilograms: float) -> None:
        """Set this body's mass and rebuild the system so the change takes effect.

        A body's mass feeds into the multibody system's mass matrix, which
        OpenSim builds once in ``initSystem()``: changing the property
        alone would have no effect on dynamics until the system is
        rebuilt, so this calls :meth:`~opensim_models.model.OpenSimModel.reinitialize`
        afterward, preserving the current posture and velocities.

        Parameters
        ----------
        kilograms : float
            New mass, in kilograms. Must be strictly positive.

        Raises
        ------
        ValueError
            If ``kilograms`` is not finite or not strictly positive.
        """
        self._raw.setMass(_positive(kilograms))
        self._owner.reinitialize()

    def _position_and_rotation(self) -> tuple[Any, Any]:
        return _position_and_rotation_in_ground(self._owner, self._raw)

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This body's own frame origin, in the ground frame, in metres.

        Computed fresh from the model's current state on every access (via
        :func:`_position_and_rotation_in_ground`), so it always reflects
        this body's current placement -- never a stale cached value. Not
        the same thing as :attr:`com` (which additionally accounts for
        this body's own ``mass_center`` offset, so the two only coincide
        when ``mass_center`` is ``(0, 0, 0)``, as for e.g. :class:`Box`/
        :class:`Screen`).
        """
        position, _ = self._position_and_rotation()
        return tuple(float(value) for value in position)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This body's own origin, expressed in its own local frame: always ``(0.0, 0.0, 0.0)``.

        A body *is* its own local frame's origin, so this is trivially
        zero -- not informative on its own, but present (alongside
        :attr:`position_global`) so code written generically against "any
        spatial element" (e.g. :class:`Marker`, :class:`ContactGeometry`,
        :class:`Joint`, all of which have a non-trivial ``position_local``)
        can read both properties uniformly, without needing to know ahead
        of time whether a given element is a whole body or an offset
        within some parent frame.
        """
        return (0.0, 0.0, 0.0)

    @property
    def com(self) -> tuple[float, float, float]:
        """Centre of mass in the ground frame, in metres.

        This body's own ``mass_center`` (in its local frame) transformed
        through its current ground-frame placement -- not the whole
        model's centre of mass (see ``Model.calcMassCenterPosition``,
        across every body, for that).
        """
        local_com = self._raw.get_mass_center()
        local = (local_com.get(0), local_com.get(1), local_com.get(2))
        return _point_in_ground(self._owner, self._raw, local)

    @property
    def inclination(self) -> tuple[float, float, float]:
        """Orientation relative to ground, as X-Y-Z body-fixed Euler degrees.

        Same convention as :attr:`opensim_models.components.Box.angle_deg`.
        """
        self._owner.model.realizePosition(self._owner.state)
        rotation = self._raw.getRotationInGround(self._owner.state)
        euler = rotation.convertRotationToBodyFixedXYZ()
        return tuple(float(np.degrees(euler.get(i))) for i in range(3))

    @property
    def corners(self) -> tuple[tuple[float, float, float], ...]:
        """Ground-frame corners of this body's attached geometry, in metres.

        The 8 corners of the axis-aligned bounding box of every geometry
        item attached to this body (local-frame bounds unioned across all
        of them, via the same ``Mesh``/``Brick``/``Cylinder``/``Sphere``
        dispatch :class:`~opensim_models._gui.visualizer.VTKVisualizer`
        uses to render them -- see :mod:`opensim_models._geometry`),
        transformed through this body's current ground-frame placement.
        Empty if this body has no attached geometry, or none of it could
        be resolved (e.g. a mesh file that can't be found).
        """
        import vtk

        position, rotation = self._position_and_rotation()
        opensim = self._owner.opensim
        geometry_property = self._raw.getPropertyByName("attached_geometry")
        bounds_list = []
        for index in range(geometry_property.size()):
            geometry = self._raw.get_attached_geometry(index)
            bounds = _geometry.local_bounds(
                vtk, opensim, geometry, self._owner._resolve_geometry_file
            )
            if bounds is not None:
                bounds_list.append(bounds)
        if not bounds_list:
            return ()

        xmin = min(b[0] for b in bounds_list)
        xmax = max(b[1] for b in bounds_list)
        ymin = min(b[2] for b in bounds_list)
        ymax = max(b[3] for b in bounds_list)
        zmin = min(b[4] for b in bounds_list)
        zmax = max(b[5] for b in bounds_list)

        corners = []
        for x in (xmin, xmax):
            for y in (ymin, ymax):
                for z in (zmin, zmax):
                    local = np.array([x, y, z])
                    corners.append(tuple(position + rotation @ local))
        return tuple(corners)


class Coordinate(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Coordinate``."""

    @property
    def value(self) -> float:
        """Current value, in the coordinate's native unit (radians or metres)."""
        return float(self._raw.getValue(self._owner.state))

    def set_value(self, value: float, *, enforce_constraints: bool = False) -> None:
        """Set the value, in the coordinate's native unit (radians or metres).

        This only writes the raw value into ``owner.state``; it does not
        propagate it to anything derived (body positions, muscle lengths,
        forces, ...). Call :meth:`~opensim_models.model.OpenSimModel.update_state`
        before reading any derived quantity, or after a batch of several
        ``set_*``/direct ``state`` edits, to bring the whole state up to
        date in one pass.

        Parameters
        ----------
        value : float
            New value, in radians (rotational coordinate) or metres
            (translational coordinate).
        enforce_constraints : bool, optional
            Passed through to OpenSim's own ``Coordinate.setValue``.
            Defaults to ``False`` (matching this package's previous
            ``set_coordinate_degrees``, which always used ``False``):
            setting ``True`` makes OpenSim immediately re-solve any
            constraint coupled to this coordinate (e.g. a
            ``CoordinateCouplerConstraint``), which is more expensive and
            not always needed mid-edit.

        Raises
        ------
        ValueError
            If ``value`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if self._raw.get_locked():
            raise ValueError(f"OpenSim coordinate {self.name!r} is locked")
        self._raw.setValue(self._owner.state, float(value), enforce_constraints)

    @property
    def value_degrees(self) -> float:
        """Current value, converted from radians to degrees.

        A convenience for a rotational coordinate; for a translational one
        this still converts as if it were an angle, same as this
        package's previous ``coordinate_degrees`` always did.
        """
        return float(np.rad2deg(self.value))

    def set_value_degrees(
        self, degrees: float, *, enforce_constraints: bool = False
    ) -> None:
        """Set the value, converting from degrees to radians first.

        See :meth:`set_value` for the ``enforce_constraints``/state-update
        caveats; this just converts units before calling it.

        Parameters
        ----------
        degrees : float
            New value in degrees.
        enforce_constraints : bool, optional
            See :meth:`set_value`.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(degrees):
            raise ValueError("degrees must be finite")
        self.set_value(float(np.deg2rad(degrees)), enforce_constraints=enforce_constraints)

    @property
    def speed(self) -> float:
        """Current speed, in the coordinate's native unit per second."""
        return float(self._raw.getSpeedValue(self._owner.state))

    def set_speed(self, value: float) -> None:
        """Set the speed, in the coordinate's native unit per second.

        Same state-update caveat as :meth:`set_value`.

        Parameters
        ----------
        value : float
            New speed, in radians/s (rotational) or metres/s (translational).

        Raises
        ------
        ValueError
            If ``value`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if self._raw.get_locked():
            raise ValueError(f"OpenSim coordinate {self.name!r} is locked")
        self._raw.setSpeedValue(self._owner.state, float(value))

    @property
    def speed_degrees(self) -> float:
        """Current speed, converted from radians/s to degrees/s."""
        return float(np.rad2deg(self.speed))

    def set_speed_degrees(self, degrees_per_second: float) -> None:
        """Set the speed, converting from degrees/s to radians/s first.

        Parameters
        ----------
        degrees_per_second : float
            New speed in degrees per second.

        Raises
        ------
        ValueError
            If ``degrees_per_second`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(degrees_per_second):
            raise ValueError("degrees_per_second must be finite")
        self.set_speed(float(np.deg2rad(degrees_per_second)))

    @property
    def locked(self) -> bool:
        """Whether this coordinate currently rejects new values."""
        return bool(self._raw.get_locked())

    def set_locked(self, locked: bool) -> None:
        """Lock or unlock this coordinate.

        Parameters
        ----------
        locked : bool
            Whether the coordinate should reject new values.
        """
        self._raw.set_locked(bool(locked))

    @property
    def range(self) -> tuple[float, float]:
        """Allowed range of motion, in the coordinate's native unit."""
        return (float(self._raw.getRangeMin()), float(self._raw.getRangeMax()))

    def set_range(self, range_: tuple[float, float]) -> None:
        """Set the allowed range of motion, in the coordinate's native unit.

        Parameters
        ----------
        range_ : tuple[float, float]
            ``(min, max)`` bounds, in radians (rotational) or metres
            (translational).

        Raises
        ------
        ValueError
            If a bound is not finite, or ``min >= max``.
        """
        min_value, max_value = range_
        if not (np.isfinite(min_value) and np.isfinite(max_value)):
            raise ValueError("range bounds must be finite")
        if min_value >= max_value:
            raise ValueError("min must be less than max")
        self._raw.setRangeMin(float(min_value))
        self._raw.setRangeMax(float(max_value))


class Marker(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Marker``."""

    @property
    def location(self) -> tuple[float, float, float]:
        """Offset within this marker's parent frame, in metres."""
        location = self._raw.get_location()
        return (location.get(0), location.get(1), location.get(2))

    def set_location(self, location: tuple[float, float, float]) -> None:
        """Set this marker's offset within its parent frame, in metres.

        Parameters
        ----------
        location : tuple[float, float, float]
            ``(x, y, z)`` offset coordinates in metres.

        Raises
        ------
        ValueError
            If a coordinate is not finite.
        """
        x, y, z = location
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("location must be finite")
        self._raw.set_location(self._owner.opensim.Vec3(float(x), float(y), float(z)))

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This marker's location, in the ground frame, in metres.

        :attr:`location` transformed through its parent frame's current
        ground-frame placement -- computed directly via OpenSim's own
        ``Marker.getLocationInGround`` (equivalent to, but cheaper than,
        going through :func:`_point_in_ground` with this marker's own
        parent frame), after realizing the model's state through
        ``Stage::Position``. Computed fresh on every access.
        """
        self._owner.model.realizePosition(self._owner.state)
        location = self._raw.getLocationInGround(self._owner.state)
        return (float(location.get(0)), float(location.get(1)), float(location.get(2)))

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This marker's offset within its parent frame, in metres -- the same value as :attr:`location`.

        Present alongside :attr:`position_global` purely for naming
        consistency with every other wrapper's ``position_global``/
        ``position_local`` pair (e.g. :attr:`Body.position_local`,
        :attr:`ContactGeometry.position_local`, :attr:`Joint.position_local`)
        -- prefer :attr:`location`/:meth:`set_location` when working with a
        ``Marker`` specifically, since those also support writing.
        """
        return self.location


class Muscle(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Muscle``."""

    @property
    def max_isometric_force(self) -> float:
        """Maximum isometric force, in newtons."""
        return float(self._raw.getMaxIsometricForce())

    def set_max_isometric_force(self, newtons: float) -> None:
        """Set the maximum isometric force, in newtons.

        Parameters
        ----------
        newtons : float
            New maximum isometric force, in newtons. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``newtons`` is not finite or not strictly positive.
        """
        self._raw.setMaxIsometricForce(_positive(newtons))

    @property
    def optimal_fiber_length(self) -> float:
        """Optimal fiber length, in metres."""
        return float(self._raw.getOptimalFiberLength())

    def set_optimal_fiber_length(self, meters: float) -> None:
        """Set the optimal fiber length, in metres.

        Parameters
        ----------
        meters : float
            New optimal fiber length, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``meters`` is not finite or not strictly positive.
        """
        self._raw.setOptimalFiberLength(_positive(meters))

    @property
    def tendon_slack_length(self) -> float:
        """Tendon slack length, in metres."""
        return float(self._raw.getTendonSlackLength())

    def set_tendon_slack_length(self, meters: float) -> None:
        """Set the tendon slack length, in metres.

        Parameters
        ----------
        meters : float
            New tendon slack length, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``meters`` is not finite or not strictly positive.
        """
        self._raw.setTendonSlackLength(_positive(meters))

    @property
    def pennation_angle(self) -> float:
        """Pennation angle at optimal fiber length, in degrees."""
        return float(np.rad2deg(self._raw.getPennationAngleAtOptimalFiberLength()))

    def set_pennation_angle(self, degrees: float) -> None:
        """Set the pennation angle at optimal fiber length, in degrees.

        Parameters
        ----------
        degrees : float
            New pennation angle, in degrees. Must be finite and within
            ``[0, 90)`` -- ``90`` itself is excluded since a pennation
            angle that large would leave no fiber length component along
            the line of action.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite or not in ``[0, 90)``.
        """
        if not np.isfinite(degrees) or not (0.0 <= degrees < 90.0):
            raise ValueError(
                "pennation angle must be a finite value in [0, 90) degrees"
            )
        self._raw.setPennationAngleAtOptimalFiberLength(float(np.deg2rad(degrees)))


class Joint(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Joint``."""

    @property
    def coordinates(self) -> dict[str, Coordinate]:
        """Coordinates owned by this joint, keyed by name.

        Via ``numCoordinates()``/``get_coordinates(i)`` -- the generic,
        multi-DOF-safe OpenSim API (``getCoordinate()``, singular, is
        documented single-DOF-only, so it is never used here). There is no
        equivalent on :class:`~opensim_models.model.OpenSimModel` itself:
        ``.coordinate(name)``/``.coordinates`` there go through the
        model's global ``CoordinateSet``, independent of which joint owns
        each one.
        """
        return {
            self._raw.get_coordinates(i).getName(): Coordinate(
                self._owner, self._raw.get_coordinates(i)
            )
            for i in range(self._raw.numCoordinates())
        }

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This joint's child frame position, in the ground frame, in metres.

        A ``Joint`` connects a parent frame and a child frame, each
        independently placeable -- resolving it to a *single* ground-frame
        point means picking one of the two, and this follows the same
        convention already used wherever else this package resolves a
        ``Joint`` to one ground-frame point (e.g.
        :func:`opensim_models.operators.rotate_object`'s ``origin``
        argument, and every joint-centre property on
        :class:`~opensim_models.models.User`, e.g. ``.left_knee``): the
        **child** frame, not the parent's. The two coincide only while
        this joint's own coordinate(s) sit at the value that makes them so
        (typically zero, for a joint built by this package's
        ``add_*_joint`` helpers, before it's actually been posed/driven).
        Computed fresh from the model's current state on every access; use
        ``self.raw.getParentFrame()`` directly if the parent side's
        position is what you actually need.
        """
        child = self._owner.opensim.Frame.safeDownCast(self._raw.getChildFrame())
        position, _ = _position_and_rotation_in_ground(self._owner, child)
        return tuple(float(value) for value in position)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This joint's child frame offset from its own immediate parent, in metres.

        Matches :attr:`position_global`'s child-frame convention. Every
        joint built by this package's ``add_*_joint`` helpers (and every
        joint loaded from a real ``.osim`` model, confirmed directly
        against the bundled Rajagopal-based base model) gives its child
        frame as an ``opensim.PhysicalOffsetFrame`` -- even when built with
        a zero offset -- in which case this is that frame's own
        ``translation`` property: the offset from the body it's welded to,
        in that body's own local axes. Falls back to ``(0.0, 0.0, 0.0)``
        in the (currently unobserved, but not provably impossible) case
        where the child frame is *not* an offset frame at all -- consistent
        with :attr:`Body.position_local`, since a frame is trivially at
        the origin of itself.
        """
        child = self._raw.getChildFrame()
        offset = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(child)
        if offset is None:
            return (0.0, 0.0, 0.0)
        translation = offset.get_translation()
        return (
            float(translation.get(0)),
            float(translation.get(1)),
            float(translation.get(2)),
        )


class Force(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Force``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Force`` subtypes, each
    with its own API). Provides only the base :class:`_ComponentWrapper`
    surface: :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Force`` object for
    anything not wrapped here (e.g. force-specific getters/setters). A
    muscle is a ``Force`` subtype in OpenSim; see :class:`Muscle` for
    muscle-specific properties on top of this base set.
    """


class Constraint(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Constraint``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Constraint`` subtypes,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Constraint`` object for
    anything not wrapped here (e.g. a ``CoordinateCouplerConstraint``'s own
    function/coordinates).
    """


class Controller(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Controller``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Controller`` subtypes,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Controller`` object for
    anything not wrapped here (e.g. which actuators it controls).
    """


class ContactGeometry(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.ContactGeometry``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``ContactGeometry``
    subtypes, e.g. ``ContactSphere``/``ContactHalfSpace``/``ContactMesh``,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.ContactGeometry``
    object for anything not wrapped here (e.g. a sphere's radius).
    """

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This contact geometry's location, in the ground frame, in metres.

        Every concrete ``opensim.ContactGeometry`` subtype (``ContactSphere``,
        ``ContactHalfSpace``, ``ContactMesh``, ...) shares the same base
        ``location`` property (see :attr:`position_local`): an offset
        within whatever ``opensim.PhysicalFrame`` it is attached to (its
        ``frame`` socket, read via ``getFrame()``). This transforms that
        offset through the attached frame's current ground-frame placement
        (:func:`_point_in_ground`), after realizing the model's state
        through ``Stage::Position``. Verified directly against a
        ``ContactSphere`` welded to an offset ``WeldJoint`` child frame:
        matches the sphere's actual ground-frame position (``frame``'s
        position plus its rotation applied to ``location``).
        """
        frame = self._raw.getFrame()
        local = self._raw.get_location()
        local_point = (local.get(0), local.get(1), local.get(2))
        return _point_in_ground(self._owner, frame, local_point)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This contact geometry's offset within its attached frame, in metres.

        A plain re-expression of OpenSim's own ``get_location()`` property
        (shared by every concrete ``ContactGeometry`` subtype) as a tuple.
        No ``location``/``set_location`` pair exists yet on this thin
        wrapper (see the class docstring for why) -- use
        ``self.raw.get_location()``/``self.raw.set_location(...)`` directly
        to change it in the meantime.
        """
        local = self._raw.get_location()
        return (float(local.get(0)), float(local.get(1)), float(local.get(2)))


class Probe(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Probe``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Probe`` subtypes, each
    with its own API). Provides only the base :class:`_ComponentWrapper`
    surface: :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Probe`` object for
    anything not wrapped here (e.g. a probe's operation/report settings).
    """


# Concrete standalone components -- a Body subclass bundling its own
# private OpenSimModel, rather than a wrapper around a piece of someone
# else's model (see each module's own docstring for why).
from .box import Box
from .screen import Screen

__all__ += ["Box", "Screen"]
