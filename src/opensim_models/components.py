"""Python-friendly wrappers around individual OpenSim components.

``OpenSimModel`` is a *container* (one ``opensim.Model``); the parts inside
it -- a ``Body``, a ``Joint``, a ``Marker``, ... -- used to come back from
its accessors (``.body(name)``, ``.bodies``, ...) as the raw SWIG-bound
``opensim.*`` object, with the full (and large) C++-mirroring API attached
and nothing about it documented in a way an IDE could show. Every class
here wraps one such raw object instead, exposing only a small,
consistently-named set of properties/``set_x()`` method pairs (never an
``@x.setter`` -- matching every other model class in this package, e.g.
:class:`~opensim_models.models.box.Box`), with real docstrings.

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

from . import _geometry

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

        Parameters
        ----------
        name : str
            New OpenSim name.
        """
        self._raw.setName(name)

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
        self._owner.model.realizePosition(self._owner.state)
        position = np.asarray(self._raw.getPositionInGround(self._owner.state).to_numpy())
        rotation_matrix = self._raw.getRotationInGround(self._owner.state).asMat33()
        rotation = np.array(
            [[rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
        )
        return position, rotation

    @property
    def com(self) -> tuple[float, float, float]:
        """Centre of mass in the ground frame, in metres.

        This body's own ``mass_center`` (in its local frame) transformed
        through its current ground-frame placement -- not the whole
        model's centre of mass (see ``Model.calcMassCenterPosition``,
        across every body, for that).
        """
        position, rotation = self._position_and_rotation()
        local_com = self._raw.get_mass_center()
        local = np.array([local_com.get(0), local_com.get(1), local_com.get(2)])
        return tuple(position + rotation @ local)

    @property
    def inclination(self) -> tuple[float, float, float]:
        """Orientation relative to ground, as X-Y-Z body-fixed Euler degrees.

        Same convention as :attr:`opensim_models.models.box.Box.angle_deg`.
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
        dispatch :class:`~opensim_models._vtk_visualizer.VTKVisualizer`
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
            New maximum isometric force. Must be strictly positive.
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
            New optimal fiber length. Must be strictly positive.
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
            New tendon slack length. Must be strictly positive.
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
            New pennation angle. Must be finite and within ``[0, 90)``.
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


class Force(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Force``.

    A muscle is a ``Force`` subtype in OpenSim; see :class:`Muscle` for
    muscle-specific properties on top of this base set (``.name``,
    ``.raw``).
    """


class Constraint(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Constraint``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring.
    """


class Controller(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Controller``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring.
    """


class ContactGeometry(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.ContactGeometry``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring.
    """


class Probe(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Probe``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring.
    """
