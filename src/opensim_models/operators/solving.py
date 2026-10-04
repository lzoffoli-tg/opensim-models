"""Numeric, crash-safe alternatives to ``Model.assemble()``.

**The problem this sidesteps.** OpenSim's own constraint-satisfaction
machinery -- ``Model.assemble()`` itself, but also anything else that routes
through the same native ``SimTK::Assembler`` (e.g. ``Coordinate.setValue(state,
value, enforce_constraints=True)`` when that coordinate participates in an
unsatisfied constraint) -- is confirmed, by direct minimal repro, to segfault
the process (not a catchable Python exception) on the bundled Rajagopal-based
``User`` model whenever it actually has real work to do: an unsatisfied
constraint with free coordinates available to adjust. This is the same class
of native crash already documented elsewhere in this package --
:meth:`~opensim_models.model.OpenSimModel.update_state` (coupled-coordinate
re-solving "would require ``Model.assemble()``, which is also not reliably
safe to call"), :func:`~opensim_models.operators.add_point_on_plane_constraint`
(an *unsatisfied* distance-type constraint between two ground-descended
bodies segfaults ``initSystem()``), and
:meth:`~opensim_models.model.OpenSimModel.settle_under_gravity` (a different
native assembler/integrator crash, same genre).

**The workaround.** Two primitives are confirmed, empirically, to always be
safe on this model: ``Coordinate.setValue(state, value, enforce_constraints=False)``
(writes the raw value, never touches the assembler) and
``Model.realizePosition(state)`` (propagates that raw value to every derived
geometric quantity -- body/marker/joint positions -- without re-solving any
constraint). :func:`solve_coordinates` builds a numeric root-finder entirely
out of those two primitives plus ``scipy.optimize.least_squares``: whatever
kinematic condition would otherwise be handed to the native assembler (e.g.
"this body's point must land on that plane", "these two points must
coincide") is instead expressed as a Python residual function driven to zero
by iterating candidate coordinate values through the safe ``set_value``/
``realizePosition`` pair. :func:`solve_point_coincidence` is the single most
common shape this takes in practice (a shoulder pad, backrest or footrest
landmark that must coincide with a body landmark) wrapped as a ready-to-use
convenience on top of :func:`solve_coordinates`.

**The safety guarantee.** Neither function here ever calls
``Model.assemble()``, ``Manager``, or ``Coordinate.setValue`` with
``enforce_constraints=True`` -- grep this module if in doubt. A failure to
converge therefore always shows up as an ordinary, catchable Python
exception (or, with ``raise_on_failure=False``, as a
``scipy.optimize.OptimizeResult`` with ``success=False``) carrying scipy's
own convergence diagnostics -- never a process crash.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .. import components
from .geometry import from_local_to_global

__all__ = ["solve_coordinates", "solve_point_coincidence"]


def _resolve_coordinate(model: "OpenSimModel", coordinate: Any) -> "components.Coordinate":
    if isinstance(coordinate, components.Coordinate):
        return coordinate
    return model.coordinate(coordinate)


def solve_coordinates(
    model: "OpenSimModel",
    residual_fn: Callable[[np.ndarray], Any],
    coordinate_names: Sequence[Any],
    x0: Sequence[float] | None = None,
    *,
    raise_on_failure: bool = True,
    **least_squares_kwargs: Any,
) -> Any:
    """Solve for coordinate values that drive ``residual_fn`` to zero.

    The generic, N-dimensional numeric core this module exists for: a
    crash-safe, Python-level replacement for whatever kinematic condition
    would otherwise be handed to OpenSim's native ``Model.assemble()`` (see
    the module docstring for why that is not reliably safe to call on this
    package's models). On every trial, this sets ``coordinate_names`` to a
    candidate ``x`` via ``Coordinate.setValue(state, value,
    enforce_constraints=False)`` (confirmed always safe -- never touches the
    native assembler), realizes the state through ``Stage::Position`` via
    ``model.model.realizePosition(model.state)`` (also confirmed always
    safe), and calls ``residual_fn(x)`` to read back whatever
    position/marker/joint-centre quantity the caller cares about -- already
    fresh, with no further ``update_state()`` needed. ``scipy.optimize.least_squares``
    then searches for the ``x`` that drives the returned residual vector to
    zero (or as close to zero as it can get).

    For the single most common recurring case -- making one or more pairs of
    points coincide -- :func:`solve_point_coincidence` is a more ergonomic
    wrapper built on top of this function; reach for this one directly when
    the condition is something else (e.g. a signed distance to a plane, an
    angle between two segments, ...).

    Parameters
    ----------
    model : OpenSimModel
        Model to solve on. Its coordinates are mutated in place, on
        ``model.state``, by every trial -- including the final one, left at
        the returned solution (or at the best ``x`` found, if it failed to
        converge; see ``raise_on_failure``).
    residual_fn : callable
        Called as ``residual_fn(x)`` with ``model``'s coordinates already
        set to the trial values ``x`` (in the order of ``coordinate_names``)
        and ``model.state`` already realized through ``Stage::Position`` --
        safe to read ``position_global``/``position_local``/joint-centre
        properties/``euclidean_distance`` immediately, with no further
        realization needed. Must return an array-like of residuals (one
        value per condition to satisfy, not necessarily the same length as
        ``coordinate_names`` -- ``least_squares`` only requires at least as
        many residuals as unknowns); should return ~0 at the desired
        solution.
    coordinate_names : sequence of str or components.Coordinate
        The free coordinates to solve for, by exact OpenSim name or as
        already-looked-up :class:`~opensim_models.components.Coordinate`
        wrappers (mixing both in the same sequence is fine). Driven in the
        coordinate's own native unit (radians for a rotational coordinate,
        metres for a translational one) -- convert explicitly in
        ``residual_fn``/when interpreting ``x`` if you need degrees.
    x0 : sequence of float or None, optional
        Initial guess, one value per entry in ``coordinate_names``, in each
        coordinate's native unit. When ``None`` (default), starts from each
        coordinate's current value (``model.state`` as it was before this
        call) -- a reasonable default whenever the model is already roughly
        posed, but a condition far from the current posture (or with
        multiple solutions) may need an explicit, closer ``x0`` to converge
        to the intended one.
    raise_on_failure : bool, optional
        When ``True`` (default), raise ``RuntimeError`` if
        ``scipy.optimize.least_squares`` reports ``success=False`` (its own
        convergence criterion -- see its ``status``/``message``), after
        still leaving ``model``'s coordinates at the best ``x`` found. When
        ``False``, never raises for non-convergence -- inspect the returned
        ``OptimizeResult``'s own ``success``/``status``/``message``/``cost``
        instead. Either way, this is the *only* kind of failure this
        function can produce: see the module docstring for the guarantee
        that it never risks the native assembler crash instead.
    **least_squares_kwargs
        Forwarded to ``scipy.optimize.least_squares`` (e.g. ``bounds``,
        ``method``, ``xtol``, ``max_nfev``) -- see its own documentation.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result object from ``scipy.optimize.least_squares``, unchanged
        (``x``, ``fun``, ``cost``, ``success``, ``status``, ``message``, ...).

    Raises
    ------
    ValueError
        If ``coordinate_names`` is empty, or ``x0`` is given but its length
        does not match ``coordinate_names``.
    RuntimeError
        If ``raise_on_failure=True`` (default) and
        ``scipy.optimize.least_squares`` does not converge. Carries the same
        ``status``/``message`` scipy itself reports, plus the best ``x``
        found -- a plain, catchable Python exception, never a process crash
        (see the module docstring).
    """
    if len(coordinate_names) == 0:
        raise ValueError("coordinate_names must not be empty")
    coordinates = [_resolve_coordinate(model, name) for name in coordinate_names]

    if x0 is None:
        x0_array = np.array([coordinate.value for coordinate in coordinates], dtype=float)
    else:
        x0_array = np.asarray(x0, dtype=float)
        if x0_array.shape != (len(coordinates),):
            raise ValueError(
                f"x0 must have exactly {len(coordinates)} values (one per "
                f"entry in coordinate_names), got shape {x0_array.shape}"
            )

    def _set_and_evaluate(x: np.ndarray) -> np.ndarray:
        for coordinate, value in zip(coordinates, x):
            # enforce_constraints=False: the one confirmed-safe way to write
            # a coordinate value on this model -- see the module docstring.
            coordinate.set_value(float(value), enforce_constraints=False)
        model.model.realizePosition(model.state)
        return np.asarray(residual_fn(x), dtype=float).ravel()

    result = least_squares(_set_and_evaluate, x0_array, **least_squares_kwargs)

    # least_squares' last internal function evaluation (e.g. for a
    # finite-difference Jacobian step) is not guaranteed to be at result.x
    # itself -- leave the model exactly at the reported solution, not at
    # whatever trial point the optimizer happened to evaluate last.
    _set_and_evaluate(result.x)

    if raise_on_failure and not result.success:
        coordinate_values = {
            coordinate.name: float(value) for coordinate, value in zip(coordinates, result.x)
        }
        raise RuntimeError(
            "solve_coordinates: scipy.optimize.least_squares did not "
            f"converge (status={result.status}, message={result.message!r}, "
            f"cost={result.cost!r}). This is a plain numeric non-convergence "
            "-- not a native crash, see this function's docstring -- most "
            "often because the requested condition is outside what "
            "coordinate_names can reach (a kinematic-closure/range-of-motion "
            "limit) or x0 started too far from any solution. model's "
            f"coordinates were left at the best values found: {coordinate_values!r}. "
            "Pass raise_on_failure=False to inspect the OptimizeResult "
            "instead of raising."
        )
    return result


def _resolve_point_pair(
    pair: tuple,
) -> tuple[Callable[[], tuple[float, float, float]], Callable[[], tuple[float, float, float]]]:
    """Normalize one ``point_pairs`` entry into a pair of zero-arg getters.

    Accepts either shape documented on :func:`solve_point_coincidence`:
    ``(getter_a, getter_b)`` already callable, or ``(frame_a, point_a,
    frame_b, point_b)`` -- a body-fixed local point on each side, turned
    into a getter via :func:`~opensim_models.operators.from_local_to_global`
    (the same ground-frame-transform helper every ``position_global`` in
    this package is built on, rather than re-deriving it here).
    """
    if len(pair) == 2:
        getter_a, getter_b = pair
        if not (callable(getter_a) and callable(getter_b)):
            raise TypeError(
                "a 2-element point_pairs entry must be (getter_a, getter_b), "
                f"both callable with no arguments -- got {pair!r}"
            )
        return getter_a, getter_b
    if len(pair) == 4:
        frame_a, point_a, frame_b, point_b = pair
        return (
            lambda frame=frame_a, point=point_a: from_local_to_global(point, frame),
            lambda frame=frame_b, point=point_b: from_local_to_global(point, frame),
        )
    raise ValueError(
        "each point_pairs entry must have either 2 elements (getter_a, "
        "getter_b) or 4 elements (frame_a, point_a, frame_b, point_b), got "
        f"{len(pair)} elements: {pair!r}"
    )


def solve_point_coincidence(
    model: "OpenSimModel",
    point_pairs: Sequence[tuple],
    coordinate_names: Sequence[Any],
    x0: Sequence[float] | None = None,
    *,
    raise_on_failure: bool = True,
    **least_squares_kwargs: Any,
) -> Any:
    """Solve for coordinate values that make each point pair coincide.

    A ready-to-use convenience over :func:`solve_coordinates` for the single
    most common recurring rigid-attachment shape in this kind of ergonomics
    study: a body-fixed landmark (a shoulder/acromion point, a heel, a
    pelvis/torso point) that must land on, or coincide with, a point fixed
    on a piece of equipment (a pad, a backrest, a footrest) -- without
    welding the two together via a native constraint (see
    :func:`~opensim_models.operators.add_point_on_plane_constraint`'s own
    docstring for why an *unsatisfied* constraint of that kind is itself a
    native-crash risk at ``initSystem()`` time; this function sidesteps the
    whole category by never constructing a constraint at all). Internally
    builds the residual as the concatenation of ``(point_a - point_b)`` for
    every pair, in order, then calls :func:`solve_coordinates`.

    Parameters
    ----------
    model : OpenSimModel
        Model to solve on -- see :func:`solve_coordinates`.
    point_pairs : sequence of tuple
        One entry per pair of points that should coincide. Each entry is
        either:

        - ``(getter_a, getter_b)``: two zero-argument callables, each
          returning a ground-frame ``(x, y, z)`` point when called -- e.g.
          ``lambda: user.right_shoulder`` or ``lambda:
          footrest.markers["contact"].position_global``. Use this form for
          anything not expressible as a fixed local point on a body/frame
          (a joint centre, a derived/projected point, a marker already
          giving a ground-frame position directly).
        - ``(frame_a, point_a, frame_b, point_b)``: ``frame_a``/``frame_b``
          are a :mod:`opensim_models.components` wrapper or raw ``opensim``
          object with a resolvable ground-frame pose (a ``Body``, ``Box``,
          ``OffsetFrame``, ...; the same objects
          :func:`~opensim_models.operators.from_local_to_global` accepts),
          and ``point_a``/``point_b`` are ``(x, y, z)`` points in that
          frame's own local axes (metres) -- e.g. a shoulder pad's contact
          point, or a footrest's heel-contact point, expressed once in that
          body's own frame and re-evaluated at its current pose on every
          trial.

        Both forms can be mixed freely across different entries of the same
        call.
    coordinate_names : sequence of str or components.Coordinate
        The free coordinates to solve for, by exact OpenSim name or as
        already-looked-up :class:`~opensim_models.components.Coordinate`
        wrappers (mixing both in the same sequence is fine). Driven in each
        coordinate's own native unit (radians for a rotational coordinate,
        metres for a translational one).
    x0 : sequence of float or None, optional
        Initial guess, one value per entry in ``coordinate_names``, in each
        coordinate's native unit. When ``None`` (default), starts from each
        coordinate's current value on ``model.state`` -- a reasonable
        default whenever the model is already roughly posed, but a target
        far from the current posture (or reachable by more than one
        coordinate combination) may need an explicit, closer ``x0`` to
        converge to the intended one.
    raise_on_failure : bool, optional
        When ``True`` (default), raise ``RuntimeError`` if
        ``scipy.optimize.least_squares`` reports ``success=False`` (its own
        convergence criterion -- see its ``status``/``message``), after
        still leaving ``model``'s coordinates at the best ``x`` found. When
        ``False``, never raises for non-convergence -- inspect the returned
        ``OptimizeResult``'s own ``success``/``status``/``message``/``cost``
        instead.
    **least_squares_kwargs
        Forwarded to ``scipy.optimize.least_squares`` (e.g. ``bounds``,
        ``method``, ``xtol``, ``max_nfev``) -- see its own documentation.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result object from ``scipy.optimize.least_squares``, unchanged
        (``x``, ``fun``, ``cost``, ``success``, ``status``, ``message``, ...).

    Raises
    ------
    ValueError
        If ``point_pairs`` or ``coordinate_names`` is empty, if any entry of
        ``point_pairs`` has neither 2 nor 4 elements, or if ``x0`` is given
        but its length does not match ``coordinate_names``.
    TypeError
        If a 2-element ``point_pairs`` entry's two elements are not both
        callable.
    RuntimeError
        If ``raise_on_failure=True`` (default) and
        ``scipy.optimize.least_squares`` does not converge. Carries the same
        ``status``/``message`` scipy itself reports, plus the best ``x``
        found -- a plain, catchable Python exception, never a process crash.
    """
    if len(point_pairs) == 0:
        raise ValueError("point_pairs must not be empty")
    getter_pairs = [_resolve_point_pair(pair) for pair in point_pairs]

    def _residual(x: np.ndarray) -> np.ndarray:
        residuals = []
        for getter_a, getter_b in getter_pairs:
            point_a = np.asarray(getter_a(), dtype=float)
            point_b = np.asarray(getter_b(), dtype=float)
            residuals.append(point_a - point_b)
        return np.concatenate(residuals)

    return solve_coordinates(
        model,
        _residual,
        coordinate_names,
        x0,
        raise_on_failure=raise_on_failure,
        **least_squares_kwargs,
    )
