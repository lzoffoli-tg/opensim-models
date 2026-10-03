"""Generic add/remove infra shared by every component-category submodule.

``add_component``/``remove_component`` are the single, category-agnostic
sink every ``add_*``/``remove_*`` function elsewhere in :mod:`opensim_models.operators`
is built on top of.
"""

from __future__ import annotations

from typing import Any

__all__ = ["add_component", "remove_component"]

_COMPONENT_SET_GETTERS: dict[str, str] = {
    "body": "updBodySet",
    "joint": "updJointSet",
    "force": "updForceSet",  # Muscle is a Force subtype; see add_muscle/remove_muscle.
    "marker": "updMarkerSet",
    "constraint": "updConstraintSet",
    "controller": "updControllerSet",
    "contact_geometry": "updContactGeometrySet",
    "probe": "updProbeSet",
}


def _unwrap(thing: Any) -> Any:
    """Return ``thing.raw`` if ``thing`` is a :mod:`opensim_models.components`
    wrapper, else ``thing`` unchanged.

    Every function in this module that accepts a component (a body, a
    joint's ``parent_frame``, a marker/frame to rotate, ...) needs to work
    whether the caller passes the raw ``opensim`` object or the
    Python-friendly wrapper :class:`~opensim_models.model.OpenSimModel`'s
    own accessors now return (e.g. ``model.body(name)``) -- this is the
    one place that difference is absorbed, rather than teaching every
    ``isinstance``/``safeDownCast`` check here about the wrapper type.
    """
    return getattr(thing, "raw", thing)


def _component_set(model: "OpenSimModel", kind: str) -> Any:
    try:
        getter = _COMPONENT_SET_GETTERS[kind]
    except KeyError as error:
        valid = ", ".join(sorted(_COMPONENT_SET_GETTERS))
        raise ValueError(
            f"Unknown component kind {kind!r}; expected one of {valid}"
        ) from error
    return getattr(model.model, getter)()


def add_component(
    model: "OpenSimModel", kind: str, component: Any, *, reinitialize: bool = False
) -> Any:
    """Add a component to the model's matching set, by category.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the component to.
    kind : str
        Component category: ``"body"``, ``"joint"``, ``"force"`` (includes
        muscles), ``"marker"``, ``"constraint"``, ``"controller"``,
        ``"contact_geometry"`` or ``"probe"``.
    component : Any
        The already-constructed OpenSim component (e.g. an
        ``opensim.Body`` or ``opensim.Millard2012EquilibriumMuscle``).
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately
        after adding, preserving the current posture/velocity, so the
        model is ready to use -- equivalent to wrapping just this call in
        :meth:`OpenSimModel.structural_change`. Leave ``False`` and wrap a
        whole batch of calls in that context manager instead when adding
        several components together (e.g. a body and the joint connecting
        it): reinitializing after each one individually would fail, since
        the body has no joint yet.

    Returns
    -------
    Any
        ``component`` itself, unchanged and unwrapped -- the same object
        passed in, now owned by ``model``. Unlike the category-specific
        functions built on top of this one (e.g. :func:`add_body`,
        :func:`add_force`, :func:`add_marker`), this generic function does
        not wrap the result in a :mod:`opensim_models.components` class,
        since it has no way to know which wrapper (if any) suits ``kind``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the recognized component categories
        listed above.
    """
    component_set = _component_set(model, kind)
    if reinitialize:
        with model.structural_change():
            component_set.adoptAndAppend(component)
            component.thisown = False  # ownership now belongs to model.model; avoids a double-free on GC
    else:
        component_set.adoptAndAppend(component)
        component.thisown = False
    return component


def remove_component(
    model: "OpenSimModel", kind: str, name: str, *, reinitialize: bool = False
) -> None:
    """Remove a named component from the model's matching set, by category.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the component from.
    kind : str
        Component category -- see :func:`add_component`.
    name : str
        Name of the component to remove.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately
        after removing, preserving the current posture/velocity --
        equivalent to wrapping just this call in
        :meth:`OpenSimModel.structural_change`. Leave ``False`` and wrap a
        whole batch of removals in that context manager instead (e.g. a
        joint and the body it connects, which must be removed together).

    Raises
    ------
    ValueError
        If ``kind`` is not a recognized component category, or if no
        component named ``name`` exists in that category.
    """
    component_set = _component_set(model, kind)
    index = component_set.getIndex(name)
    if index < 0:
        raise ValueError(f"No {kind} named {name!r} in the model")
    if reinitialize:
        with model.structural_change():
            component_set.remove(index)
    else:
        component_set.remove(index)


