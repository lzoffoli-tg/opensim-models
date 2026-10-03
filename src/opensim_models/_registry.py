"""Generic registry/iteration helpers shared by :mod:`opensim_models.model`
and :mod:`opensim_models.operators`.

Neither of these depends on ``OpenSimModel``'s class body -- they operate on
raw ``opensim`` objects (a ``Set``-like container, a bare ``opensim.Model``)
via duck typing -- so they live in their own leaf module rather than
``model.py``, letting ``operators.py`` import them without importing
``model.py`` itself for this purpose (``operators.py`` still imports the real
``OpenSimModel`` class separately, for the handful of places that need an
actual ``isinstance`` check).
"""

from __future__ import annotations

import weakref
from typing import Any

# Bridges a bare opensim.Model (e.g. obtained from a component's own
# getModel()) back to the OpenSimModel wrapper that owns its `state` --
# needed because SWIG hands out a fresh Python proxy on every getModel()
# call, so identity can't be compared directly, only the underlying
# pointer address (`.this`) can; see _find_owner. Weak-valued so a
# garbage-collected OpenSimModel's entry disappears on its own.
_owners: "weakref.WeakValueDictionary[int, Any]" = weakref.WeakValueDictionary()


def _register_owner(instance: Any) -> None:
    _owners[int(instance.model.this)] = instance


def _find_owner(raw_model: Any) -> Any | None:
    """Return the live ``OpenSimModel`` wrapping ``raw_model``, if any.

    ``raw_model`` is a bare ``opensim.Model``, typically obtained from a
    component via its own ``getModel()``. Returns ``None`` if ``raw_model``
    is ``None`` or was never wrapped by a (still-alive) ``OpenSimModel``.
    """
    if raw_model is None:
        return None
    return _owners.get(int(raw_model.this))


def _iter_set(component_set: Any) -> Any:
    for index in range(component_set.getSize()):
        yield component_set.get(index)
