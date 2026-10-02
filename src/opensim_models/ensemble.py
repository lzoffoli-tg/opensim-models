"""An assembly of several independently-editable :class:`OpenSimModel` containers.

``OpenSimModel`` (see :mod:`opensim_models.model`) is a *container*, in CAD
terms: one ``opensim.Model``, holding components wrapped by
:mod:`opensim_models.components`. ``OpenSimEnsemble`` is the *assembly*
level above it: several containers kept separate -- ``user + screen``
leaves both independently editable afterward, unlike
:meth:`~opensim_models.model.OpenSimModel.add_model`, which permanently
clones one container's components into another. :meth:`show`/:meth:`export`
build a merged model from every container's *current* state on demand
instead, so a change made to a container after the ensemble was built
still shows up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .model import OpenSimModel

__all__ = ["OpenSimEnsemble"]


class OpenSimEnsemble:
    """A loose assembly of :class:`~opensim_models.model.OpenSimModel` containers.

    Built via ``model_a + model_b`` (or directly: ``OpenSimEnsemble((model_a,
    model_b))``); chaining (``a + b + c``, where any operand may already be
    an ``OpenSimEnsemble``) flattens into one assembly rather than nesting.

    Parameters
    ----------
    containers : sequence of OpenSimModel
        The containers making up this assembly, in order.
    """

    def __init__(self, containers: Sequence[OpenSimModel]) -> None:
        self._containers = tuple(containers)
        self._combined: OpenSimModel | None = None

    @property
    def containers(self) -> tuple[OpenSimModel, ...]:
        """The containers making up this assembly, in order."""
        return self._containers

    def combined(self) -> OpenSimModel:
        """Build a fresh ``OpenSimModel`` by merging every container's *current* state.

        Reuses :meth:`~opensim_models.model.OpenSimModel.add_model`'s own
        merge semantics (automatic renaming on collision, shared ground),
        applied fresh every call rather than cached -- this is what lets a
        later change to a container (``user.rotate(...)``,
        ``screen.set_angle_deg(...)``) show up here. For a *permanent*
        cross-container interaction (a joint/constraint spanning two
        containers), fuse the relevant containers first via
        ``container_a.add_model(container_b)``, then build the ensemble
        from the fused result instead -- nothing added directly to one
        ``combined()`` result survives the next ``show()``/``export()``/
        ``combined()`` call, same as nothing surviving past a plain
        ``add_model`` call on a model nobody kept combining into.

        Always an independent ``OpenSimModel``, never the same instance
        :meth:`show` keeps internally -- see that method for why it needs
        a different, reused-in-place object instead of calling this.
        """
        combined = OpenSimModel(model_path=None)
        for container in self._containers:
            combined.add_model(container)
        return combined

    def show(self, *args: Any, **kwargs: Any) -> None:
        """Rebuild the same internal ``OpenSimModel`` from every container's
        current state, and show it.

        Same parameters as :meth:`~opensim_models.model.OpenSimModel.show`.
        Deliberately does *not* call :meth:`combined` (a fresh
        ``OpenSimModel`` every time): reusing the *same* instance across
        calls instead, rebuilt in place, lets that instance's own
        ``show()`` close its own previous window before opening a fresh
        one -- exactly the close-then-reopen logic already proven safe
        throughout this package (see ``OpenSimModel.show()``). Managing
        that handoff here instead, between two *different* ``OpenSimModel``
        instances (the previous call's and this one's), was tried and
        rejected: confirmed directly to hang on the second ``show()`` call,
        racing the previous instance's background (Tk) thread tearing down
        its window against this call already starting a new one.

        Rebuilding this instance's components (below) still has to wait
        for its *own* previous window to actually finish closing first,
        not just signal it to -- mutating ``model.model``/``model.state``
        while that window's background thread might still be mid-tick,
        reading them, was tried and rejected too: confirmed directly to
        leave the process unable to exit afterward.
        """
        if self._combined is None:
            self._combined = OpenSimModel(model_path=None)
        else:
            if self._combined._player_window is not None:
                self._combined._player_window.close()
                self._combined._player_window.wait_closed(timeout=5.0)
                self._combined._player_window = None
                self._combined._player = None
            self._combined._visualizer = None
            self._combined.model = self._combined.opensim.Model()
            self._combined._unlock_coordinates()
            self._combined.state = self._combined.model.initSystem()
            self._combined._geometry_dirs = []
            self._combined._anchor_names = set()
            self._combined._merged = {}
        for container in self._containers:
            self._combined.add_model(container)
        self._combined.show(*args, **kwargs)

    @property
    def visualizer(self) -> Any | None:
        """The last :meth:`show`'s visualizer, or ``None`` before any ``show()`` call."""
        return self._combined.visualizer if self._combined is not None else None

    @property
    def player(self) -> Any | None:
        """The last :meth:`show`'s playback state machine, or ``None``.

        ``None`` before any ``show()`` call, or if it had no ``motion``.
        """
        return self._combined.player if self._combined is not None else None

    def export(self, model_path: str | Path, *, geometry_dir_name: str = "Geometry") -> Path:
        """Build :meth:`combined`'s current state and export it.

        Same parameters as :meth:`~opensim_models.model.OpenSimModel.export`.
        """
        return self.combined().export(model_path, geometry_dir_name=geometry_dir_name)

    def __add__(self, other: "OpenSimModel | OpenSimEnsemble") -> "OpenSimEnsemble":
        """Flatten ``self``'s containers with ``other``'s into one assembly.

        Raises
        ------
        TypeError
            If ``other`` is neither an ``OpenSimModel`` nor an
            ``OpenSimEnsemble``.
        """
        if isinstance(other, OpenSimEnsemble):
            return OpenSimEnsemble(self._containers + other._containers)
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel or OpenSimEnsemble, got {type(other).__name__!r}"
            )
        return OpenSimEnsemble(self._containers + (other,))

    def __radd__(self, other: "OpenSimModel") -> "OpenSimEnsemble":
        """Support ``other + self`` when ``other`` did not implement ``__add__``.

        Raises
        ------
        TypeError
            If ``other`` is not an ``OpenSimModel``.
        """
        if not isinstance(other, OpenSimModel):
            raise TypeError(
                f"other must be an OpenSimModel, got {type(other).__name__!r}"
            )
        return OpenSimEnsemble((other,) + self._containers)
