"""An in-process, interactive VTK visualizer for an ``OpenSimModel``.

Replaces the native Simbody visualizer (``opensim.ModelVisualizer``) that
:meth:`~opensim_models.model.OpenSimModel.show` used to open: that
visualizer runs as a *separate process* (a bundled ``simbody-visualizer``
executable) communicating over a pipe, and its own
``Visualizer.InputListener`` interface exposes exactly three callbacks --
``keyPressed``, ``menuSelected``, ``sliderMoved`` -- with no mouse-move/
hover event, no camera-transform getter (only a setter, and the user can
freely orbit/pan/zoom with the mouse, which is never reported back), and
no picking/ray-intersection API at all. There is therefore no way to know
where the mouse is over that window, or what is under it.

VTK's ``vtkRenderWindowInteractor``, by contrast, runs *in this same
process*: its ``MouseMoveEvent`` fires here, its camera transform is
always readable, and ``vtkCellPicker`` resolves a screen pixel to both a
3D world point and the actor under it -- exactly what a live coordinate/
component readout needs. The tradeoff is reimplementing geometry loading
(meshes and the native ``Brick``/``Cylinder``/``Sphere`` primitives) and
visuals that the Simbody visualizer provided for free; this module covers
only what :meth:`OpenSimModel.show` and the playback window
(:mod:`opensim_models._player`) actually need.

The interactor's own blocking ``Start()`` event loop is never used: it runs
a native Win32 message loop in C++ that does not release the GIL, so even
on its own background thread it stalls every other Python thread in the
process (confirmed directly -- a plain
``threading.Thread(target=interactor.Start).start()`` hangs the calling
thread's very next line). Instead, :meth:`VTKVisualizer.process_events`
wraps ``vtkRenderWindowInteractor.ProcessEvents()``, a single non-blocking
pump of pending window messages; :mod:`opensim_models._player` calls it
once per tick of its own Tk ``after()`` loop, on the same thread as Tk's
event processing, so one thread drives both GUIs without contention.

For the same reason, a ``VTKVisualizer`` must be *constructed* on that
same thread too, not handed over from another one afterward: its native
window is thread-affine (a plain Win32 rule -- a window's messages are
only ever delivered to the thread that created it), and
:mod:`opensim_models._player` additionally reparents that window into its
own Tk frame, a call that blocks waiting for the *creating* thread to
process it -- which deadlocks if that thread is the caller's and has
already moved on past ``show()``, as confirmed directly. ``_player.py``'s
``start_player`` is therefore what constructs every ``VTKVisualizer``,
never :meth:`~opensim_models.model.OpenSimModel.show` itself.

A pure-white actor on a near-white background looks exactly like a blank/
broken render at a glance -- confirmed the hard way, chasing a GL-context
red herring across several reparenting-timing variants before noticing
the *background* was rendering correctly all along and only the
(uncoloured, default-white) actor itself was invisible against it.
:meth:`_add_body_actors` sets an explicit material colour for exactly
this reason -- it is a visibility fix, not a cosmetic one.
"""

from __future__ import annotations

import threading
from typing import Any

from . import _geometry

__all__ = ["VTKVisualizer", "DEFAULT_SIZE"]

#: Initial render window size in pixels, as (width, height). Exposed so
#: :mod:`opensim_models._player` can size its viewer frame to match before
#: a ``VTKVisualizer`` even exists (it must embed that frame's handle
#: *during* construction -- see ``VTKVisualizer.__init__``'s ``embed``
#: parameter -- so construction can't happen first).
DEFAULT_SIZE = (900, 700)

# Ground reference plane: a flat grid in OpenSim's XZ plane (Y=0), matching
# every ``opensim.Model``'s default gravity direction ((0, -9.80665, 0),
# confirmed directly) -- i.e. "down" is -Y, so the ground belongs at Y=0,
# not Z=0. Size/resolution are just enough to read as a floor/scale
# reference without dominating the view for a human-scale (~1-2 m) model.
_GROUND_HALF_SIZE = 5.0  # metres
_GROUND_RESOLUTION = 10  # grid divisions per side


class VTKVisualizer:
    """Opens an interactive VTK window rendering ``model``'s current geometry.

    One actor per piece of geometry attached to one of ``model``'s bodies
    (a mesh file, or a native ``Brick``/``Cylinder``/``Sphere``), built
    once at construction; call :meth:`show` after any posture/structural
    change to re-sync every actor's transform and redraw (this mirrors
    ``opensim.ModelVisualizer.show(state)``'s own signature, so
    :mod:`opensim_models._player` drives either one the same way).

    Construction returns as soon as the first frame is rendered; the
    window stays interactive (camera orbit *and* hover tracking) only for
    as long as something calls :meth:`process_events` regularly (see that
    method, and the module docstring, for why that is a deliberate
    non-blocking pump rather than the interactor's own ``Start()``) --
    :mod:`opensim_models._player` is what actually does this, once per
    tick, for every ``VTKVisualizer`` :meth:`~opensim_models.model.OpenSimModel.show`
    creates.

    Parameters
    ----------
    model : OpenSimModel
        Model to render. Read once, at construction, for which bodies and
        geometry exist; call :meth:`show` to reflect a structural change
        (an added/removed body) with a new ``VTKVisualizer`` instead --
        this one's actors are fixed at construction time.
    embed : callable or None, optional
        If given, called once, with this visualizer's native window
        handle (:meth:`get_window_id`'s value), after the window exists
        but before its first paint -- :mod:`opensim_models._player`
        passes a callback that reparents it into its own Tk frame right
        then.

    Raises
    ------
    RuntimeError
        If the ``vtk`` package is not installed.
    """

    def __init__(self, model: "OpenSimModel", *, embed: Any = None) -> None:
        try:
            import vtk
        except ImportError as error:
            raise RuntimeError(
                "The 'vtk' package is required to show an OpenSimModel. "
                "Install it with 'pip install vtk'."
            ) from error
        self._vtk = vtk
        self._model = model

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.92, 0.93, 0.95)
        self._render_window = vtk.vtkRenderWindow()
        self._render_window.AddRenderer(self._renderer)
        self._render_window.SetSize(*DEFAULT_SIZE)
        self._render_window.SetWindowName(self._window_title())
        self._interactor = vtk.vtkRenderWindowInteractor()
        self._interactor.SetRenderWindow(self._render_window)
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)

        # (opensim.Body, vtkActor) pairs, to resync transforms in show();
        # actor -> (model name, component/body name), for hover_text().
        self._actor_bodies: list[tuple[Any, Any]] = []
        self._actor_components: dict[Any, tuple[str, str]] = {}
        self._hover_lock = threading.Lock()
        self._hover_text = ""

        self._build_actors()
        self._interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self._interactor.AddObserver("LeaveEvent", self._on_mouse_leave)
        self._interactor.Initialize()
        if embed is not None:
            embed(self.get_window_id())
        self.show(model.state)
        # framed on the model's own actors *before* the ground plane is
        # added below -- added after, its 10 m-wide extent would otherwise
        # dominate `ResetCamera`'s fit for any human-scale (~1-2 m) model.
        self._renderer.ResetCamera()
        self._ground_actor = self._build_ground_actor()
        self._renderer.AddActor(self._ground_actor)
        self._render_window.Render()

    def _window_title(self) -> str:
        name = self._model.model.getName()
        return f"opensim-models: {name}" if name else "opensim-models"

    def _build_ground_actor(self) -> Any:
        """Build the ground reference plane actor (see the module-level constants)."""
        plane = self._vtk.vtkPlaneSource()
        half = _GROUND_HALF_SIZE
        plane.SetOrigin(-half, 0.0, -half)
        plane.SetPoint1(half, 0.0, -half)
        plane.SetPoint2(-half, 0.0, half)
        plane.SetXResolution(_GROUND_RESOLUTION)
        plane.SetYResolution(_GROUND_RESOLUTION)
        mapper = self._vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = self._vtk.vtkActor()
        actor.SetMapper(mapper)
        properties = actor.GetProperty()
        properties.SetColor(0.85, 0.85, 0.82)
        properties.SetAmbient(0.6)
        properties.EdgeVisibilityOn()
        properties.SetEdgeColor(0.6, 0.6, 0.6)
        return actor

    def set_ground_visible(self, visible: bool) -> None:
        """Show or hide the ground reference plane, and redraw.

        Backed by ``vtkActor.SetVisibility``: when hidden, the ground is
        skipped entirely by the renderer -- not drawn, not in the depth
        buffer -- so this controls the same thing whether what is looking
        at the render is a human watching the live window or something
        reading pixels back from it (e.g. a future screenshot/video
        export).
        """
        self._ground_actor.SetVisibility(bool(visible))
        self._render_window.Render()

    def get_ground_visible(self) -> bool:
        """Return whether the ground reference plane is currently shown."""
        return bool(self._ground_actor.GetVisibility())

    def _build_actors(self) -> None:
        opensim = self._model.opensim
        model_name = self._model.model.getName() or type(self._model).__name__
        body_set = self._model.model.getBodySet()
        for index in range(body_set.getSize()):
            body = body_set.get(index)
            self._add_body_actors(body, model_name, opensim)

    def _add_body_actors(self, body: Any, model_name: str, opensim: Any) -> None:
        geometry_property = body.getPropertyByName("attached_geometry")
        for index in range(geometry_property.size()):
            geometry = body.get_attached_geometry(index)
            source = self._build_geometry_source(geometry, opensim)
            if source is None:
                continue
            mapper = self._vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = self._vtk.vtkActor()
            actor.SetMapper(mapper)
            # an actor's default (unset) material is plain white, which
            # under VTK's default headlight looks indistinguishable from
            # the light-gray background it was mistaken for exactly that,
            # at a glance, more than once while debugging this module.
            actor.GetProperty().SetColor(0.75, 0.76, 0.8)
            self._renderer.AddActor(actor)
            self._actor_bodies.append((body, actor))
            self._actor_components[actor] = (model_name, body.getName())

    def _build_geometry_source(self, geometry: Any, opensim: Any) -> Any | None:
        """Return a VTK source/reader for one attached ``DecorativeGeometry``.

        ``None`` for a mesh whose file can't be resolved, or a geometry
        type not handled -- see :func:`opensim_models._geometry.build_source`,
        shared with :class:`~opensim_models.components.Body`'s ``.corners``.
        """
        return _geometry.build_source(
            self._vtk, opensim, geometry, self._model._resolve_geometry_file
        )

    def _on_mouse_move(self, _obj: Any, _event: str) -> None:
        x, y = self._interactor.GetEventPosition()
        self._picker.Pick(x, y, 0, self._renderer)
        actor = self._picker.GetActor()
        if actor in self._actor_components:
            model_name, component_name = self._actor_components[actor]
            position = self._picker.GetPickPosition()
            text = (
                f"{model_name}-{component_name} "
                f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
            )
        else:
            text = ""
        with self._hover_lock:
            self._hover_text = text

    def _on_mouse_leave(self, _obj: Any, _event: str) -> None:
        with self._hover_lock:
            self._hover_text = ""

    def hover_text(self) -> str:
        """Return the current hover readout: ``"<model>-<component> (x, y, z)"``.

        Empty when the pointer is outside the window or not over a
        rendered piece of geometry. The lock guarding this is a leftover
        safety net from the previous threaded design, now harmless rather
        than load-bearing: the mouse-move observer only ever fires inside
        :meth:`process_events`, called from the same thread that reads
        this back (see :mod:`opensim_models._player`'s tick loop).
        """
        with self._hover_lock:
            return self._hover_text

    def show(self, state: Any) -> None:
        """Sync every actor's transform from ``state`` and redraw.

        Same signature as ``opensim.ModelVisualizer.show(state)``, so
        anything already written against that (e.g.
        :mod:`opensim_models._player`) drives this the same way.
        """
        self._model.model.realizePosition(state)
        for body, actor in self._actor_bodies:
            position = body.getPositionInGround(state)
            rotation_matrix = body.getRotationInGround(state).asMat33()
            matrix = self._vtk.vtkMatrix4x4()
            for row in range(3):
                for column in range(3):
                    matrix.SetElement(row, column, rotation_matrix.get(row, column))
                matrix.SetElement(row, 3, position.get(row))
            actor.SetUserMatrix(matrix)
        self._render_window.Render()

    def process_events(self) -> None:
        """Pump pending window messages once (mouse move, drag/orbit, resize, close).

        Non-blocking -- unlike ``vtkRenderWindowInteractor.Start()``, which
        loops internally until the window closes (see the module
        docstring for why that cannot run on a background thread here).
        Must be called repeatedly, from whatever thread owns this
        process's single native event-pump duty, for the window to stay
        responsive at all; :mod:`opensim_models._player` calls it once per
        tick of its own Tk loop.
        """
        self._interactor.ProcessEvents()

    def get_size(self) -> tuple[int, int]:
        """Return the current render window size in pixels, as ``(width, height)``."""
        width, height = self._render_window.GetSize()
        return (width, height)

    def resize(self, width: int, height: int) -> None:
        """Resize the render window to ``(width, height)`` pixels and redraw.

        Called by :mod:`opensim_models._player` whenever the Tk frame this
        window is embedded into is resized, to keep the two in sync.
        """
        self._render_window.SetSize(width, height)
        self._render_window.Render()

    def get_window_id(self) -> Any:
        """Return the native window handle (``HWND`` on Windows).

        For :mod:`opensim_models._player` to embed this window directly
        into its own Tk frame -- no window search needed (unlike the old
        Simbody visualizer, a separate process), since this window was
        created in this same process.
        """
        return self._render_window.GetGenericWindowId()

    def close(self) -> None:
        """Release the render window.

        Idempotent. There is no interactor loop to stop (see the module
        docstring): :meth:`process_events` simply stops being called once
        the owning :mod:`opensim_models._player` window closes.
        """
        self._render_window.Finalize()
