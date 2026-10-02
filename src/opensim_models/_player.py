"""The unified viewer+playback window for an ``OpenSimModel``.

Used by :meth:`~opensim_models.model.OpenSimModel.show`: opens a single Tk
window embedding the (VTK-based, see :mod:`opensim_models._vtk_visualizer`)
3D view in its upper area, with two parts below it:

- Playback controls (play/pause, stop, fast-forward, fast-backward, a
  loop/cycle toggle, and a draggable progress slider) driving a motion
  loaded via ``show(motion=...)`` -- disabled whenever no motion is
  loaded, rather than the controls disappearing, so there is always
  somewhere to look for them.
- A ground reference plane toggle, next to the loop/cycle one -- unlike
  the playback controls, never disabled, since it controls the 3D view
  itself rather than anything about a loaded motion.
- A bottom status bar showing live hover info from the visualizer
  (``"<model>-<component> (x, y, z)"``), regardless of whether a motion
  is loaded.

This used to be two separate windows (the native 3D view plus a Tk window
kept docked under it), from when this project drove the *native* Simbody
visualizer (a separate process) and the only way to add playback/status
chrome to it at all was a second, independent window -- Simbody's own
widget API (``Visualizer.addSlider``/``addMenu``/``setWindowTitle`` --
anything taking a ``SimTK::String``) is not callable from Python on at
least some OpenSim builds. The visualizer is no longer Simbody's, and
runs in this same process now, so its native window is instead reparented
(Windows only -- see ``_embed_visualizer_window``) directly into a frame
inside this one Tk window: one window, no inter-window docking to keep
in sync.

Embedding was briefly suspected of corrupting VTK's OpenGL context (a
blank window, logging ``wglMakeCurrent failed ... resource is in use``)
and reverted to the two-window design once -- that diagnosis was wrong.
The window was never blank: its actors had no material colour set, so
VTK's default (plain white) rendered indistinguishably from the
near-white background underneath real geometry that was there all along,
proven by sampling actual pixel values on and off the actor rather than
eyeballing a screenshot. :meth:`~opensim_models._vtk_visualizer.VTKVisualizer._add_body_actors`
now sets an explicit colour for exactly this reason. The
``wglMakeCurrent`` log lines turned out to be unrelated, triggered by
*this project's own test scripts* calling VTK interactor methods directly
from a thread other than the one that created the window while chasing
the (nonexistent) embedding bug -- not something a real caller, driving
input through the OS's normal per-thread message dispatch, would ever
hit.

This module's ``start_player`` still constructs ``model``'s
:class:`~opensim_models._vtk_visualizer.VTKVisualizer` itself (setting
:attr:`~opensim_models.model.OpenSimModel.visualizer`), on the same
background thread that runs this window and calls
:meth:`~opensim_models._vtk_visualizer.VTKVisualizer.process_events`
every tick: a Win32 window's messages are only ever delivered to the
thread that created it, so both the embedding reparent call and every
later ``process_events()`` pump need to happen on that same thread.
"""

from __future__ import annotations

import re
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MotionData", "MotionPlayer", "PlayerWindow", "start_player"]

# Ordered, signed multiples of real time used by fast-forward/fast-backward:
# repeatedly clicking one steps outward (slower -> faster) in that
# direction; there is no "0x" entry since that would just be pause.
_SPEED_STEPS = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
_NEUTRAL_SPEED_INDEX = _SPEED_STEPS.index(1.0)

# OpenSim's own Coordinate::MotionType C++ enum (Rotational=1): not exposed
# as a named constant through these Python bindings, so used as a literal.
_ROTATIONAL_MOTION_TYPE = 1

_NO_MOTION_TEXT = "(nessuna animazione)"


class MotionData:
    """A time-indexed table of coordinate values, ready to drive a model.

    Parameters
    ----------
    model : OpenSimModel
        Model the motion will be applied to -- used only to tell a
        rotational coordinate (its column may need a degrees-to-radians
        conversion) from a translational one (never converted, regardless
        of the table's own ``inDegrees`` metadata), matching the convention
        every OpenSim motion/``.mot`` file already uses.
    table : opensim.TimeSeriesTable
        Already-loaded motion data: one column per OpenSim coordinate name
        (e.g. ``"hip_flexion_r"``; columns that don't match a coordinate of
        ``model`` are kept but simply never applied), one row per time
        sample.

    Raises
    ------
    ValueError
        If ``table`` has fewer than 2 rows.
    """

    def __init__(self, model: "OpenSimModel", table: Any) -> None:
        # opensim.TimeSeriesTable itself already rejects a non-increasing
        # time column (both built via appendRow and loaded from file), so
        # that isn't re-checked here.
        times = np.asarray(table.getIndependentColumn(), dtype=float)
        if times.size < 2:
            raise ValueError("motion must have at least 2 rows")

        in_degrees = (
            table.hasTableMetaDataKey("inDegrees")
            and str(table.getTableMetaDataAsString("inDegrees")).strip().lower() == "yes"
        )

        columns: dict[str, np.ndarray] = {}
        for name in table.getColumnLabels():
            values = np.asarray(table.getDependentColumn(name).to_numpy(), dtype=float)
            if (
                in_degrees
                and model.coordinates.contains(name)
                and model.coordinate(name).getMotionType() == _ROTATIONAL_MOTION_TYPE
            ):
                values = np.radians(values)
            columns[name] = values

        self.times = times
        self.columns = columns

    @property
    def start_time(self) -> float:
        """Return the motion's first sample time, in seconds."""
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        """Return the motion's last sample time, in seconds."""
        return float(self.times[-1])

    def values_at(self, time: float) -> dict[str, float]:
        """Return every column's value at ``time``, in metres/radians.

        Linearly interpolated between the two bracketing samples; ``time``
        outside ``[start_time, end_time]`` is clamped to that range first.

        Parameters
        ----------
        time : float
            Query time, in seconds.
        """
        clamped = min(max(time, self.start_time), self.end_time)
        return {
            name: float(np.interp(clamped, self.times, values))
            for name, values in self.columns.items()
        }


class MotionPlayer:
    """Pure playback state machine for a motion -- no GUI, no OpenSim call.

    Kept independent of both so its transitions (play/pause, stop,
    fast-forward/fast-backward, cycle, seek, advancing by elapsed real
    time) are exercised directly by unit tests, without a live visualizer.

    Parameters
    ----------
    start_time, end_time : float
        The motion's own time range, in seconds (e.g.
        :attr:`MotionData.start_time`/:attr:`MotionData.end_time`).
    loop : bool, optional
        Whether playback wraps back around at either end (``True``)
        instead of stopping there (``False``, default).

    Raises
    ------
    ValueError
        If ``end_time`` is not strictly greater than ``start_time``.
    """

    def __init__(self, start_time: float, end_time: float, *, loop: bool = False) -> None:
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        self.start_time = start_time
        self.end_time = end_time
        self.loop = loop
        self.time = start_time
        self.playing = False
        self._speed_index = _NEUTRAL_SPEED_INDEX

    @property
    def speed(self) -> float:
        """Return the current signed playback speed (multiples of real time)."""
        return _SPEED_STEPS[self._speed_index]

    @property
    def fraction(self) -> float:
        """Return how far through the motion :attr:`time` is, in ``[0, 1]``."""
        span = self.end_time - self.start_time
        return (self.time - self.start_time) / span if span else 0.0

    def play_pause(self) -> None:
        """Toggle between playing and paused, keeping the current position/speed."""
        self.playing = not self.playing

    def stop(self) -> None:
        """Stop playback and reset to the start, at normal forward speed."""
        self.playing = False
        self._speed_index = _NEUTRAL_SPEED_INDEX
        self.time = self.start_time

    def toggle_loop(self) -> None:
        """Toggle whether playback wraps around at either end instead of stopping."""
        self.loop = not self.loop

    def fast_forward(self) -> None:
        """Step the playback speed one notch faster forward, and start playing.

        Repeated calls step through ``1x, 2x, 4x, 8x`` (capped); from a
        negative (reverse) speed, steps back up toward ``1x`` first.
        """
        self._speed_index = min(self._speed_index + 1, len(_SPEED_STEPS) - 1)
        self.playing = True

    def fast_backward(self) -> None:
        """Step the playback speed one notch faster backward, and start playing.

        Repeated calls step through ``-1x, -2x, -4x, -8x`` (capped); from a
        positive (forward) speed, steps back down toward ``-1x`` first.
        """
        self._speed_index = max(self._speed_index - 1, 0)
        self.playing = True

    def seek_fraction(self, fraction: float) -> None:
        """Jump directly to ``fraction`` (in ``[0, 1]``) through the motion."""
        fraction = min(max(fraction, 0.0), 1.0)
        self.time = self.start_time + fraction * (self.end_time - self.start_time)

    def advance(self, dt: float) -> None:
        """Move playback forward by ``dt`` real seconds, at the current speed.

        A no-op while paused. Past either end: wraps around if
        :attr:`loop`, otherwise clamps there and pauses.

        Parameters
        ----------
        dt : float
            Elapsed real time, in seconds. Non-positive values are ignored.
        """
        if not self.playing or dt <= 0:
            return
        self.time += dt * self.speed
        span = self.end_time - self.start_time
        if self.time > self.end_time:
            if self.loop and span > 0:
                self.time = self.start_time + (self.time - self.end_time) % span
            else:
                self.time = self.end_time
                self.playing = False
        elif self.time < self.start_time:
            if self.loop and span > 0:
                self.time = self.end_time - (self.start_time - self.time) % span
            else:
                self.time = self.start_time
                self.playing = False


# ---------------------------------------------------------------------------
# Embedding the visualizer's native window into a Tk frame (Windows only)
# ---------------------------------------------------------------------------
#
# Windows-only: reparenting an arbitrary native window by handle is a
# Win32-specific operation; there is no equivalent here for other
# platforms, so there the visualizer keeps its own separate top-level
# window instead (still fully functional, just not embedded).
#
# VTK's own `SetParentId` (create the render window as a child to begin
# with) was tried first and rejected: it fails outright on this VTK build
# (`vtkWin32OpenGLRenderWindow`, error 1400/ERROR_INVALID_WINDOW_HANDLE,
# confirmed by direct testing, both via the SWIG-encoded pointer string and
# a `ctypes.c_void_p`). Reparenting the already-created top-level window
# after the fact -- strip its WS_POPUP/title-bar styles, `SetParent` it
# under the Tk frame, `MoveWindow` it to fill that frame -- works reliably
# and is the standard Win32 technique for embedding a foreign window.

_IS_WINDOWS = sys.platform.startswith("win")

_VTK_WINDOW_ID_PATTERN = re.compile(r"_([0-9a-fA-F]+)_p_void")


def _hwnd_from_vtk_window_id(window_id: Any) -> int:
    """Convert ``VTKVisualizer.get_window_id()``'s SWIG pointer encoding to a plain int HWND."""
    match = _VTK_WINDOW_ID_PATTERN.match(str(window_id))
    if match is None:
        raise ValueError(f"Unrecognized VTK window id format: {window_id!r}")
    return int(match.group(1), 16)


if _IS_WINDOWS:
    import ctypes

    _user32 = ctypes.windll.user32

    _GWL_STYLE = -16
    _WS_CHILD = 0x40000000
    _WS_VISIBLE = 0x10000000
    _WS_CLIPSIBLINGS = 0x04000000
    _WS_CLIPCHILDREN = 0x02000000
    _WS_POPUP = 0x80000000
    _WS_CAPTION = 0x00C00000
    _WS_THICKFRAME = 0x00040000
    _WS_SYSMENU = 0x00080000
    _SWP_NOMOVE = 0x0002
    _SWP_NOSIZE = 0x0001
    _SWP_NOZORDER = 0x0004
    _SWP_FRAMECHANGED = 0x0020
    _SW_HIDE = 0

    def _embed_visualizer_window(hwnd: int, parent_hwnd: int) -> None:
        """Reparent the visualizer's top-level ``hwnd`` into ``parent_hwnd``, filling it.

        Hides ``hwnd`` first as cheap insurance against it painting a
        frame as a top-level window before this call lands (it is called
        right after that window is created, before its first paint, so in
        practice there is nothing to hide yet). Sets
        ``WS_CLIPCHILDREN``/``WS_CLIPSIBLINGS`` alongside ``WS_CHILD``,
        per Microsoft's own documented rules for any window used for
        OpenGL rendering: the *top-level* window this used to be had no
        siblings/children for clipping to matter for, so it never needed
        them until now.
        """
        _user32.ShowWindow(hwnd, _SW_HIDE)
        style = _user32.GetWindowLongPtrW(hwnd, _GWL_STYLE)
        style &= ~(_WS_POPUP | _WS_CAPTION | _WS_THICKFRAME | _WS_SYSMENU)
        style |= _WS_CHILD | _WS_VISIBLE | _WS_CLIPSIBLINGS | _WS_CLIPCHILDREN
        _user32.SetWindowLongPtrW(hwnd, _GWL_STYLE, style)
        _user32.SetParent(hwnd, parent_hwnd)
        _user32.SetWindowPos(
            hwnd, None, 0, 0, 0, 0, _SWP_FRAMECHANGED | _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
        )

    def _resize_embedded_window(hwnd: int, width: int, height: int) -> None:
        _user32.MoveWindow(hwnd, 0, 0, width, height, True)


def _apply_time(model: "OpenSimModel", data: MotionData, time: float) -> None:
    for name, value in data.values_at(time).items():
        if not model.coordinates.contains(name):
            continue
        coordinate = model.coordinate(name)
        if coordinate.get_locked():
            continue
        coordinate.setValue(model.state, value, False)
    model.model.realizePosition(model.state)
    model.visualizer.show(model.state)


class PlayerWindow:
    """Handle to a running playback/status window (see :func:`start_player`).

    Attributes
    ----------
    motion_player : MotionPlayer or None
        The playback state machine, if a motion was loaded; ``None`` if
        the window was opened without one (controls shown disabled).
    """

    def __init__(self, motion_player: MotionPlayer | None, stop_event: threading.Event) -> None:
        self.motion_player = motion_player
        self._stop_event = stop_event

    def close(self) -> None:
        """Close the window and stop its background thread.

        Safe to call from any thread; idempotent. The window closes on
        its own next tick (at most one frame interval later), since
        Tk widgets may only safely be touched from the thread running
        their own ``mainloop``.
        """
        self._stop_event.set()


def start_player(
    model: "OpenSimModel", motion: Any = None, *, loop: bool = False, fps: float = 30.0
) -> PlayerWindow:
    """Open the unified viewer+playback window for ``model``.

    Also constructs ``model``'s :class:`~opensim_models._vtk_visualizer.VTKVisualizer`
    (setting :attr:`~opensim_models.model.OpenSimModel.visualizer`) -- on
    the same background thread this starts for the window itself, rather
    than on the caller's thread, so that one thread both creates the VTK
    window and is the one later embedding/resizing/pumping it. A Win32
    window may only safely be manipulated by the thread that created it;
    constructing it here instead of in :meth:`~opensim_models.model.OpenSimModel.show`
    avoids a cross-thread deadlock in the embedding step confirmed by
    direct testing (``SetParent``/``ShowWindow`` block forever waiting for
    the *creating* thread to pump messages it was never going to pump,
    since that thread -- the caller's -- had already moved on).

    Runs its own ``Tk`` event loop on that same background daemon thread,
    so this call returns once that window is up (or construction failed);
    call :meth:`PlayerWindow.close` to shut it down (e.g. before opening a
    new one for the same model). Only that background thread should touch
    ``model`` (its ``state``, coordinates, visualizer) while the window is
    open -- OpenSim's ``State``/``Model`` are not thread-safe, so driving
    the same model concurrently from the caller's own thread is not safe.

    Parameters
    ----------
    model : OpenSimModel
        Model to show/animate; any previous :attr:`~opensim_models.model.OpenSimModel.visualizer`
        should already be closed (see :meth:`~opensim_models.model.OpenSimModel.show`).
    motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, optional
        A motion file path (``.mot``/``.sto``), or an already-loaded/built
        table (e.g. one written by a prior analysis) -- see
        :class:`MotionData`. When ``None`` (default), the window still
        opens (with its status bar live), but the playback controls are
        shown disabled, since there is nothing to play.
    loop : bool, optional
        Initial state of the cycle/loop toggle. Ignored if ``motion`` is
        ``None``. Defaults to ``False``.
    fps : float, optional
        Target refresh rate for advancing playback, refreshing the status
        bar and redrawing the visualizer, in frames per second. Defaults
        to ``30.0``.

    Returns
    -------
    PlayerWindow
        Handle exposing the playback state machine (``.motion_player``,
        ``None`` if ``motion`` was ``None``) and a way to close the window
        (``.close()``).

    Raises
    ------
    RuntimeError
        If the 3D visualizer could not be constructed (e.g. ``vtk`` is not
        installed, or no graphical backend is available).
    ValueError
        If ``motion`` resolves to fewer than 2 rows.
    """
    if _IS_WINDOWS:
        # keeps the embedded window's MoveWindow calls and Tk's own
        # winfo_width/height agreeing on the same (unscaled) pixel
        # coordinates; without this, the embedded 3D view would drift out
        # of sync with its frame on a display with Windows display scaling.
        _user32.SetProcessDPIAware()

    data: MotionData | None = None
    player: MotionPlayer | None = None
    if motion is not None:
        opensim = model.opensim
        table = opensim.TimeSeriesTable(str(motion)) if isinstance(motion, (str, Path)) else motion
        data = MotionData(model, table)
        player = MotionPlayer(data.start_time, data.end_time, loop=loop)

    stop_event = threading.Event()
    ready = threading.Event()
    failure: dict[str, Exception] = {}

    def run() -> None:
        import tkinter as tk
        from tkinter import ttk

        from ._vtk_visualizer import DEFAULT_SIZE, VTKVisualizer

        root = tk.Tk()
        root.title(model.model.getName() or "opensim-models")

        default_width, default_height = DEFAULT_SIZE
        viewer_frame = tk.Frame(root, bg="black", width=default_width, height=default_height)
        viewer_frame.pack(side="top", fill="both", expand=True)
        # has no Tk-managed children (the embedded VTK window is a raw
        # Win32 child, invisible to Tk's own geometry management) -- lock
        # in the explicit size above instead of collapsing to one that fits
        # (empty) content, until <Configure> (bound once VTKVisualizer
        # embeds into this frame, below) takes over sizing it from then on.
        viewer_frame.pack_propagate(False)

        controls_frame = tk.Frame(root)
        controls_frame.pack(side="bottom", fill="x")

        slider_var = tk.DoubleVar(value=0.0)
        time_var = tk.StringVar(value=_NO_MOTION_TEXT)
        hover_var = tk.StringVar(value="")
        # guards the slider's own `command` callback (fired on *any* value
        # change, including the programmatic ones `tick` below makes every
        # frame) so only an actual user drag seeks playback
        user_is_dragging = {"value": False}

        def on_slider_drag(value: str) -> None:
            if user_is_dragging["value"] and player is not None:
                player.seek_fraction(float(value) / 1000.0)
                _apply_time(model, data, player.time)

        slider = ttk.Scale(
            controls_frame, from_=0, to=1000, orient="horizontal",
            variable=slider_var, command=on_slider_drag, length=360,
        )
        slider.bind("<Button-1>", lambda event: user_is_dragging.update(value=True))
        slider.bind("<ButtonRelease-1>", lambda event: user_is_dragging.update(value=False))
        slider.grid(row=0, column=0, columnspan=6, padx=8, pady=(8, 2), sticky="ew")
        controls_frame.columnconfigure(tuple(range(6)), weight=1)

        ttk.Label(controls_frame, textvariable=time_var).grid(
            row=1, column=0, columnspan=6, pady=(0, 6)
        )

        def do_play_pause() -> None:
            player.play_pause()
            play_button.config(text="Pause" if player.playing else "Play")

        def do_stop() -> None:
            player.stop()
            play_button.config(text="Play")
            _apply_time(model, data, player.time)

        def do_fast_forward() -> None:
            player.fast_forward()
            play_button.config(text="Pause")

        def do_fast_backward() -> None:
            player.fast_backward()
            play_button.config(text="Pause")

        cycle_var = tk.BooleanVar(value=player.loop if player is not None else loop)

        def do_toggle_loop() -> None:
            player.toggle_loop()
            cycle_var.set(player.loop)

        rew_button = ttk.Button(controls_frame, text="◄◄", width=4, command=do_fast_backward)
        rew_button.grid(row=2, column=0, padx=4, pady=(0, 8))
        play_button = ttk.Button(controls_frame, text="Play", width=6, command=do_play_pause)
        play_button.grid(row=2, column=1, padx=4, pady=(0, 8))
        stop_button = ttk.Button(controls_frame, text="Stop", width=6, command=do_stop)
        stop_button.grid(row=2, column=2, padx=4, pady=(0, 8))
        ff_button = ttk.Button(controls_frame, text="►►", width=4, command=do_fast_forward)
        ff_button.grid(row=2, column=3, padx=4, pady=(0, 8))
        cycle_check = ttk.Checkbutton(
            controls_frame, text="Cycle", width=6, variable=cycle_var, command=do_toggle_loop,
        )
        cycle_check.grid(row=2, column=4, padx=4, pady=(0, 8))

        # Matches vtkActor's own default visibility (True), so this needs
        # no sync against the visualizer at startup -- ground_check's
        # `command` only reads `model.visualizer` later, once a click can
        # actually happen (construction has long since finished by then).
        ground_var = tk.BooleanVar(value=True)

        def do_toggle_ground() -> None:
            model.visualizer.set_ground_visible(ground_var.get())

        ground_check = ttk.Checkbutton(
            controls_frame, text="Ground", width=6, variable=ground_var, command=do_toggle_ground,
        )
        ground_check.grid(row=2, column=5, padx=4, pady=(0, 8))

        playback_widgets = (slider, rew_button, play_button, stop_button, ff_button, cycle_check)
        if player is None:
            for widget in playback_widgets:
                widget.config(state="disabled")

        separator = ttk.Separator(controls_frame, orient="horizontal")
        separator.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        status_bar = tk.Label(
            controls_frame, textvariable=hover_var, anchor="w", bg="#222222", fg="white",
        )
        status_bar.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 0))

        # winfo_id() alone realizes viewer_frame's native window without
        # pumping Tk's event queue, unlike update()/update_idletasks() --
        # calling either of those *before* constructing VTKVisualizer
        # (next) was tried and rejected, confirmed directly to hang
        # forever inside vtkRenderWindowInteractor.Initialize(), seemingly
        # from some conflict with Tk's own event-loop setup already having
        # touched this thread's message queue first.
        hwnd_box: dict[str, int] = {}
        embed = None
        if _IS_WINDOWS:
            parent_hwnd = viewer_frame.winfo_id()

            def embed(window_id: Any) -> None:
                # called by VTKVisualizer.__init__ itself, between its
                # window's creation and its first paint. Only reparents/
                # restyles here, deliberately not also sizing it (below,
                # once construction is done) in the same call: an
                # immediate `MoveWindow` forces a synchronous repaint
                # before this window's *first* real `Render()` -- still a
                # couple lines away at this point, inside
                # `VTKVisualizer.__init__`.
                hwnd = _hwnd_from_vtk_window_id(window_id)
                hwnd_box["hwnd"] = hwnd
                _embed_visualizer_window(hwnd, parent_hwnd)

        try:
            model._visualizer = VTKVisualizer(model, embed=embed)
        except Exception as error:  # noqa: BLE001 -- reported via `failure`, not swallowed
            failure["error"] = error
            ready.set()
            root.destroy()
            return

        root.update_idletasks()
        root.geometry(f"{default_width}x{default_height + controls_frame.winfo_reqheight()}")
        root.update()

        if "hwnd" in hwnd_box:
            hwnd = hwnd_box["hwnd"]
            _resize_embedded_window(hwnd, viewer_frame.winfo_width(), viewer_frame.winfo_height())
            model.visualizer.resize(viewer_frame.winfo_width(), viewer_frame.winfo_height())

            def on_viewer_configure(event: Any) -> None:
                if event.width > 1 and event.height > 1:
                    _resize_embedded_window(hwnd, event.width, event.height)
                    model.visualizer.resize(event.width, event.height)

            viewer_frame.bind("<Configure>", on_viewer_configure)

        last_tick = _time.monotonic()

        def tick() -> None:
            if stop_event.is_set():
                root.destroy()
                return
            nonlocal last_tick
            now = _time.monotonic()
            dt = now - last_tick
            last_tick = now
            model.visualizer.process_events()
            if player is not None:
                if player.playing:
                    player.advance(dt)
                    _apply_time(model, data, player.time)
                    if not player.playing:
                        play_button.config(text="Play")
                slider_var.set(player.fraction * 1000.0)
                time_var.set(
                    f"{player.time - player.start_time:6.2f} / "
                    f"{player.end_time - player.start_time:.2f} s  ({player.speed:+.0f}x)"
                )
            hover_var.set(model.visualizer.hover_text() or " ")
            root.after(max(1, round(1000.0 / fps)), tick)

        if player is not None:
            _apply_time(model, data, player.time)
        tick()
        ready.set()
        root.mainloop()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(timeout=10.0)
    if "error" in failure:
        raise RuntimeError(
            "Unable to open the 3D visualizer. Check that 'vtk' is "
            "installed (pip install vtk) and that a graphical backend "
            "is available."
        ) from failure["error"]
    return PlayerWindow(player, stop_event)
