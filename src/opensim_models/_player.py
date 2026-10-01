"""Playback controls for a motion, driving an ``OpenSimModel``'s visualizer.

Used by :meth:`~opensim_models.model.OpenSimModel.show` when given a
``motion``: opens a small Tk control window (play/pause, stop, fast-forward,
fast-backward, a loop/cycle toggle, and a draggable progress slider)
alongside the native Simbody visualizer window, and drives that visualizer
by repeatedly calling ``ModelVisualizer.show(state)`` as playback advances.

A companion Tk window, rather than widgets added directly to the native
Simbody visualizer: that visualizer's own interactive widgets
(``Visualizer.addSlider``/``addMenu``/``setWindowTitle`` -- anything taking
a ``SimTK::String``) are not callable from Python on at least some current
OpenSim builds. Passing a plain ``str`` raises ``TypeError: ... argument
... of type 'String const &'`` regardless of what is actually passed or in
what order -- a native-binding limitation of the installed OpenSim
package, unrelated to this one, confirmed to affect even the unrelated
``setWindowTitle``. A separate Tk window sidesteps it entirely, while still
driving the same visualizer through ``ModelVisualizer.show(state)``, which
takes no ``SimTK::String`` and is already used by ``OpenSimModel.show``.
"""

from __future__ import annotations

import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = ["MotionData", "MotionPlayer", "start_player"]

# Ordered, signed multiples of real time used by fast-forward/fast-backward:
# repeatedly clicking one steps outward (slower -> faster) in that
# direction; there is no "0x" entry since that would just be pause.
_SPEED_STEPS = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
_NEUTRAL_SPEED_INDEX = _SPEED_STEPS.index(1.0)

# OpenSim's own Coordinate::MotionType C++ enum (Rotational=1): not exposed
# as a named constant through these Python bindings, so used as a literal.
_ROTATIONAL_MOTION_TYPE = 1


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
# Docking the control window under the native Simbody visualizer window
# ---------------------------------------------------------------------------
#
# Windows-only: the Simbody visualizer is a separate native window owned by
# a different process (a bundled "simbody-visualizer" executable) -- not
# something Tk or OpenSim's own Python API can query or reposition.
# ctypes + the raw Win32 API is the only way to find and track it; there is
# no equivalent on other platforms, so docking is simply unavailable there
# and the control window is left wherever Tk places it by default.

_IS_WINDOWS = sys.platform.startswith("win")
_PLAYER_HEIGHT = 130  # px; the control window's fixed height


class _Rect(NamedTuple):
    x: int
    y: int
    width: int
    height: int


if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _GA_ROOT = 2  # GetAncestor flag: the root owner window

    class _WinRect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _find_visualizer_window() -> int | None:
        """Find the Simbody visualizer's native window handle, if one is open.

        Matched by window class (FreeGLUT windows are always class
        ``"GLUT"``) and title prefix (OpenSim's ``ModelVisualizer`` titles
        it ``"OpenSim <version>: ..."`` -- the rest varies by version and
        model, so only that stable prefix is matched). When several such
        windows are open at once (more than one model shown), the
        foreground one is preferred, since showing/raising a visualizer
        brings it to the front; this cannot otherwise tell which
        visualizer belongs to which `OpenSimModel`, since the Python API
        does not expose the native window handle or the underlying
        subprocess's id directly.
        """
        candidates: list[int] = []

        def callback(hwnd: int, _lparam: int) -> bool:
            if not _user32.IsWindowVisible(hwnd):
                return True
            length = _user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            title = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, title, length + 1)
            if not title.value.startswith("OpenSim"):
                return True
            class_name = ctypes.create_unicode_buffer(64)
            _user32.GetClassNameW(hwnd, class_name, 64)
            if class_name.value == "GLUT":
                candidates.append(hwnd)
            return True

        _user32.EnumWindows(_WNDENUMPROC(callback), 0)
        if not candidates:
            return None
        foreground = _user32.GetForegroundWindow()
        return foreground if foreground in candidates else candidates[-1]

    def _get_window_rect(hwnd: int) -> _Rect:
        rect = _WinRect()
        _user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return _Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)

    def _move_window(hwnd: int, rect: _Rect) -> None:
        _user32.MoveWindow(hwnd, rect.x, rect.y, rect.width, rect.height, True)

    def _top_level_hwnd(root: Any) -> int:
        """Return the real, OS-decorated top-level window handle for ``root``.

        ``root.winfo_id()`` is a *child* window sized to the Tk content
        area (its own ``MoveWindow`` coordinates are relative to that
        parent, not the screen), not the outer window ``EnumWindows``
        would find -- walk up to the actual root via ``GetAncestor``.
        """
        return _user32.GetAncestor(root.winfo_id(), _GA_ROOT)

    def _measure_window_borders(root: Any, top_hwnd: int) -> tuple[int, int]:
        """Return the (width, height) overhead of the OS window chrome.

        ``root.geometry("WxH+X+Y")``'s ``W``/``H`` set the Tk *content*
        area, matching ``root.winfo_width()``/``winfo_height()``, but
        ``GetWindowRect`` (used to match the visualizer's size) reports
        the *outer* window, border/title bar included -- compare the two,
        measured at the same instant, to get that fixed, style-dependent
        overhead once, then compensate every subsequent ``geometry()``
        call with it so the two stay pixel-for-pixel aligned.
        """
        outer = _get_window_rect(top_hwnd)
        return (outer.width - root.winfo_width(), outer.height - root.winfo_height())


def _get_tk_window_rect(root: Any) -> _Rect:
    return _Rect(root.winfo_x(), root.winfo_y(), root.winfo_width(), root.winfo_height())


def _dock_player_window(root: Any, visualizer_rect: _Rect, borders: tuple[int, int]) -> None:
    """Position/size ``root`` directly below ``visualizer_rect``, matching its width.

    A single rule, not a special case for "full screen": when the
    visualizer spans the whole screen, its own width already equals the
    screen's, so the control window ends up spanning it too.
    """
    border_width, border_height = borders
    root.geometry(
        f"{visualizer_rect.width - border_width}x{_PLAYER_HEIGHT - border_height}"
        f"+{visualizer_rect.x}+{visualizer_rect.y + visualizer_rect.height}"
    )


class _DockingTracker:
    """Keeps the control window docked under the visualizer window (Windows only).

    Call :meth:`sync` once per tick. Moving/resizing the visualizer
    re-docks the control window under it, matching its width; dragging the
    control window instead moves the visualizer by the same delta (so it
    stays docked below it either way) -- this only tracks *position*
    changes on the control window's side, never its size, since that is
    always fully driven by the visualizer (and the window is built
    non-resizable to begin with).
    """

    def __init__(self) -> None:
        self._hwnd: int | None = None
        self._borders: tuple[int, int] | None = None
        self._last_visualizer_rect: _Rect | None = None
        self._last_player_rect: _Rect | None = None

    def sync(self, root: Any) -> None:
        if not _IS_WINDOWS:
            return

        if self._borders is None:
            self._borders = _measure_window_borders(root, _top_level_hwnd(root))

        if self._hwnd is None or not _user32.IsWindow(self._hwnd):
            self._hwnd = _find_visualizer_window()
            if self._hwnd is None:
                return
            _dock_player_window(root, _get_window_rect(self._hwnd), self._borders)
            root.update_idletasks()
            self._last_visualizer_rect = _get_window_rect(self._hwnd)
            self._last_player_rect = _get_tk_window_rect(root)
            return

        visualizer_rect = _get_window_rect(self._hwnd)
        player_rect = _get_tk_window_rect(root)

        if visualizer_rect != self._last_visualizer_rect:
            _dock_player_window(root, visualizer_rect, self._borders)
        elif (player_rect.x, player_rect.y) != (
            self._last_player_rect.x,
            self._last_player_rect.y,
        ):
            moved_rect = _Rect(
                visualizer_rect.x + (player_rect.x - self._last_player_rect.x),
                visualizer_rect.y + (player_rect.y - self._last_player_rect.y),
                visualizer_rect.width,
                visualizer_rect.height,
            )
            _move_window(self._hwnd, moved_rect)
            visualizer_rect = moved_rect
            _dock_player_window(root, visualizer_rect, self._borders)

        root.update_idletasks()
        self._last_visualizer_rect = visualizer_rect
        self._last_player_rect = _get_tk_window_rect(root)


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


def start_player(
    model: "OpenSimModel", motion: Any, *, loop: bool = False, fps: float = 30.0
) -> MotionPlayer:
    """Open a Tk playback control window driving ``model``'s visualizer.

    Requires :meth:`~opensim_models.model.OpenSimModel.show` to have been
    called already (``model.visualizer`` must be available). Runs its own
    ``Tk`` event loop on a background daemon thread, so this returns
    immediately; the window (and the background thread driving it) closes
    on its own when the user closes it. Only that background thread should
    touch ``model`` (its ``state``, coordinates, visualizer) while the
    player window is open -- OpenSim's ``State``/``Model`` are not
    thread-safe, so driving the same model concurrently from the caller's
    own thread is not safe.

    Parameters
    ----------
    model : OpenSimModel
        Model to animate; must already have an open visualizer (see
        :meth:`~opensim_models.model.OpenSimModel.show`).
    motion : str, pathlib.Path, or opensim.TimeSeriesTable
        A motion file path (``.mot``/``.sto``), or an already-loaded/built
        table (e.g. one written by a prior analysis) -- see
        :class:`MotionData`.
    loop : bool, optional
        Initial state of the cycle/loop toggle. Defaults to ``False``.
    fps : float, optional
        Target refresh rate for advancing playback and redrawing the
        visualizer, in frames per second. Defaults to ``30.0``.

    Returns
    -------
    MotionPlayer
        The playback state machine backing the window, in case the caller
        wants to inspect or drive it programmatically (e.g. in a test).

    Raises
    ------
    RuntimeError
        If ``model`` has no open visualizer yet.
    ValueError
        If ``motion`` resolves to fewer than 2 rows or a non-increasing
        time column (see :class:`MotionData`).
    """
    if model.visualizer is None:
        raise RuntimeError("call model.show() before model.show(motion=...) can animate it")

    if _IS_WINDOWS:
        # keeps GetWindowRect/MoveWindow and Tk's own winfo_x/y/width/height
        # agreeing on the same (unscaled) pixel coordinates; without this,
        # docking would drift on a display with Windows display scaling.
        _user32.SetProcessDPIAware()

    opensim = model.opensim
    table = opensim.TimeSeriesTable(str(motion)) if isinstance(motion, (str, Path)) else motion
    data = MotionData(model, table)
    player = MotionPlayer(data.start_time, data.end_time, loop=loop)
    docking = _DockingTracker()

    ready = threading.Event()

    def run() -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Playback")
        root.resizable(False, False)  # size is always driven by the visualizer, not the user

        slider_var = tk.DoubleVar(value=0.0)
        time_var = tk.StringVar(value="")
        # guards the slider's own `command` callback (fired on *any* value
        # change, including the programmatic ones `tick` below makes every
        # frame) so only an actual user drag seeks playback
        user_is_dragging = {"value": False}

        def on_slider_drag(value: str) -> None:
            if user_is_dragging["value"]:
                player.seek_fraction(float(value) / 1000.0)
                _apply_time(model, data, player.time)

        slider = ttk.Scale(
            root, from_=0, to=1000, orient="horizontal",
            variable=slider_var, command=on_slider_drag, length=360,
        )
        slider.bind("<Button-1>", lambda event: user_is_dragging.update(value=True))
        slider.bind("<ButtonRelease-1>", lambda event: user_is_dragging.update(value=False))
        slider.grid(row=0, column=0, columnspan=5, padx=8, pady=(8, 2), sticky="ew")

        ttk.Label(root, textvariable=time_var).grid(row=1, column=0, columnspan=5, pady=(0, 6))

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

        cycle_var = tk.BooleanVar(value=player.loop)

        def do_toggle_loop() -> None:
            player.toggle_loop()
            cycle_var.set(player.loop)

        rew_button = ttk.Button(root, text="◄◄", width=4, command=do_fast_backward)
        rew_button.grid(row=2, column=0, padx=4, pady=(0, 8))
        play_button = ttk.Button(root, text="Play", width=6, command=do_play_pause)
        play_button.grid(row=2, column=1, padx=4, pady=(0, 8))
        stop_button = ttk.Button(root, text="Stop", width=6, command=do_stop)
        stop_button.grid(row=2, column=2, padx=4, pady=(0, 8))
        ff_button = ttk.Button(root, text="►►", width=4, command=do_fast_forward)
        ff_button.grid(row=2, column=3, padx=4, pady=(0, 8))
        cycle_check = ttk.Checkbutton(
            root, text="Cycle", width=6, variable=cycle_var, command=do_toggle_loop,
        )
        cycle_check.grid(row=2, column=4, padx=4, pady=(0, 8))

        last_tick = _time.monotonic()

        def tick() -> None:
            nonlocal last_tick
            now = _time.monotonic()
            dt = now - last_tick
            last_tick = now
            docking.sync(root)
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
            root.after(max(1, round(1000.0 / fps)), tick)

        root.update_idletasks()  # settle geometry before the first dock/tick
        _apply_time(model, data, player.time)
        tick()
        ready.set()
        root.mainloop()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(timeout=5.0)
    return player
