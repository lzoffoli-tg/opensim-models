from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _ensure_visualizer_dll_path() -> None:
    """Make the conda ``Library/bin`` DLLs visible to the visualizer process.

    On Windows, ``simbody-visualizer.exe`` depends on native DLLs (Simbody,
    freeglut, etc.) that live in the active environment's ``Library/bin``
    directory. When Python is launched without going through conda's shell
    activation, that directory is missing from ``PATH`` and the visualizer
    process fails to start; OpenSim then reports this as a generic
    ``initSystem`` exception with no indication that a DLL was involved.
    """
    if not sys.platform.startswith("win"):
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(library_bin) not in path_entries:
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")


class OpensimViewer:
    """Display an OpenSim model with the native Simbody visualizer.

    Parameters
    ----------
    model : opensim.Model or OpensimUser
        Model to display, or an ``OpensimUser`` containing one.
    state : opensim.State or None, optional
        State to display. When omitted, the user's current state is used when
        available; otherwise the model is initialized by ``show``.
    geometry_path : str or pathlib.Path or None, optional
        Additional directory containing VTP geometry files, e.g. the
        repository's ``assets/meshes`` directory. This is useful for models
        created outside the repository.

    Raises
    ------
    TypeError
        If ``model`` does not expose the OpenSim model API required by the
        viewer.
    """

    def __init__(
        self,
        model: Any,
        state: Any | None = None,
        geometry_path: str | Path | None = None,
    ) -> None:
        """Create a viewer without opening a window.

        Parameters
        ----------
        model : opensim.Model or OpensimUser
            Model to display, or a user containing a model.
        state : opensim.State or None, optional
            State to display when the window is opened.
        geometry_path : str or pathlib.Path or None, optional
            Extra geometry search directory.
        """
        self._source = model
        self._model = getattr(model, "model", model)
        self._state = state if state is not None else getattr(model, "state", None)
        self._geometry_path = Path(geometry_path) if geometry_path else None
        self._visualizer: Any | None = None
        if not all(
            hasattr(self._model, attribute)
            for attribute in ("setUseVisualizer", "initSystem", "getVisualizer")
        ):
            raise TypeError("model must be an OpenSim Model or an OpensimUser")

    @property
    def model(self) -> Any:
        """Return the OpenSim model displayed by this viewer."""
        return self._model

    @property
    def state(self) -> Any | None:
        """Return the state that will be displayed, if initialized."""
        return self._state

    @property
    def visualizer(self) -> Any | None:
        """Return the native OpenSim visualizer after :meth:`show` starts it."""
        return self._visualizer

    def show(self) -> None:
        """Open a native Simbody window showing the current model state.

        The method initializes the OpenSim visualizer lazily. If enabling the
        visualizer requires a new state, coordinate values from the supplied
        state are copied to the new state so a pose configured on an
        ``OpensimUser`` is preserved.
        Raises
        ------
        RuntimeError
            If OpenSim cannot create or expose its visualizer.
        """
        _ensure_visualizer_dll_path()
        coordinates = self._capture_coordinate_values()
        self._model.setUseVisualizer(True)
        try:
            self._state = self._model.initSystem()
            self._visualizer = self._model.getVisualizer()
            if self._geometry_path is not None:
                self._visualizer.addDirToGeometrySearchPaths(
                    str(self._geometry_path.resolve())
                )
            self._restore_coordinate_values(coordinates)
            self._visualizer.show(self._state)
        except Exception as error:
            raise RuntimeError(
                "Unable to open the OpenSim visualizer. Check the graphical "
                "backend and the model geometry files. On Windows, this can "
                "also happen if the active environment's 'Library/bin' "
                "directory (containing the Simbody/OpenGL DLLs) is missing "
                "from PATH."
            ) from error

    def _capture_coordinate_values(self) -> dict[str, float]:
        if self._state is None:
            return {}
        coordinates = self._model.getCoordinateSet()
        return {
            coordinates.get(index)
            .getName(): coordinates.get(index)
            .getValue(self._state)
            for index in range(coordinates.getSize())
        }

    def _restore_coordinate_values(self, values: dict[str, float]) -> None:
        if not values:
            return
        coordinates = self._model.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            value = values[coordinate.getName()]
            if not coordinate.get_locked():
                coordinate.setValue(self._state, value, False)
        self._model.realizePosition(self._state)
