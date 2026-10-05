"""Public, off-screen animation export for OpenSim motion tables."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .model import OpenSimModel, import_opensim

__all__ = ["save_animation"]


def save_animation(
    model: Any,
    sto_table: Any,
    path: str | Path,
    *,
    size: tuple[int, int] = (900, 700),
) -> Path:
    """Render an OpenSim motion table to an MP4 without opening a window.

    Parameters
    ----------
    model : OpenSimModel or opensim.Model
        Model whose coordinates are driven by the motion. Mesh geometry
        should be resolvable through the model's registered geometry paths
        (for an ``OpenSimModel``) or OpenSim's geometry search paths (for an
        ``opensim.Model``).
    sto_table : str, pathlib.Path or opensim.TimeSeriesTable
        Path to a ``.sto`` file or a loaded OpenSim table. Columns named
        exactly like model coordinates, or ending in
        ``/<coordinate-name>/value`` (as in states tables), are applied.
        Other state columns, such as speeds and muscle values, are ignored.
        Rotational values are converted from degrees only when the table's
        ``inDegrees`` metadata is ``yes``.
    path : str or pathlib.Path
        Destination ``.mp4`` path. Existing files are overwritten; parent
        directories are created when needed.
    size : tuple[int, int], optional
        Output frame dimensions in pixels, as an even ``(width, height)``
        pair. Even dimensions are required by the H.264 pixel format.
        Defaults to ``(900, 700)``.

    Returns
    -------
    pathlib.Path
        The destination path.

    Notes
    -----
    The constant video frame rate is inferred from the average time step
    in the table. The motion is sampled uniformly over its full time range;
    for uniformly sampled input this applies every stored pose exactly.
    The model's coordinate values are restored after rendering.

    Raises
    ------
    ValueError
        If the table has fewer than two samples, invalid times, no matching
        coordinate columns, an invalid output size, or a non-``.mp4`` path.
    """
    opensim = import_opensim()
    if not isinstance(model, (OpenSimModel, opensim.Model)):
        raise TypeError("model must be an OpenSimModel or opensim.Model")
    if isinstance(model, OpenSimModel):
        raw_model = model.model
        state = model.state
        visualizer_model = model
    else:
        raw_model = model
        state = raw_model.initSystem()

        def resolve_geometry_file(filename: str) -> Path | None:
            try:
                resolved = opensim.ModelVisualizer.findGeometryFile(filename)
            except RuntimeError:
                return None
            return Path(str(resolved)) if resolved else None

        visualizer_model = SimpleNamespace(
            model=raw_model,
            state=state,
            opensim=opensim,
            _resolve_geometry_file=resolve_geometry_file,
        )

    if isinstance(sto_table, (str, Path)):
        table = opensim.TimeSeriesTable(str(sto_table))
    elif isinstance(sto_table, opensim.TimeSeriesTable):
        table = sto_table
    else:
        raise TypeError("sto_table must be a path or opensim.TimeSeriesTable")

    times = np.asarray(table.getIndependentColumn(), dtype=float)
    if times.size < 2:
        raise ValueError("motion table must have at least 2 rows")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("motion times must be finite and strictly increasing")

    try:
        width, height = size
    except (TypeError, ValueError) as error:
        raise ValueError("size must be a (width, height) pair of positive integers") from error
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or width <= 0
        or height <= 0
    ):
        raise ValueError("size must be a (width, height) pair of positive integers")
    if width % 2 or height % 2:
        raise ValueError("size width and height must both be even for H.264 video")

    coordinates = raw_model.getCoordinateSet()
    coordinate_names = {
        coordinates.get(index).getName()
        for index in range(coordinates.getSize())
    }
    columns: dict[str, np.ndarray] = {}
    for raw_label in table.getColumnLabels():
        label = str(raw_label)
        if label in coordinate_names:
            coordinate_name = label
        elif label.endswith("/value"):
            coordinate_name = label.rsplit("/", 2)[-2]
        else:
            continue
        if coordinate_name not in coordinate_names:
            continue
        values = np.asarray(table.getDependentColumn(label).to_numpy(), dtype=float)
        if values.size != times.size or not np.all(np.isfinite(values)):
            raise ValueError(f"coordinate column {label!r} must contain finite values")
        columns[coordinate_name] = values

    if not columns:
        raise ValueError("motion table contains no columns matching model coordinates")

    in_degrees = (
        table.hasTableMetaDataKey("inDegrees")
        and str(table.getTableMetaDataAsString("inDegrees")).strip().lower() == "yes"
    )
    rotational_motion_type = 1
    for name, values in columns.items():
        coordinate = coordinates.get(name)
        if in_degrees and coordinate.getMotionType() == rotational_motion_type:
            columns[name] = np.radians(values)

    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("path must have an .mp4 extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = float(times[-1] - times[0])
    frame_count = times.size - 1
    fps = frame_count / duration
    sample_times = np.linspace(times[0], times[-1], frame_count + 1)
    original_values = {
        name: coordinates.get(name).getValue(state)
        for name in columns
        if not coordinates.get(name).get_locked()
    }

    from ._gui.visualizer import VTKVisualizer
    import imageio

    visualizer = VTKVisualizer(
        visualizer_model,
        offscreen=True,
        size=(int(width), int(height)),
    )
    writer = None
    try:
        writer = imageio.get_writer(
            str(destination), fps=fps, codec="libx264", quality=8,
            macro_block_size=1,
        )
        for time_s in sample_times:
            for name, values in columns.items():
                coordinate = coordinates.get(name)
                if not coordinate.get_locked():
                    coordinate.setValue(
                        state,
                        float(np.interp(time_s, times, values)),
                        False,
                    )
            raw_model.realizePosition(state)
            visualizer.show(state)
            writer.append_data(visualizer.capture_frame())
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                for name, value in original_values.items():
                    coordinates.get(name).setValue(state, value, False)
                raw_model.realizePosition(state)
            finally:
                visualizer.close()

    return destination
