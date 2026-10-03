"""The interactive viewer+playback window for :meth:`~opensim_models.model.OpenSimModel.show`.

Internal to the package (the leading underscore on this subpackage, same
convention as every other ``_*`` module at the top level) -- organized one
file per concern:

- ``visualizer.py`` -- the VTK-based 3D view (``VTKVisualizer``).
- ``player.py`` -- the Tk window tying it together with playback controls
  (``start_player``, ``PlayerWindow``, ``MotionData``, ``MotionPlayer``).
- ``tooltip.py`` -- the rounded-corner hover tooltip widget.
- ``export.py`` -- saving the current view as a PNG, or a loaded motion as
  an MP4.
- ``win32_embed.py`` -- the Windows-specific native-window embedding used
  to put the 3D view inside the Tk window instead of a separate one.

:meth:`~opensim_models.model.OpenSimModel.show` is the only public entry
point into any of this -- nothing here is meant to be imported directly by
package consumers.
"""

from __future__ import annotations
