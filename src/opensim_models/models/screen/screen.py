"""A parametric, plexiglass display-panel model."""

from __future__ import annotations

import math
from pathlib import Path

from ...model import OpenSimModel

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MESHES_DIR = _ASSETS_DIR / "meshes"
_MESH_FILENAME = "screen_panel.stl"

_THICKNESS_MM = 1.0
# PMMA ("plexiglass"/acrylic glass) density.
_PLEXIGLASS_DENSITY_KG_M3 = 1180.0


def _parse_ratio(ratio: str) -> tuple[float, float]:
    """Parse a ``"width:height"`` aspect ratio string into its two components."""
    parts = ratio.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid aspect ratio {ratio!r}; expected e.g. '16:9'")
    try:
        width_ratio, height_ratio = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid aspect ratio {ratio!r}; expected e.g. '16:9'") from error
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"Invalid aspect ratio {ratio!r}; both sides must be positive")
    return width_ratio, height_ratio


def _size_from_diagonal(inches: float, ratio: str) -> tuple[float, float]:
    """Return ``(width_mm, height_mm)`` for a diagonal size and aspect ratio."""
    width_ratio, height_ratio = _parse_ratio(ratio)
    diagonal_mm = inches * 25.4
    scale = diagonal_mm / math.hypot(width_ratio, height_ratio)
    return width_ratio * scale, height_ratio * scale


def _write_box_mesh(
    destination_path: Path, width_mm: float, height_mm: float, thickness_mm: float
) -> None:
    """Write a watertight, axis-aligned box STL (metres), centred on the origin.

    Width, height and thickness run along the local X, Y and Z axes
    respectively, matching the ``mass_center = (0, 0, 0)`` convention used
    for the ``opensim.Body`` this mesh is attached to.
    """
    hx, hy, hz = width_mm / 2000.0, height_mm / 2000.0, thickness_mm / 2000.0

    def corner(signs: tuple[int, int, int]) -> tuple[float, float, float]:
        sx, sy, sz = signs
        return (sx * hx, sy * hy, sz * hz)

    # Each face is a quad, its 4 corners listed counter-clockwise when
    # viewed from outside the box (right-hand rule around ``normal``).
    faces = [
        ((1.0, 0.0, 0.0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((-1.0, 0.0, 0.0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
        ((0.0, 1.0, 0.0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
        ((0.0, -1.0, 0.0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ((0.0, 0.0, 1.0), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0.0, 0.0, -1.0), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
    ]

    lines = ["solid screen_panel"]
    for normal, quad in faces:
        points = [corner(signs) for signs in quad]
        for triangle in ((points[0], points[1], points[2]), (points[0], points[2], points[3])):
            lines.append(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}")
            lines.append("    outer loop")
            for vertex in triangle:
                lines.append(f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append("endsolid screen_panel")
    destination_path.write_text("\n".join(lines) + "\n")


class Screen(OpenSimModel):
    """A rigid, 1 mm-thick plexiglass display panel.

    The panel is a single ``opensim.Body`` ("screen_panel") welded to ground
    at (``center_x``, ``center_y``, ``center_z``) and tilted about ground's
    X axis by ``angle_deg``: ``0`` lies flat on the ground, ``90`` (default)
    stands upright, as a display normally sits.

    Its size comes from ``width_mm``/``height_mm`` once *both* are set;
    otherwise (including the default, where both are ``None``) it is
    derived from a diagonal size in ``inches`` split by ``ratio``
    (``"width:height"``, e.g. ``"16:9"``). This priority is re-evaluated on
    every setter call, so setting ``width_mm`` and ``height_mm`` one at a
    time works: the panel stays sized from the diagonal until both are set.

    Mass and inertia are derived from the panel's volume assuming a
    plexiglass (PMMA) density, and a matching box mesh is (re)generated and
    written to ``assets/meshes/screen_panel.stl`` next to this module
    whenever the panel's dimensions change.

    Parameters
    ----------
    width_mm, height_mm : float or None, optional
        Explicit panel size in millimetres, used once both are set;
        otherwise the panel is sized from ``inches``/``ratio``.
    inches : float or None, optional
        Diagonal size in inches, used when ``width_mm``/``height_mm`` are
        not both given. Defaults to ``22``.
    ratio : str or None, optional
        Aspect ratio as ``"width:height"`` (e.g. ``"16:9"``), used together
        with ``inches``. Defaults to ``"16:9"``.
    center_x, center_y, center_z : float, optional
        Panel centre in ground coordinates (metres). Defaults to ``0.0``.
    angle_deg : float, optional
        Panel inclination relative to the ground, in degrees: ``0`` lies
        flat, ``90`` (default) stands upright.

    Raises
    ------
    ValueError
        If neither sizing method is usable (``width_mm``/``height_mm`` are
        not both given, and ``inches``/``ratio`` are not both given), or if
        ``ratio`` is not a valid ``"width:height"`` string.
    """

    def __init__(
        self,
        width_mm: float | None = None,
        height_mm: float | None = None,
        inches: float | None = 22,
        ratio: str | None = "16:9",
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        angle_deg: float = 90,
    ) -> None:
        """Build the panel body/joint from the given size and pose parameters."""
        super().__init__(model_path=None)
        _MESHES_DIR.mkdir(parents=True, exist_ok=True)
        self.add_geometry_directory(_MESHES_DIR)
        self.opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(_MESHES_DIR.resolve()))

        self._width_mm = width_mm
        self._height_mm = height_mm
        self._inches = inches
        self._ratio = ratio
        self._center_x = center_x
        self._center_y = center_y
        self._center_z = center_z
        self._angle_deg = angle_deg
        self._rebuild()

    @property
    def width_mm(self) -> float | None:
        """Explicit panel width in millimetres, or ``None`` if sized from the diagonal."""
        return self._width_mm

    def set_width_mm(self, width_mm: float | None) -> None:
        """Set the explicit panel width in millimetres and rebuild the panel."""
        self._width_mm = width_mm
        self._rebuild()

    @property
    def height_mm(self) -> float | None:
        """Explicit panel height in millimetres, or ``None`` if sized from the diagonal."""
        return self._height_mm

    def set_height_mm(self, height_mm: float | None) -> None:
        """Set the explicit panel height in millimetres and rebuild the panel."""
        self._height_mm = height_mm
        self._rebuild()

    @property
    def inches(self) -> float | None:
        """Diagonal size in inches (only used when ``width_mm``/``height_mm`` are ``None``)."""
        return self._inches

    def set_inches(self, inches: float | None) -> None:
        """Set the diagonal size in inches and rebuild the panel."""
        self._inches = inches
        self._rebuild()

    @property
    def ratio(self) -> str | None:
        """Aspect ratio ``"width:height"`` (only used when ``width_mm``/``height_mm`` are ``None``)."""
        return self._ratio

    def set_ratio(self, ratio: str | None) -> None:
        """Set the aspect ratio (``"width:height"``, e.g. ``"16:9"``) and rebuild the panel."""
        self._ratio = ratio
        self._rebuild()

    @property
    def center_x(self) -> float:
        """Panel centre X coordinate in ground, in metres."""
        return self._center_x

    def set_center_x(self, center_x: float) -> None:
        """Set the panel centre X coordinate in ground (metres) and rebuild the panel."""
        self._center_x = center_x
        self._rebuild()

    @property
    def center_y(self) -> float:
        """Panel centre Y coordinate in ground, in metres."""
        return self._center_y

    def set_center_y(self, center_y: float) -> None:
        """Set the panel centre Y coordinate in ground (metres) and rebuild the panel."""
        self._center_y = center_y
        self._rebuild()

    @property
    def center_z(self) -> float:
        """Panel centre Z coordinate in ground, in metres."""
        return self._center_z

    def set_center_z(self, center_z: float) -> None:
        """Set the panel centre Z coordinate in ground (metres) and rebuild the panel."""
        self._center_z = center_z
        self._rebuild()

    @property
    def angle_deg(self) -> float:
        """Panel inclination relative to the ground, in degrees (0 = flat, 90 = upright)."""
        return self._angle_deg

    def set_angle_deg(self, angle_deg: float) -> None:
        """Set the panel inclination relative to the ground (degrees) and rebuild the panel."""
        self._angle_deg = angle_deg
        self._rebuild()

    def _resolve_size_mm(self) -> tuple[float, float]:
        """Return the effective (width_mm, height_mm), applying the sizing priority.

        Explicit ``width_mm``/``height_mm`` win only once *both* are set;
        otherwise the diagonal (``inches``/``ratio``) is used. This keeps
        every individual setter usable on its own -- e.g. calling
        ``set_width_mm`` then ``set_height_mm`` in sequence never hits an
        invalid in-between state.
        """
        if self._width_mm is not None and self._height_mm is not None:
            return self._width_mm, self._height_mm
        if self._inches is None or self._ratio is None:
            raise ValueError(
                "inches and ratio are required when width_mm/height_mm are not both given"
            )
        return _size_from_diagonal(self._inches, self._ratio)

    def _rebuild(self) -> None:
        """Recreate the panel body, mesh and joint from the current parameters."""
        width_mm, height_mm = self._resolve_size_mm()

        width_m = width_mm / 1000.0
        height_m = height_mm / 1000.0
        thickness_m = _THICKNESS_MM / 1000.0
        volume = width_m * height_m * thickness_m
        mass = volume * _PLEXIGLASS_DENSITY_KG_M3
        inertia = (
            mass * (height_m**2 + thickness_m**2) / 12.0,
            mass * (width_m**2 + thickness_m**2) / 12.0,
            mass * (width_m**2 + height_m**2) / 12.0,
        )

        mesh_path = _MESHES_DIR / _MESH_FILENAME
        _write_box_mesh(mesh_path, width_mm, height_mm, _THICKNESS_MM)

        self.model = self.opensim.Model()
        body = self.opensim.Body(
            "screen_panel",
            mass,
            self.opensim.Vec3(0, 0, 0),
            self.opensim.Inertia(*inertia),
        )
        self.model.addBody(body)
        body.attachGeometry(self.opensim.Mesh(_MESH_FILENAME))

        joint = self.opensim.WeldJoint(
            "screen_panel_joint",
            self.model.getGround(),
            self.opensim.Vec3(self._center_x, self._center_y, self._center_z),
            self.opensim.Vec3(math.radians(self._angle_deg - 90.0), 0.0, 0.0),
            body,
            self.opensim.Vec3(0, 0, 0),
            self.opensim.Vec3(0, 0, 0),
        )
        self.model.addJoint(joint)

        self.model.finalizeConnections()
        self.state = self.model.initSystem()
        self._visualizer = None
