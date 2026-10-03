"""A parametric, plexiglass display-panel component."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from . import Body
from ..model import OpenSimModel
from ..operators import rotate_object, translate_object

__all__ = ["Screen"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MESHES_DIR = _ASSETS_DIR / "meshes"
_MESH_FILENAME = "screen_panel.stl"

_BODY_NAME = "screen_panel"
_JOINT_NAME = "screen_panel_joint"

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
        raise ValueError(
            f"Invalid aspect ratio {ratio!r}; expected e.g. '16:9'"
        ) from error
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
        for triangle in (
            (points[0], points[1], points[2]),
            (points[0], points[2], points[3]),
        ):
            lines.append(
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}"
            )
            lines.append("    outer loop")
            for vertex in triangle:
                lines.append(
                    f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}"
                )
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append("endsolid screen_panel")
    destination_path.write_text("\n".join(lines) + "\n")


class Screen(Body):
    """A rigid, 1 mm-thick plexiglass display panel -- a single component.

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
    written to ``screen_panel.stl`` inside :attr:`mesh_dir` (defaulting to
    ``assets/meshes`` next to this module) whenever the panel's dimensions
    change.

    Like any other component, a ``Screen`` has no ``show()`` of its own --
    :meth:`~opensim_models.model.OpenSimModel.show` operates on
    containers, never on a single part. A ``Screen`` is self-contained (it
    carries its own private ``OpenSimModel`` internally to do the actual
    OpenSim/state work this needs), but to actually *see* it, add it to a
    container first: ``model + screen`` (or ``screen + model``) returns a
    new, independent ``OpenSimModel`` with the panel's body merged in,
    ready to ``.show()``.

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
    mesh_dir : str, pathlib.Path or None, optional
        Directory the panel mesh (``screen_panel.stl``) is (re)written to,
        created (along with any missing parent) if it doesn't already
        exist. Defaults to ``None``, meaning the package's own bundled
        ``assets/meshes`` folder next to this module -- set this when that
        default location isn't writable (e.g. an admin-owned install) or
        to collect a model's generated meshes somewhere else of your
        choosing.
    name : str or None, optional
        OpenSim name for the panel's body. Defaults to ``None``, meaning
        ``"screen_panel"`` -- equivalent to leaving this unset and calling
        :meth:`set_name` right after construction, except it skips that
        extra rename (and the ``finalizeConnections()`` it triggers). Once
        set, the name sticks across any rebuild (any setter above, or
        :meth:`set_name` itself) and across :meth:`copy`, same as every
        other attribute of this panel.

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
        mesh_dir: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        """Build the panel body/joint from the given size and pose parameters."""
        # A private, never-exposed OpenSimModel -- see the class docstring
        # for how this panel actually becomes visible (`model + screen`).
        self._container = OpenSimModel(model_path=None)
        self._mesh_dir = Path(mesh_dir) if mesh_dir is not None else _MESHES_DIR
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)

        self._width_mm = width_mm
        self._height_mm = height_mm
        self._inches = inches
        self._ratio = ratio
        self._center_x = center_x
        self._center_y = center_y
        self._center_z = center_z
        self._angle_deg = angle_deg
        self._name = name if name is not None else _BODY_NAME
        self._rebuild()

    @property
    def width_mm(self) -> float | None:
        """Explicit panel width in millimetres, or ``None`` if sized from the diagonal."""
        return self._width_mm

    def set_width_mm(self, width_mm: float | None) -> None:
        """Set the explicit panel width in millimetres and rebuild the panel.

        The new size only takes effect once :attr:`height_mm` is also set
        (see :meth:`_resolve_size_mm`'s priority, described in the class
        docstring); until then the panel stays sized from
        :attr:`inches`/:attr:`ratio`.

        Parameters
        ----------
        width_mm : float or None
            New explicit panel width, in millimetres. ``None`` reverts to
            sizing from :attr:`inches`/:attr:`ratio`.

        Raises
        ------
        ValueError
            If, after this change, the panel cannot be sized at all --
            i.e. ``width_mm``/:attr:`height_mm` are not both set, and
            :attr:`inches`/:attr:`ratio` are not both set either.
        """
        self._width_mm = width_mm
        self._rebuild()

    @property
    def height_mm(self) -> float | None:
        """Explicit panel height in millimetres, or ``None`` if sized from the diagonal."""
        return self._height_mm

    def set_height_mm(self, height_mm: float | None) -> None:
        """Set the explicit panel height in millimetres and rebuild the panel.

        The new size only takes effect once :attr:`width_mm` is also set
        (see :meth:`set_width_mm`); until then the panel stays sized from
        :attr:`inches`/:attr:`ratio`.

        Parameters
        ----------
        height_mm : float or None
            New explicit panel height, in millimetres. ``None`` reverts to
            sizing from :attr:`inches`/:attr:`ratio`.

        Raises
        ------
        ValueError
            If, after this change, the panel cannot be sized at all --
            i.e. :attr:`width_mm`/``height_mm`` are not both set, and
            :attr:`inches`/:attr:`ratio` are not both set either.
        """
        self._height_mm = height_mm
        self._rebuild()

    @property
    def inches(self) -> float | None:
        """Diagonal size in inches (only used when ``width_mm``/``height_mm`` are ``None``)."""
        return self._inches

    def set_inches(self, inches: float | None) -> None:
        """Set the diagonal size in inches and rebuild the panel.

        Only used when :attr:`width_mm`/:attr:`height_mm` are not both
        set (see the class docstring's sizing priority).

        Parameters
        ----------
        inches : float or None
            New diagonal size, in inches. ``None`` makes the diagonal
            sizing unusable, which is only a problem if
            :attr:`width_mm`/:attr:`height_mm` are not both set either
            (see ``Raises``).

        Raises
        ------
        ValueError
            If, after this change, the panel cannot be sized at all --
            i.e. :attr:`width_mm`/:attr:`height_mm` are not both set, and
            ``inches``/:attr:`ratio` are not both set either.
        """
        self._inches = inches
        self._rebuild()

    @property
    def ratio(self) -> str | None:
        """Aspect ratio ``"width:height"``, or ``None``.

        Only used when :attr:`width_mm`/:attr:`height_mm` are not both
        set.
        """
        return self._ratio

    def set_ratio(self, ratio: str | None) -> None:
        """Set the aspect ratio and rebuild the panel.

        Only used when :attr:`width_mm`/:attr:`height_mm` are not both
        set (see the class docstring's sizing priority).

        Parameters
        ----------
        ratio : str or None
            New aspect ratio as ``"width:height"`` (e.g. ``"16:9"``), both
            sides strictly positive numbers. ``None`` makes the diagonal
            sizing unusable, which is only a problem if
            :attr:`width_mm`/:attr:`height_mm` are not both set either
            (see ``Raises``).

        Raises
        ------
        ValueError
            If ``ratio`` is used (i.e. :attr:`width_mm`/:attr:`height_mm`
            are not both set) and is not a valid ``"width:height"``
            string (wrong number of parts, non-numeric parts, or either
            side not strictly positive); or if, after this change, the
            panel cannot be sized at all -- i.e.
            :attr:`width_mm`/:attr:`height_mm` are not both set, and
            :attr:`inches`/``ratio`` are not both set either.
        """
        self._ratio = ratio
        self._rebuild()

    @property
    def center_x(self) -> float:
        """Panel centre X coordinate in ground, in metres."""
        return self._center_x

    def set_center_x(self, center_x: float) -> None:
        """Set the panel centre X coordinate in ground and rebuild the panel.

        Not validated (no finiteness check) before being passed through to
        the new ``WeldJoint``.

        Parameters
        ----------
        center_x : float
            New panel centre X coordinate in the ground frame, in metres.
        """
        self._center_x = center_x
        self._rebuild()

    @property
    def center_y(self) -> float:
        """Panel centre Y coordinate in ground, in metres."""
        return self._center_y

    def set_center_y(self, center_y: float) -> None:
        """Set the panel centre Y coordinate in ground and rebuild the panel.

        Not validated (no finiteness check) before being passed through to
        the new ``WeldJoint``.

        Parameters
        ----------
        center_y : float
            New panel centre Y coordinate in the ground frame, in metres.
        """
        self._center_y = center_y
        self._rebuild()

    @property
    def center_z(self) -> float:
        """Panel centre Z coordinate in ground, in metres."""
        return self._center_z

    def set_center_z(self, center_z: float) -> None:
        """Set the panel centre Z coordinate in ground and rebuild the panel.

        Not validated (no finiteness check) before being passed through to
        the new ``WeldJoint``.

        Parameters
        ----------
        center_z : float
            New panel centre Z coordinate in the ground frame, in metres.
        """
        self._center_z = center_z
        self._rebuild()

    @property
    def origin(self) -> tuple[float, float, float]:
        """Current panel centre in the ground frame, in metres.

        Same value as ``(center_x, center_y, center_z)`` combined into one
        tuple -- and, since a screen panel has ``mass_center=(0, 0, 0)`` in
        its own frame (see :meth:`_rebuild`), also the same value as the
        inherited :attr:`~opensim_models.components.Body.position_global`.
        Kept as its own, longer-standing name here, mirroring
        :attr:`~opensim_models.components.Box.origin`, since "origin"
        reads more naturally for a panel's centre than the generic
        ``position_global``. Read directly off the body's placement, so it
        reflects the constructor's ``center_x``/``center_y``/``center_z``,
        :meth:`set_origin`, :meth:`set_center_x`/:meth:`set_center_y`/
        :meth:`set_center_z`, or any :meth:`rotate`/:meth:`translate`
        applied since -- never a stale cached value.
        """
        return self.position_global

    def set_origin(self, origin: tuple[float, float, float]) -> None:
        """Set the panel's centre in the ground frame, keeping its current orientation.

        Rebuilds the body/mesh/joint, same as a size/pose setter (see
        :meth:`set_width_mm`); unlike :class:`~opensim_models.components.Box`'s
        ``set_origin`` (which must explicitly re-pass ``angle_deg`` into its
        rebuild, since both are arguments to the same call), :attr:`angle_deg`
        here is simply untouched -- it is already its own separate stored
        field (see :meth:`_rebuild`), not derived from ``origin``. Prefer
        :meth:`translate` for a relative shift instead of an absolute
        position. Unlike the size setters, this does not validate ``origin``
        (no finiteness check) before passing it through to the new
        ``WeldJoint``.

        Parameters
        ----------
        origin : tuple[float, float, float]
            New panel centre ``(x, y, z)`` in the ground frame, in metres.
        """
        self._center_x, self._center_y, self._center_z = origin
        self._rebuild()

    @property
    def angle_deg(self) -> float:
        """Panel inclination relative to the ground, in degrees (0 = flat, 90 = upright)."""
        return self._angle_deg

    def set_angle_deg(self, angle_deg: float) -> None:
        """Set the panel inclination relative to the ground and rebuild the panel.

        Internally this is converted to a rotation of ``angle_deg - 90``
        degrees about ground's X axis for the new ``WeldJoint`` (so ``0``
        tilts the panel flat and ``90`` applies no extra tilt, leaving it
        upright); not validated (no finiteness check) before that
        conversion.

        Parameters
        ----------
        angle_deg : float
            New panel inclination relative to the ground, in degrees:
            ``0`` lies flat, ``90`` stands upright.
        """
        self._angle_deg = angle_deg
        self._rebuild()

    @property
    def mesh_dir(self) -> Path:
        """Directory ``screen_panel.stl`` is (re)written to on every rebuild."""
        return self._mesh_dir

    def set_mesh_dir(self, mesh_dir: str | Path) -> None:
        """Move where ``screen_panel.stl`` is written to, and rebuild the panel there.

        Creates ``mesh_dir`` (along with any missing parent) if it doesn't
        already exist, then rebuilds the panel -- same as a size/pose
        setter (see :meth:`set_width_mm`) -- which writes a fresh
        ``screen_panel.stl`` at the new location. The mesh file previously
        written to the old :attr:`mesh_dir`, if any, is left behind as-is.

        Parameters
        ----------
        mesh_dir : str or pathlib.Path
            New directory for the panel mesh.
        """
        self._mesh_dir = Path(mesh_dir)
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)
        self._rebuild()

    def rotate(
        self,
        origin: Any,
        direction: tuple[float, float, float],
        angle_deg: float,
        inplace: bool = True,
    ) -> Any:
        """Rotate this panel by ``angle_deg`` about the axis through ``origin`` along ``direction``.

        Thin wrapper around :func:`~opensim_models.operators.rotate_object`
        applied to this panel's own private container (always the
        whole-model case of that function, since ``self._container`` holds
        nothing but this one panel); see that function for the full
        parameter/return documentation, including every accepted
        ``origin`` form and the exact conditions under which it raises.

        Parameters
        ----------
        origin : tuple[float, float, float], opensim.Marker, opensim.Joint, or opensim.Frame
            Pivot point for the rotation, in the ground frame, in metres
            (when given as a plain coordinate). :attr:`~opensim_models.components.Body.com`
            (inherited) is a common choice, to spin the panel about its
            own centre.
        direction : tuple[float, float, float]
            Direction of the rotation axis through ``origin``, in the
            ground frame. Need not be a unit vector; must not be the zero
            vector.
        angle_deg : float
            Rotation angle, in degrees.
        inplace : bool, optional
            Defaults to ``True``: mutates this panel and returns its new
            ground-frame position (a ``tuple[float, float, float]``, in
            metres). When ``False``, this panel is left untouched and a
            standalone, independent ``OpenSimModel`` holding a rotated
            copy is returned instead (not another ``Screen``, since the
            rotation is generic to any container).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            This panel's new ground-frame position (``inplace=True``), or
            a rotated-copy ``OpenSimModel`` (``inplace=False``).

        Raises
        ------
        ValueError
            If ``direction`` is a zero vector, or ``origin`` is a
            coordinate without exactly 3 values.
        """
        return rotate_object(self._container, origin, direction, angle_deg, inplace=inplace)

    def translate(self, direction: tuple[float, float, float], inplace: bool = True) -> Any:
        """Translate this panel by ``direction`` (``dx, dy, dz``), in ground frame.

        Thin wrapper around
        :func:`~opensim_models.operators.translate_object` applied to this
        panel's own private container (always the whole-model case of that
        function); see that function for the full parameter/return
        documentation.

        Parameters
        ----------
        direction : tuple[float, float, float]
            Displacement ``(dx, dy, dz)``, in the ground frame, in metres.
        inplace : bool, optional
            Defaults to ``True``: mutates this panel and returns its new
            ground-frame position (a ``tuple[float, float, float]``, in
            metres). When ``False``, this panel is left untouched and a
            standalone, independent ``OpenSimModel`` holding a translated
            copy is returned instead (not another ``Screen``, since the
            translation is generic to any container).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            This panel's new ground-frame position (``inplace=True``), or
            a translated-copy ``OpenSimModel`` (``inplace=False``).
        """
        return translate_object(self._container, direction, inplace=inplace)

    def copy(self) -> "Screen":
        """Return a new, independent ``Screen`` with the same size, pose and sizing mode.

        Returns
        -------
        Screen
            A fresh ``Screen`` built from this one's current
            ``width_mm``/``height_mm``/``inches``/``ratio``/``center_x``/
            ``center_y``/``center_z``/``angle_deg``/``mesh_dir``/``name``
            -- its own private container, entirely independent of this
            panel's.
        """
        return Screen(
            width_mm=self._width_mm,
            height_mm=self._height_mm,
            inches=self._inches,
            ratio=self._ratio,
            center_x=self._center_x,
            center_y=self._center_y,
            center_z=self._center_z,
            angle_deg=self._angle_deg,
            mesh_dir=self._mesh_dir,
            name=self._name,
        )

    def set_name(self, name: str) -> None:
        """Rename this panel's body, and keep the name across any future rebuild.

        Any size/pose setter (:meth:`set_width_mm`, :meth:`set_height_mm`,
        :meth:`set_inches`, :meth:`set_ratio`, :meth:`set_center_x`,
        :meth:`set_center_y`, :meth:`set_center_z`, :meth:`set_angle_deg`)
        tears down and recreates the underlying ``opensim.Body`` from
        scratch (see :meth:`_rebuild`); overriding the base
        :meth:`~opensim_models.components.Body.set_name` to also remember
        ``name`` here is what makes a rename survive that, instead of
        silently reverting to ``"screen_panel"`` (or whatever ``name`` was
        passed to the constructor) on the next rebuild.

        Parameters
        ----------
        name : str
            New OpenSim name.
        """
        super().set_name(name)
        self._name = name

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

        # Named per self._name (not the fixed _MESH_FILENAME), so two
        # Screens sharing the same mesh_dir get their own mesh file instead
        # of overwriting each other's -- same fix, same reasoning, as
        # opensim_models.components.box.Box._rebuild. Defaults to
        # _MESH_FILENAME exactly (self._name defaults to _BODY_NAME, see
        # __init__), so an unnamed Screen's mesh path is unchanged from
        # before this existed.
        mesh_filename = f"{self._name}.stl"
        mesh_path = self._mesh_dir / mesh_filename
        _write_box_mesh(mesh_path, width_mm, height_mm, _THICKNESS_MM)

        container = self._container
        container.model = container.opensim.Model()
        container.model.setName("Screen")
        body = container.opensim.Body(
            self._name,
            mass,
            container.opensim.Vec3(0, 0, 0),
            container.opensim.Inertia(*inertia),
        )
        container.model.addBody(body)
        body.attachGeometry(container.opensim.Mesh(mesh_filename))

        joint = container.opensim.WeldJoint(
            _JOINT_NAME,
            container.model.getGround(),
            container.opensim.Vec3(self._center_x, self._center_y, self._center_z),
            container.opensim.Vec3(math.radians(self._angle_deg - 90.0), 0.0, 0.0),
            body,
            container.opensim.Vec3(0, 0, 0),
            container.opensim.Vec3(0, 0, 0),
        )
        container.model.addJoint(joint)

        container.model.finalizeConnections()
        container.state = container.model.initSystem()
        container._visualizer = None

        # The rebuild above replaces container.model wholesale, so the
        # previous body/joint SWIG proxies (and this wrapper's own
        # inherited `_owner`/`_raw`) are now stale -- re-point them at the
        # fresh body, same as any other _ComponentWrapper construction.
        super().__init__(container, container.model.getBodySet().get(self._name))
