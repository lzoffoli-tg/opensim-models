"""A parametric rectangular-prism ("box") component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import Body
from .._primitives import write_box_mesh
from ..model import OpenSimModel
from ..operators import rotate_object, translate_object

__all__ = ["Box"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MESHES_DIR = _ASSETS_DIR / "meshes"
_MESH_FILENAME = "box.stl"

_BODY_NAME = "box"
_JOINT_NAME = "box_joint"


def _positive(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"value must be strictly positive, got {value!r}")
    return float(value)


class Box(Body):
    """A rigid rectangular prism, welded to ground -- a single component.

    A :class:`~opensim_models.components.Body` wrapping one ``opensim.Body``
    ("box") sized ``width`` (local X) x ``height`` (local Y) x ``depth``
    (local Z), in metres, with a matching box mesh (re)generated and
    written to ``box.stl`` inside :attr:`mesh_dir` (defaulting to
    ``assets/meshes`` next to this module) whenever a dimension changes.
    Mass is given directly (``mass_kg``, not derived from a material
    density); the inertia tensor is the analytical one for a solid
    rectangular prism of that mass and those dimensions.

    Like any other component, a ``Box`` has no ``show()`` of its own --
    :meth:`~opensim_models.model.OpenSimModel.show` operates on
    containers, never on a single part. A ``Box`` is self-contained (it
    carries its own private ``OpenSimModel`` internally to do the actual
    OpenSim/state work this needs -- placement, mass, corners -- before it
    belongs to anything), but to actually *see* it, add it to a container
    first: ``model + box`` (or ``box + model``) returns a new, independent
    ``OpenSimModel`` with the box's body merged in, ready to ``.show()``.

    ``origin``/``angle_deg`` set the box's initial placement (a
    ``WeldJoint`` to ground: ``position=origin``,
    ``orientation_deg=angle_deg``, X-Y-Z body-fixed Euler degrees about
    ground's own axes). There are several ways to move it afterward, all
    equally valid: :meth:`set_origin`/:meth:`set_angle_deg` (each
    preserving the other half of the current pose); a dimension setter
    (``set_width``/``set_height``/``set_depth``/``set_mass_kg``, which also
    preserves the current pose across the rebuild it requires); or
    :meth:`rotate`/:meth:`translate` (or
    :func:`~opensim_models.operators.rotate_object`/
    :func:`~opensim_models.operators.translate_object` called on it
    directly) -- ``com`` is a natural pivot for the former
    (``box.rotate(box.com, axis, angle_deg)`` spins it about its own
    centre). :attr:`origin`, :attr:`angle_deg` and :attr:`corners` are
    always read directly off the box's current placement, so whichever of
    these actually moved it last, they immediately reflect it -- never a
    stale cached value.

    Parameters
    ----------
    width, height, depth : float
        Full extents along local X, Y, Z respectively, in metres. Must be
        strictly positive.
    origin : tuple[float, float, float], optional
        Initial box centre in ground coordinates, in metres. Defaults to
        ``(0.0, 0.0, 0.0)``.
    angle_deg : tuple[float, float, float], optional
        Initial box orientation relative to ground, as X-Y-Z body-fixed
        Euler angles in degrees. Defaults to ``(0.0, 0.0, 0.0)``.
    mass_kg : float, optional
        Mass, in kilograms. Must be strictly positive. Defaults to ``1.0``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory the box mesh (``box.stl``) is (re)written to, created
        (along with any missing parent) if it doesn't already exist.
        Defaults to ``None``, meaning the package's own bundled
        ``assets/meshes`` folder next to this module -- set this when that
        default location isn't writable (e.g. an admin-owned install) or
        to collect a model's generated meshes somewhere else of your
        choosing.

    Raises
    ------
    ValueError
        If ``width``, ``height``, ``depth`` or ``mass_kg`` is not finite or
        not strictly positive.
    """

    def __init__(
        self,
        width: float,
        height: float,
        depth: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angle_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass_kg: float = 1.0,
        mesh_dir: str | Path | None = None,
    ) -> None:
        """Build the box body/joint from the given dimensions, pose and mass."""
        # A private, never-exposed OpenSimModel: just enough of a container
        # for this one body/joint to be placeable and queryable (position,
        # state) before it belongs to any real container -- see the class
        # docstring for how it actually becomes visible (`model + box`).
        self._container = OpenSimModel(model_path=None)
        self._mesh_dir = Path(mesh_dir) if mesh_dir is not None else _MESHES_DIR
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)

        self._width = _positive(width)
        self._height = _positive(height)
        self._depth = _positive(depth)
        self._mass_kg = _positive(mass_kg)
        self._rebuild(origin=origin, angle_deg=angle_deg)

    @property
    def width(self) -> float:
        """Full extent along local X, in metres."""
        return self._width

    def set_width(self, width: float) -> None:
        """Set the full extent along local X and rebuild the box, keeping its current pose.

        Rebuilds the body, mesh and joint (new mesh written to
        ``box.stl`` inside :attr:`mesh_dir`, inertia recomputed from the new
        dimensions and the current :attr:`mass_kg`), re-reading
        :attr:`origin`/:attr:`angle_deg` beforehand so the box's placement
        is unchanged.

        Parameters
        ----------
        width : float
            New full extent along local X, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``width`` is not finite or not strictly positive.
        """
        self._width = _positive(width)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def height(self) -> float:
        """Full extent along local Y, in metres."""
        return self._height

    def set_height(self, height: float) -> None:
        """Set the full extent along local Y and rebuild the box, keeping its current pose.

        Same rebuild as :meth:`set_width`, along the Y axis instead.

        Parameters
        ----------
        height : float
            New full extent along local Y, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``height`` is not finite or not strictly positive.
        """
        self._height = _positive(height)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def depth(self) -> float:
        """Full extent along local Z, in metres."""
        return self._depth

    def set_depth(self, depth: float) -> None:
        """Set the full extent along local Z and rebuild the box, keeping its current pose.

        Same rebuild as :meth:`set_width`, along the Z axis instead.

        Parameters
        ----------
        depth : float
            New full extent along local Z, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``depth`` is not finite or not strictly positive.
        """
        self._depth = _positive(depth)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def mass_kg(self) -> float:
        """Mass, in kilograms."""
        return self._mass_kg

    def set_mass_kg(self, mass_kg: float) -> None:
        """Set the mass and rebuild the box (inertia scales with it), keeping its current pose.

        Unlike the generic :meth:`~opensim_models.components.Body.set_mass`
        (which changes mass only, assuming the inertia tensor is kept
        independently), this recomputes the analytical inertia tensor for
        the new mass at the box's current dimensions, then rebuilds the
        body/mesh/joint, re-reading :attr:`origin`/:attr:`angle_deg`
        beforehand so the box's placement is unchanged.

        Parameters
        ----------
        mass_kg : float
            New mass, in kilograms. Must be finite and strictly positive.

        Raises
        ------
        ValueError
            If ``mass_kg`` is not finite or not strictly positive.
        """
        self._mass_kg = _positive(mass_kg)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def mass(self) -> float:
        """Mass, in kilograms -- same value as :attr:`mass_kg`.

        Overrides :attr:`~opensim_models.components.Body.mass` only so its
        setter stays routed through :meth:`set_mass_kg` (inertia must scale
        with mass for a box, unlike the generic
        :meth:`~opensim_models.components.Body.set_mass`, which assumes an
        inertia tensor set independently).
        """
        return self._mass_kg

    def set_mass(self, kilograms: float) -> None:
        """Set the mass; a direct alias for :meth:`set_mass_kg` (same thing, inertia included).

        Parameters
        ----------
        kilograms : float
            New mass, in kilograms. Must be finite and strictly positive;
            see :meth:`set_mass_kg`.

        Raises
        ------
        ValueError
            If ``kilograms`` is not finite or not strictly positive.
        """
        self.set_mass_kg(kilograms)

    @property
    def mesh_dir(self) -> Path:
        """Directory ``box.stl`` is (re)written to on every rebuild."""
        return self._mesh_dir

    def set_mesh_dir(self, mesh_dir: str | Path) -> None:
        """Move where ``box.stl`` is written to, and rebuild the box there.

        Creates ``mesh_dir`` (along with any missing parent) if it doesn't
        already exist, then rebuilds the box -- same as a dimension setter
        (see :meth:`set_width`) -- which writes a fresh ``box.stl`` at the
        new location. The mesh file previously written to the old
        :attr:`mesh_dir`, if any, is left behind as-is.

        Parameters
        ----------
        mesh_dir : str or pathlib.Path
            New directory for the box mesh.
        """
        self._mesh_dir = Path(mesh_dir)
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def origin(self) -> tuple[float, float, float]:
        """Current box centre in the ground frame, in metres.

        Read directly off the body's placement, so it reflects the
        constructor's ``origin``, :meth:`set_origin`, a dimension setter
        (pose-preserving), or any :meth:`rotate`/:meth:`translate` applied
        since -- never a stale cached value.
        """
        self._container.model.realizePosition(self._container.state)
        position = self.raw.getPositionInGround(self._container.state)
        return (position.get(0), position.get(1), position.get(2))

    def set_origin(self, origin: tuple[float, float, float]) -> None:
        """Set the box's centre in the ground frame, keeping its current orientation.

        Rebuilds the body/mesh/joint, same as a dimension setter (see
        :meth:`set_width`), re-reading :attr:`angle_deg` beforehand so the
        box's orientation is unchanged; prefer :meth:`translate` for a
        relative shift instead of an absolute position. Unlike the
        dimension/mass setters, this does not validate ``origin`` (no
        finiteness check) before passing it through to the new
        ``WeldJoint``.

        Parameters
        ----------
        origin : tuple[float, float, float]
            New box centre ``(x, y, z)`` in the ground frame, in metres.
        """
        self._rebuild(origin=origin, angle_deg=self.angle_deg)

    @property
    def angle_deg(self) -> tuple[float, float, float]:
        """Current box orientation in the ground frame, as X-Y-Z body-fixed Euler degrees.

        Read directly off the body's placement (via its rotation matrix
        converted through ``opensim.Rotation``), so it reflects the
        constructor's ``angle_deg``, :meth:`set_angle_deg`, a dimension
        setter (pose-preserving), or any :meth:`rotate` applied since --
        never a stale cached value.
        """
        self._container.model.realizePosition(self._container.state)
        rotation = self.raw.getRotationInGround(self._container.state)
        euler = rotation.convertRotationToBodyFixedXYZ()
        return tuple(float(np.degrees(euler.get(i))) for i in range(3))

    def set_angle_deg(self, angle_deg: tuple[float, float, float]) -> None:
        """Set the box's orientation, keeping its current centre.

        Rebuilds the body/mesh/joint, same as a dimension setter (see
        :meth:`set_width`), re-reading :attr:`origin` beforehand so the
        box's centre is unchanged; prefer :meth:`rotate` for a relative
        rotation (e.g. about :attr:`com`) instead of an absolute
        orientation. Unlike the dimension/mass setters, this does not
        validate ``angle_deg`` (no finiteness check) before converting it
        to radians and passing it through to the new ``WeldJoint``.

        Parameters
        ----------
        angle_deg : tuple[float, float, float]
            New box orientation relative to ground, as X-Y-Z body-fixed
            Euler angles in degrees about ground's own axes.
        """
        self._rebuild(origin=self.origin, angle_deg=angle_deg)

    @property
    def com(self) -> tuple[float, float, float]:
        """Centre of mass in the ground frame, in metres.

        A single, uniform-density body with ``mass_center=(0, 0, 0)`` in
        its own frame, so this coincides with :attr:`origin`; exposed
        separately (same inherited calculation as
        :attr:`~opensim_models.components.Body.com`) as the natural pivot
        for :meth:`rotate`.
        """
        return super().com

    @property
    def corners(self) -> tuple[tuple[float, float, float], ...]:
        """Ground-frame coordinates of the box's 8 corners, in metres.

        Every combination of +/- half of ``width``/``height``/``depth``
        along the box's current local X/Y/Z axes, transformed through its
        current placement in ground -- so, like :attr:`origin`/
        :attr:`angle_deg`, always up to date. An exact analytical
        alternative to the generic, mesh-bounds-based
        :attr:`~opensim_models.components.Body.corners` -- equivalent for a
        box, but without reading the mesh file back.
        """
        self._container.model.realizePosition(self._container.state)
        position = np.asarray(self.raw.getPositionInGround(self._container.state).to_numpy())
        rotation_matrix = self.raw.getRotationInGround(self._container.state).asMat33()
        rotation = np.array(
            [[rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
        )
        half_size = np.array([self._width, self._height, self._depth]) / 2.0
        signs = np.array(
            [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        return tuple(tuple(position + rotation @ (sign * half_size)) for sign in signs)

    def rotate(
        self,
        origin: Any,
        direction: tuple[float, float, float],
        angle_deg: float,
        inplace: bool = True,
    ) -> Any:
        """Rotate this box by ``angle_deg`` about the axis through ``origin`` along ``direction``.

        Thin wrapper around :func:`~opensim_models.operators.rotate_object`
        applied to this box's own private container (always the whole-model
        case of that function, since ``self._container`` holds nothing but
        this one box); see that function for the full parameter/return
        documentation, including every accepted ``origin`` form and the
        exact conditions under which it raises.

        Parameters
        ----------
        origin : tuple[float, float, float], opensim.Marker, opensim.Joint, or opensim.Frame
            Pivot point for the rotation, in the ground frame, in metres
            (when given as a plain coordinate). :attr:`com` is a common
            choice, to spin the box about its own centre.
        direction : tuple[float, float, float]
            Direction of the rotation axis through ``origin``, in the
            ground frame. Need not be a unit vector; must not be the zero
            vector.
        angle_deg : float
            Rotation angle, in degrees.
        inplace : bool, optional
            Defaults to ``True``: mutates this box and returns its new
            ground-frame position (a ``tuple[float, float, float]``, in
            metres). When ``False``, this box is left untouched and a
            standalone, independent ``OpenSimModel`` holding a rotated
            copy is returned instead (not another ``Box``, since the
            rotation is generic to any container).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            This box's new ground-frame position (``inplace=True``), or a
            rotated-copy ``OpenSimModel`` (``inplace=False``).

        Raises
        ------
        ValueError
            If ``direction`` is a zero vector, or ``origin`` is a
            coordinate without exactly 3 values.
        """
        return rotate_object(self._container, origin, direction, angle_deg, inplace=inplace)

    def translate(self, direction: tuple[float, float, float], inplace: bool = True) -> Any:
        """Translate this box by ``direction`` (``dx, dy, dz``), in ground frame.

        Thin wrapper around
        :func:`~opensim_models.operators.translate_object` applied to this
        box's own private container (always the whole-model case of that
        function); see that function for the full parameter/return
        documentation.

        Parameters
        ----------
        direction : tuple[float, float, float]
            Displacement ``(dx, dy, dz)``, in the ground frame, in metres.
        inplace : bool, optional
            Defaults to ``True``: mutates this box and returns its new
            ground-frame position (a ``tuple[float, float, float]``, in
            metres). When ``False``, this box is left untouched and a
            standalone, independent ``OpenSimModel`` holding a translated
            copy is returned instead (not another ``Box``, since the
            translation is generic to any container).

        Returns
        -------
        tuple[float, float, float] or OpenSimModel
            This box's new ground-frame position (``inplace=True``), or a
            translated-copy ``OpenSimModel`` (``inplace=False``).
        """
        return translate_object(self._container, direction, inplace=inplace)

    def copy(self) -> "Box":
        """Return a new, independent ``Box`` with the same dimensions, mass and pose.

        Returns
        -------
        Box
            A fresh ``Box`` built from this one's current
            ``width``/``height``/``depth``/``mass_kg``/``origin``/``angle_deg``/
            ``mesh_dir`` -- its own private container, entirely independent
            of this box's.
        """
        return Box(
            self._width,
            self._height,
            self._depth,
            origin=self.origin,
            angle_deg=self.angle_deg,
            mass_kg=self._mass_kg,
            mesh_dir=self._mesh_dir,
        )

    def _rebuild(
        self, *, origin: tuple[float, float, float], angle_deg: tuple[float, float, float]
    ) -> None:
        """Recreate the box body, mesh and joint at the given pose."""
        width, height, depth = self._width, self._height, self._depth
        mass = self._mass_kg
        inertia = (
            mass * (height**2 + depth**2) / 12.0,
            mass * (width**2 + depth**2) / 12.0,
            mass * (width**2 + height**2) / 12.0,
        )

        mesh_path = self._mesh_dir / _MESH_FILENAME
        write_box_mesh(mesh_path, width, height, depth)

        container = self._container
        container.model = container.opensim.Model()
        container.model.setName("Box")
        body = container.opensim.Body(
            _BODY_NAME, mass, container.opensim.Vec3(0, 0, 0), container.opensim.Inertia(*inertia)
        )
        container.model.addBody(body)
        body.attachGeometry(container.opensim.Mesh(_MESH_FILENAME))

        joint = container.opensim.WeldJoint(
            _JOINT_NAME,
            container.model.getGround(),
            container.opensim.Vec3(*origin),
            container.opensim.Vec3(*np.radians(angle_deg)),
            body,
            container.opensim.Vec3(0, 0, 0),
            container.opensim.Vec3(0, 0, 0),
        )
        container.model.addJoint(joint)

        container.model.finalizeConnections()
        container.state = container.model.initSystem()
        container._visualizer = None

        # The rebuild above replaces container.model wholesale, so the
        # previous `body`/joint SWIG proxies (and this wrapper's own
        # inherited `_owner`/`_raw`) are now stale -- re-point them at the
        # fresh body, same as any other _ComponentWrapper construction.
        super().__init__(container, container.model.getBodySet().get(_BODY_NAME))
