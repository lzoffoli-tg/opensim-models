"""A parametric rectangular-prism ("box") model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...model import OpenSimModel
from ..._primitives import write_box_mesh

__all__ = ["Box"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MESHES_DIR = _ASSETS_DIR / "meshes"
_MESH_FILENAME = "box.stl"

_BODY_NAME = "box"
_JOINT_NAME = "box_joint"


class Box(OpenSimModel):
    """A rigid rectangular prism, welded to ground.

    A single ``opensim.Body`` ("box") sized ``width`` (local X) x
    ``height`` (local Y) x ``depth`` (local Z), in metres, with a matching
    box mesh (re)generated and written to ``assets/meshes/box.stl`` next
    to this module whenever a dimension changes. Mass is given directly
    (``mass_kg``, not derived from a material density); the inertia tensor
    is the analytical one for a solid rectangular prism of that mass and
    those dimensions.

    ``origin``/``angle_deg`` set the box's initial placement (a
    ``WeldJoint`` to ground: ``position=origin``,
    ``orientation_deg=angle_deg``, X-Y-Z body-fixed Euler degrees about
    ground's own axes). There are several ways to move it afterward, all
    equally valid: :meth:`set_origin`/:meth:`set_angle_deg` (each
    preserving the other half of the current pose); a dimension setter
    (``set_width``/``set_height``/``set_depth``/``set_mass_kg``, which also
    preserves the current pose across the rebuild it requires); or, since a
    ``Box`` is an :class:`~opensim_models.model.OpenSimModel` like any
    other, the inherited :meth:`~opensim_models.model.OpenSimModel.rotate`/
    :meth:`~opensim_models.model.OpenSimModel.translate` (or
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
    ) -> None:
        """Build the box body/joint from the given dimensions, pose and mass."""
        super().__init__(model_path=None)
        _MESHES_DIR.mkdir(parents=True, exist_ok=True)
        self.add_geometry_directory(_MESHES_DIR)

        self._width = self._positive(width)
        self._height = self._positive(height)
        self._depth = self._positive(depth)
        self._mass_kg = self._positive(mass_kg)
        self._rebuild(origin=origin, angle_deg=angle_deg)

    @property
    def width(self) -> float:
        """Full extent along local X, in metres."""
        return self._width

    def set_width(self, width: float) -> None:
        """Set the full extent along local X (metres) and rebuild the box, keeping its current pose."""
        self._width = self._positive(width)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def height(self) -> float:
        """Full extent along local Y, in metres."""
        return self._height

    def set_height(self, height: float) -> None:
        """Set the full extent along local Y (metres) and rebuild the box, keeping its current pose."""
        self._height = self._positive(height)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def depth(self) -> float:
        """Full extent along local Z, in metres."""
        return self._depth

    def set_depth(self, depth: float) -> None:
        """Set the full extent along local Z (metres) and rebuild the box, keeping its current pose."""
        self._depth = self._positive(depth)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def mass_kg(self) -> float:
        """Mass, in kilograms."""
        return self._mass_kg

    def set_mass_kg(self, mass_kg: float) -> None:
        """Set the mass (kg) and rebuild the box (inertia scales with it), keeping its current pose."""
        self._mass_kg = self._positive(mass_kg)
        self._rebuild(origin=self.origin, angle_deg=self.angle_deg)

    @property
    def origin(self) -> tuple[float, float, float]:
        """Current box centre in the ground frame, in metres.

        Read directly off the body's placement, so it reflects the
        constructor's ``origin``, :meth:`set_origin`, a dimension setter
        (pose-preserving), or any ``rotate``/``translate`` applied since --
        never a stale cached value.
        """
        self.model.realizePosition(self.state)
        position = self.body(_BODY_NAME).getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    def set_origin(self, origin: tuple[float, float, float]) -> None:
        """Set the box's centre in the ground frame (metres), keeping its current orientation.

        Rebuilds the body/mesh/joint, same as a dimension setter (see
        :meth:`set_width`); prefer :meth:`~opensim_models.model.OpenSimModel.translate`
        for a relative shift instead of an absolute position.
        """
        self._rebuild(origin=origin, angle_deg=self.angle_deg)

    @property
    def angle_deg(self) -> tuple[float, float, float]:
        """Current box orientation in the ground frame, as X-Y-Z body-fixed Euler degrees.

        Read directly off the body's placement (via its rotation matrix
        converted through ``opensim.Rotation``), so it reflects the
        constructor's ``angle_deg``, :meth:`set_angle_deg`, a dimension
        setter (pose-preserving), or any ``rotate`` applied since -- never
        a stale cached value.
        """
        self.model.realizePosition(self.state)
        rotation = self.body(_BODY_NAME).getRotationInGround(self.state)
        euler = rotation.convertRotationToBodyFixedXYZ()
        return tuple(float(np.degrees(euler.get(i))) for i in range(3))

    def set_angle_deg(self, angle_deg: tuple[float, float, float]) -> None:
        """Set the box's orientation (X-Y-Z body-fixed Euler degrees), keeping its current centre.

        Rebuilds the body/mesh/joint, same as a dimension setter (see
        :meth:`set_width`); prefer :meth:`~opensim_models.model.OpenSimModel.rotate`
        for a relative rotation (e.g. about :attr:`com`) instead of an
        absolute orientation.
        """
        self._rebuild(origin=self.origin, angle_deg=angle_deg)

    @property
    def com(self) -> tuple[float, float, float]:
        """Centre of mass in the ground frame, in metres.

        A single, uniform-density body with ``mass_center=(0, 0, 0)`` in
        its own frame, so this coincides with :attr:`origin`; exposed
        separately (via OpenSim's own whole-model calculation, the same
        one :attr:`~opensim_models.models.User.com` uses) as the natural
        pivot for :meth:`~opensim_models.model.OpenSimModel.rotate`.
        """
        self.model.realizePosition(self.state)
        position = self.model.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def corners(self) -> list[tuple[float, float, float]]:
        """Ground-frame coordinates of the box's 8 corners, in metres.

        Every combination of +/- half of ``width``/``height``/``depth``
        along the box's current local X/Y/Z axes, transformed through its
        current placement in ground -- so, like :attr:`origin`/
        :attr:`angle_deg`, always up to date.
        """
        self.model.realizePosition(self.state)
        body = self.body(_BODY_NAME)
        position = np.asarray(body.getPositionInGround(self.state).to_numpy())
        rotation_matrix = body.getRotationInGround(self.state).asMat33()
        rotation = np.array(
            [[rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
        )
        half_size = np.array([self._width, self._height, self._depth]) / 2.0
        signs = np.array(
            [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        return [tuple(position + rotation @ (sign * half_size)) for sign in signs]

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

        mesh_path = _MESHES_DIR / _MESH_FILENAME
        write_box_mesh(mesh_path, width, height, depth)

        self.model = self.opensim.Model()
        body = self.opensim.Body(
            _BODY_NAME, mass, self.opensim.Vec3(0, 0, 0), self.opensim.Inertia(*inertia)
        )
        self.model.addBody(body)
        body.attachGeometry(self.opensim.Mesh(_MESH_FILENAME))

        joint = self.opensim.WeldJoint(
            _JOINT_NAME,
            self.model.getGround(),
            self.opensim.Vec3(*origin),
            self.opensim.Vec3(*np.radians(angle_deg)),
            body,
            self.opensim.Vec3(0, 0, 0),
            self.opensim.Vec3(0, 0, 0),
        )
        self.model.addJoint(joint)

        self.model.finalizeConnections()
        self.state = self.model.initSystem()
        self._visualizer = None
