"""A parametric solid-cylinder component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import Body
from .._primitives import write_cylinder_mesh
from ..model import OpenSimModel
from ..operators import rotate_object, translate_object

__all__ = ["Cylinder"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MESHES_DIR = _ASSETS_DIR / "meshes"

_BODY_NAME = "cylinder"
_JOINT_NAME = "cylinder_joint"


def _positive(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"value must be strictly positive, got {value!r}")
    return float(value)


class Cylinder(Body):
    """A rigid solid cylinder welded to ground as a single component.

    The cylinder's axis is its local Y axis, matching OpenSim's
    ``opensim.Cylinder`` convention. ``radius`` and ``height`` are in
    metres; ``mass_kg`` is specified directly and its inertia is computed
    analytically for a solid cylinder.

    As with :class:`~opensim_models.components.Box`, ``Cylinder`` owns a
    private model for its body and joint. Add it to an
    :class:`~opensim_models.model.OpenSimModel` (``model + cylinder``) to
    display or compose it.

    Parameters
    ----------
    radius : float
        Cylinder base radius in metres. Must be finite and positive.
    height : float
        Cylinder height along local Y in metres. Must be finite and positive.
    center_x, center_y, center_z : float, optional
        Initial cylinder centre in ground coordinates, in metres.
    angle_deg : tuple[float, float, float], optional
        Initial orientation relative to ground, as X-Y-Z body-fixed Euler
        angles in degrees.
    mass_kg : float, optional
        Mass in kilograms. Must be finite and positive. Defaults to 1.0.
    mesh_dir : str, pathlib.Path or None, optional
        Directory for the generated STL mesh. Defaults to this component's
        ``assets/meshes`` directory.
    name : str or None, optional
        OpenSim body name. Defaults to ``"cylinder"``.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        angle_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass_kg: float = 1.0,
        mesh_dir: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        self._container = OpenSimModel(model_path=None)
        self._mesh_dir = Path(mesh_dir) if mesh_dir is not None else _MESHES_DIR
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)

        self._radius = _positive(radius)
        self._height = _positive(height)
        self._mass_kg = _positive(mass_kg)
        self._name = name if name is not None else _BODY_NAME
        self._center_x = center_x
        self._center_y = center_y
        self._center_z = center_z
        self._angle_deg = angle_deg
        self._rebuild()

    @property
    def radius(self) -> float:
        """Cylinder base radius in metres."""
        return self._radius

    def set_radius(self, radius: float) -> None:
        """Set the radius and rebuild the cylinder, preserving its pose."""
        radius = _positive(radius)
        origin, angle_deg = self.origin, self.angle_deg
        self._radius = radius
        self._rebuild(origin=origin, angle_deg=angle_deg)

    @property
    def height(self) -> float:
        """Cylinder height along local Y, in metres."""
        return self._height

    def set_height(self, height: float) -> None:
        """Set the height and rebuild the cylinder, preserving its pose."""
        height = _positive(height)
        origin, angle_deg = self.origin, self.angle_deg
        self._height = height
        self._rebuild(origin=origin, angle_deg=angle_deg)

    @property
    def mass_kg(self) -> float:
        """Mass in kilograms."""
        return self._mass_kg

    def set_mass_kg(self, mass_kg: float) -> None:
        """Set the mass and recompute the cylinder's inertia, preserving its pose."""
        mass_kg = _positive(mass_kg)
        origin, angle_deg = self.origin, self.angle_deg
        self._mass_kg = mass_kg
        self._rebuild(origin=origin, angle_deg=angle_deg)

    @property
    def mass(self) -> float:
        """Mass in kilograms, same as :attr:`mass_kg`."""
        return self._mass_kg

    def set_mass(self, kilograms: float) -> None:
        """Set the mass; alias for :meth:`set_mass_kg`."""
        self.set_mass_kg(kilograms)

    @property
    def center_x(self) -> float:
        """Current centre X coordinate in ground, in metres."""
        return self.origin[0]

    def set_center_x(self, center_x: float) -> None:
        """Set the centre X coordinate, preserving the other pose values."""
        origin = self.origin
        self._rebuild(
            origin=(center_x, origin[1], origin[2]), angle_deg=self.angle_deg
        )

    @property
    def center_y(self) -> float:
        """Current centre Y coordinate in ground, in metres."""
        return self.origin[1]

    def set_center_y(self, center_y: float) -> None:
        """Set the centre Y coordinate, preserving the other pose values."""
        origin = self.origin
        self._rebuild(
            origin=(origin[0], center_y, origin[2]), angle_deg=self.angle_deg
        )

    @property
    def center_z(self) -> float:
        """Current centre Z coordinate in ground, in metres."""
        return self.origin[2]

    def set_center_z(self, center_z: float) -> None:
        """Set the centre Z coordinate, preserving the other pose values."""
        origin = self.origin
        self._rebuild(
            origin=(origin[0], origin[1], center_z), angle_deg=self.angle_deg
        )

    @property
    def origin(self) -> tuple[float, float, float]:
        """Current cylinder centre in ground coordinates, in metres."""
        return self.position_global

    def set_origin(self, origin: tuple[float, float, float]) -> None:
        """Set the centre, preserving the current orientation."""
        self._rebuild(origin=origin, angle_deg=self.angle_deg)

    @property
    def angle_deg(self) -> tuple[float, float, float]:
        """Current X-Y-Z body-fixed Euler orientation in degrees."""
        return self.inclination

    def set_angle_deg(self, angle_deg: tuple[float, float, float]) -> None:
        """Set the orientation, preserving the current centre."""
        self._rebuild(origin=self.origin, angle_deg=angle_deg)

    @property
    def mesh_dir(self) -> Path:
        """Directory where the cylinder's mesh is written."""
        return self._mesh_dir

    def set_mesh_dir(self, mesh_dir: str | Path) -> None:
        """Change the mesh directory and regenerate the mesh there."""
        origin, angle_deg = self.origin, self.angle_deg
        self._mesh_dir = Path(mesh_dir)
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._container.add_geometry_directory(self._mesh_dir)
        self._rebuild(origin=origin, angle_deg=angle_deg)

    def rotate(
        self,
        origin: Any,
        direction: tuple[float, float, float],
        angle_deg: float,
        inplace: bool = True,
    ) -> Any:
        """Rotate this cylinder using the generic object-rotation operator."""
        return rotate_object(
            self._container, origin, direction, angle_deg, inplace=inplace
        )

    def translate(
        self, direction: tuple[float, float, float], inplace: bool = True
    ) -> Any:
        """Translate this cylinder using the generic object-translation operator."""
        return translate_object(self._container, direction, inplace=inplace)

    def copy(self) -> "Cylinder":
        """Return an independent cylinder with the same dimensions, mass and pose."""
        return Cylinder(
            self._radius,
            self._height,
            center_x=self.center_x,
            center_y=self.center_y,
            center_z=self.center_z,
            angle_deg=self.angle_deg,
            mass_kg=self._mass_kg,
            mesh_dir=self._mesh_dir,
            name=self._name,
        )

    def set_name(self, name: str) -> None:
        """Rename the body and preserve that name across later rebuilds."""
        super().set_name(name)
        self._name = name

    def _rebuild(
        self,
        *,
        origin: tuple[float, float, float] | None = None,
        angle_deg: tuple[float, float, float] | None = None,
    ) -> None:
        """Recreate the body, STL mesh and welded joint."""
        if origin is None:
            origin = (self._center_x, self._center_y, self._center_z)
        if angle_deg is None:
            angle_deg = self._angle_deg

        self._center_x, self._center_y, self._center_z = origin
        self._angle_deg = angle_deg

        radius, height, mass = self._radius, self._height, self._mass_kg
        axial_inertia = mass * radius**2 / 2.0
        transverse_inertia = mass * (3.0 * radius**2 + height**2) / 12.0
        mesh_filename = f"{self._name}.stl"
        write_cylinder_mesh(self._mesh_dir / mesh_filename, radius, height)

        container = self._container
        container.model = container.opensim.Model()
        container.model.setName("Cylinder")
        body = container.opensim.Body(
            self._name,
            mass,
            container.opensim.Vec3(0, 0, 0),
            container.opensim.Inertia(
                transverse_inertia, axial_inertia, transverse_inertia
            ),
        )
        container.model.addBody(body)
        body.attachGeometry(container.opensim.Mesh(mesh_filename))

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
        super().__init__(container, container.model.getBodySet().get(self._name))
