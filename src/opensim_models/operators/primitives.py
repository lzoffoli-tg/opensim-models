"""Add a body of a primitive shape (box/cylinder/sphere) in one call: the
body, its joint, and its geometry (native primitive or a generated mesh)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .bodies import add_body
from .joints import add_free_joint, add_pin_joint, add_ball_joint, add_slider_joint, add_weld_joint

__all__ = ["add_box_body", "add_cylinder_body", "add_sphere_body"]

_JOINT_ADDERS = {
    "free": add_free_joint,
    "pin": add_pin_joint,
    "ball": add_ball_joint,
    "slider": add_slider_joint,
    "weld": add_weld_joint,
}


def _add_primitive_body(
    model: "OpenSimModel",
    name: str,
    mass: float,
    inertia: tuple[float, float, float, float, float, float],
    *,
    position: tuple[float, float, float],
    orientation_deg: tuple[float, float, float],
    joint_type: str,
    write_mesh: Any,
    native_geometry_factory: Any,
    mesh: bool,
    mesh_dir: str | Path | None,
    mesh_filename: str,
    reinitialize: bool,
) -> tuple[Any, Any]:
    if joint_type not in _JOINT_ADDERS:
        valid = ", ".join(sorted(_JOINT_ADDERS))
        raise ValueError(f"Unknown joint_type {joint_type!r}; expected one of {valid}")
    if mesh and mesh_dir is None:
        raise ValueError("mesh_dir is required when mesh=True")

    mesh_files = None
    if mesh:
        mesh_path = Path(mesh_dir) / mesh_filename
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(mesh_path)
        mesh_files = mesh_path

    def build() -> tuple[Any, Any]:
        body = add_body(model, name, mass, inertia=inertia, mesh_files=mesh_files)
        if not mesh:
            body.raw.attachGeometry(native_geometry_factory())
        joint = _JOINT_ADDERS[joint_type](
            model,
            f"{name}_joint",
            body,
            position=position,
            orientation_deg=orientation_deg,
        )
        return body, joint

    if reinitialize:
        with model.structural_change():
            return build()
    return build()


def add_box_body(
    model: "OpenSimModel",
    name: str,
    size: tuple[float, float, float],
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a box-shaped body, positioned and oriented by its own joint.

    Mass and (central) inertia are derived analytically from ``size`` and
    ``density``, so they always match what is actually drawn.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    size : tuple[float, float, float]
        Full extents ``(size_x, size_y, size_z)`` of the box, in metres.
    density : float, optional
        Density, in kg/m^3, used to derive mass and inertia from ``size``.
        Defaults to ``1000.0`` (water), a generic placeholder.
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        Joint connecting the body to ground: one of ``"weld"`` (default,
        rigid), ``"free"``, ``"pin"``, ``"ball"`` or ``"slider"``.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Brick`` geometry (no file written). When ``True``,
        generate and attach an actual box mesh file instead (useful for
        exporting the model portably) -- requires ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately,
        preserving the current posture/velocity, so the model is ready to
        use. Leave ``False`` and wrap several primitive-body calls in one
        :meth:`OpenSimModel.structural_change` block to add them as a
        batch, rebuilding the system only once.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body, wrapped in
        :class:`~opensim_models.components.Body` (exposing
        ``.mass``/``.set_mass()`` and the read-only ``.com``/
        ``.inclination``/``.corners``), and the joint connecting it to
        ground (named ``f"{name}_joint"``), wrapped in
        :class:`~opensim_models.components.Joint` (exposing
        ``.coordinates``, whose size/unit depends on ``joint_type``: 0 for
        ``"weld"``, 1 (radians) for ``"pin"``, 3 (radians) for ``"ball"``,
        1 (metres) for ``"slider"``, or 6 (3 radians + 3 metres) for
        ``"free"``).

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    size_x, size_y, size_z = size
    volume = size_x * size_y * size_z
    mass = volume * density
    inertia = (
        mass * (size_y**2 + size_z**2) / 12.0,
        mass * (size_x**2 + size_z**2) / 12.0,
        mass * (size_x**2 + size_y**2) / 12.0,
        0.0,
        0.0,
        0.0,
    )

    def write_mesh(path: Path) -> None:
        from .. import _primitives

        _primitives.write_box_mesh(path, size_x, size_y, size_z)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Brick(
            model.opensim.Vec3(size_x / 2.0, size_y / 2.0, size_z / 2.0)
        ),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


def add_cylinder_body(
    model: "OpenSimModel",
    name: str,
    radius: float,
    height: float,
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a cylinder-shaped body, positioned and oriented by its own joint.

    The cylinder's axis is its local Y axis (matching ``opensim.Cylinder``'s
    own convention); use ``orientation_deg`` to point it elsewhere. Mass
    and (central) inertia are derived analytically from ``radius``,
    ``height`` and ``density``.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    radius : float
        Base radius, in metres.
    height : float
        Height along the cylinder's own Y axis, in metres.
    density : float, optional
        Density, in kg/m^3. Defaults to ``1000.0`` (water).
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        Joint connecting the body to ground: one of ``"weld"`` (default,
        rigid), ``"free"``, ``"pin"``, ``"ball"`` or ``"slider"``.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Cylinder`` geometry (no file written). When ``True``,
        generate and attach an actual mesh file instead -- requires
        ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately,
        preserving the current posture/velocity, so the model is ready to
        use. Leave ``False`` and wrap several primitive-body calls in one
        :meth:`OpenSimModel.structural_change` block to add them as a
        batch, rebuilding the system only once.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body, wrapped in
        :class:`~opensim_models.components.Body` (exposing
        ``.mass``/``.set_mass()`` and the read-only ``.com``/
        ``.inclination``/``.corners``), and the joint connecting it to
        ground (named ``f"{name}_joint"``), wrapped in
        :class:`~opensim_models.components.Joint` (exposing
        ``.coordinates``, whose size/unit depends on ``joint_type``: 0 for
        ``"weld"``, 1 (radians) for ``"pin"``, 3 (radians) for ``"ball"``,
        1 (metres) for ``"slider"``, or 6 (3 radians + 3 metres) for
        ``"free"``).

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    volume = math.pi * radius**2 * height
    mass = volume * density
    inertia = (
        mass * (3.0 * radius**2 + height**2) / 12.0,
        mass * radius**2 / 2.0,
        mass * (3.0 * radius**2 + height**2) / 12.0,
        0.0,
        0.0,
        0.0,
    )

    def write_mesh(path: Path) -> None:
        from .. import _primitives

        _primitives.write_cylinder_mesh(path, radius, height)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Cylinder(radius, height / 2.0),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


def add_sphere_body(
    model: "OpenSimModel",
    name: str,
    radius: float,
    *,
    density: float = 1000.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_type: str = "weld",
    mesh: bool = False,
    mesh_dir: str | Path | None = None,
    reinitialize: bool = False,
) -> tuple[Any, Any]:
    """Add a sphere-shaped body, positioned and oriented by its own joint.

    Mass and (central) inertia are derived analytically from ``radius`` and
    ``density``. ``orientation_deg`` has no visible effect on the sphere
    itself (a solid sphere is rotationally symmetric) but still applies to
    the joint, e.g. if ``joint_type`` gives it a non-symmetric-axis motion.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body; its joint is named ``f"{name}_joint"``.
    radius : float
        Radius, in metres.
    density : float, optional
        Density, in kg/m^3. Defaults to ``1000.0`` (water).
    position : tuple[float, float, float], optional
        Centre of mass in ``model``'s ground frame, in metres.
    orientation_deg : tuple[float, float, float], optional
        Orientation relative to ground, as X-Y-Z body-fixed Euler angles in
        degrees about ground's own axes.
    joint_type : str, optional
        Joint connecting the body to ground: one of ``"weld"`` (default,
        rigid), ``"free"``, ``"pin"``, ``"ball"`` or ``"slider"``.
    mesh : bool, optional
        When ``False`` (default), attach a lightweight native
        ``opensim.Sphere`` geometry (no file written). When ``True``,
        generate and attach an actual mesh file instead -- requires
        ``mesh_dir``.
    mesh_dir : str, pathlib.Path or None, optional
        Directory to write the generated mesh file to, when ``mesh=True``.
    reinitialize : bool, optional
        When ``True`` (default ``False``), rebuild the system immediately,
        preserving the current posture/velocity, so the model is ready to
        use. Leave ``False`` and wrap several primitive-body calls in one
        :meth:`OpenSimModel.structural_change` block to add them as a
        batch, rebuilding the system only once.

    Returns
    -------
    tuple[components.Body, components.Joint]
        The newly created body, wrapped in
        :class:`~opensim_models.components.Body` (exposing
        ``.mass``/``.set_mass()`` and the read-only ``.com``/
        ``.inclination``/``.corners``), and the joint connecting it to
        ground (named ``f"{name}_joint"``), wrapped in
        :class:`~opensim_models.components.Joint` (exposing
        ``.coordinates``, whose size/unit depends on ``joint_type``: 0 for
        ``"weld"``, 1 (radians) for ``"pin"``, 3 (radians) for ``"ball"``,
        1 (metres) for ``"slider"``, or 6 (3 radians + 3 metres) for
        ``"free"``).

    Raises
    ------
    ValueError
        If ``joint_type`` is not recognized, or if ``mesh=True`` and
        ``mesh_dir`` is not given.
    """
    volume = 4.0 / 3.0 * math.pi * radius**3
    mass = volume * density
    moment = 2.0 / 5.0 * mass * radius**2
    inertia = (moment, moment, moment, 0.0, 0.0, 0.0)

    def write_mesh(path: Path) -> None:
        from .. import _primitives

        _primitives.write_sphere_mesh(path, radius)

    return _add_primitive_body(
        model,
        name,
        mass,
        inertia,
        position=position,
        orientation_deg=orientation_deg,
        joint_type=joint_type,
        write_mesh=write_mesh,
        native_geometry_factory=lambda: model.opensim.Sphere(radius),
        mesh=mesh,
        mesh_dir=mesh_dir,
        mesh_filename=f"{name}.stl",
        reinitialize=reinitialize,
    )


