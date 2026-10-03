"""Add/remove a single ``opensim.Body``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import components
from ._shared import add_component, remove_component

__all__ = ["add_body", "remove_body"]

def _attach_mesh_files(model: "OpenSimModel", body: Any, mesh_files: Any) -> None:
    if mesh_files is None:
        return
    if isinstance(mesh_files, (str, Path)):
        mesh_files = [mesh_files]
    for mesh_file in mesh_files:
        mesh_path = Path(mesh_file)
        # Registers the search path (needed immediately: a Mesh resolves
        # its file when finalizing properties, i.e. right in attachGeometry
        # below, not lazily when the model is later shown).
        model.add_geometry_directory(mesh_path.parent)
        body.attachGeometry(model.opensim.Mesh(mesh_path.name))


def add_body(
    model: "OpenSimModel",
    name: str,
    mass: float,
    *,
    mass_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: tuple[float, float, float, float, float, float] = (0.0,) * 6,
    mesh_files: str | Path | list[str | Path] | None = None,
    reinitialize: bool = False,
) -> Any:
    """Construct an ``opensim.Body`` and add it to the model.

    The body is not connected to anything yet: the model cannot
    initialize its system until a joint connects it (see :func:`add_joint`)
    -- leave ``reinitialize=False`` (the default) and wrap both this call
    and the one adding the joint in
    :meth:`OpenSimModel.structural_change`.

    Parameters
    ----------
    model : OpenSimModel
        Model to add the body to.
    name : str
        Name for the new body.
    mass : float
        Mass, in kilograms.
    mass_center : tuple[float, float, float], optional
        Centre of mass in the body's own frame, in metres. Defaults to the
        body's origin.
    inertia : tuple[float, ...], optional
        ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`` central inertia tensor. Defaults
        to all zeros.
    mesh_files : str, pathlib.Path, list thereof, or None, optional
        Path(s) to existing mesh file(s) (``.vtp``, ``.stl``, ``.obj``) to
        attach to the body as geometry, and whose containing directories
        are registered for OpenSim's geometry search (see
        :meth:`OpenSimModel.add_geometry_directory`) and for :meth:`OpenSimModel.show`.
        For a body whose mesh is a primitive shape, see
        :func:`add_box_body`, :func:`add_cylinder_body` and
        :func:`add_sphere_body` instead, which can generate it for you.
    reinitialize : bool, optional
        See :func:`add_component`.

    Returns
    -------
    components.Body
        The newly created body, wrapped in
        :class:`~opensim_models.components.Body`, which exposes
        ``.mass``/``.set_mass()`` and the read-only, state-dependent
        ``.com``/``.inclination``/``.corners`` (ground-frame centre of
        mass, orientation, and bounding-box corners) on top of the common
        ``.name``/``.raw``/``.set_name()``.
    """
    body = model.opensim.Body(
        name,
        float(mass),
        model.opensim.Vec3(*mass_center),
        model.opensim.Inertia(*inertia),
    )
    add_component(model, "body", body, reinitialize=False)
    _attach_mesh_files(model, body, mesh_files)
    if reinitialize:
        model.reinitialize()
    return components.Body(model, body)


def remove_body(
    model: "OpenSimModel", name: str, *, reinitialize: bool = False
) -> None:
    """Remove a body from the model.

    Remove the joint connecting it (see :func:`remove_joint`) first, and
    any force/constraint referencing it, otherwise OpenSim will crash
    natively rather than raising a catchable error -- the same ordering
    :meth:`OpenSimModel.remove_model` follows internally. Wrap the body and
    joint removal together in :meth:`OpenSimModel.structural_change`.

    Parameters
    ----------
    model : OpenSimModel
        Model to remove the body from.
    name : str
        Name of the body to remove.
    reinitialize : bool, optional
        See :func:`add_component`.

    Raises
    ------
    ValueError
        If no body named ``name`` exists in the model.
    """
    remove_component(model, "body", name, reinitialize=reinitialize)


