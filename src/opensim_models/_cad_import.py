"""CAD import helpers used by :meth:`opensim_models.OpenSimModel.from_step`.

This module isolates the optional, heavy ``pythonocc-core`` (``OCC``)
dependency: nothing here is imported unless :meth:`OpenSimModel.from_step`
is actually called.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _import_occ_error() -> RuntimeError:
    return RuntimeError(
        "pythonocc-core is required to build a model from a CAD (.step/.stp) "
        "file. Install it, e.g. via 'conda install -c conda-forge "
        "pythonocc-core'."
    )


# STEP length unit names (as reported by STEPControl_Reader.FileUnits) mapped
# to their scale factor to metres. Checked in order, so multi-word units
# ("millimetre") are matched before the shorter unit they contain ("metre").
_UNIT_SCALES_TO_METRES = (
    ("millimetre", 0.001),
    ("millimeter", 0.001),
    ("centimetre", 0.01),
    ("centimeter", 0.01),
    ("kilometre", 1000.0),
    ("kilometer", 1000.0),
    ("inch", 0.0254),
    ("foot", 0.3048),
    ("metre", 1.0),
    ("meter", 1.0),
)


def _file_length_unit_scale(reader: Any) -> float:
    """Return the scale factor to metres for a STEP reader's declared unit.

    Interface_Static's "xstep.cascade.unit" translation knob proved
    unreliable across pythonocc-core versions (it does not consistently
    convert STEP files to the requested unit), so the file's declared unit
    is read directly and the geometry is rescaled manually instead.
    """
    from OCC.Core.TColStd import TColStd_SequenceOfAsciiString

    lengths, angles, solid_angles = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.FileUnits(lengths, angles, solid_angles)
    if lengths.Length() == 0:
        return 1.0
    unit_name = lengths.Value(1).ToCString().lower()
    for name, scale in _UNIT_SCALES_TO_METRES:
        if name in unit_name:
            return scale
    return 1.0


def _read_step_solids(step_path: str | Path) -> list[tuple[str, Any]]:
    """Read every solid in a STEP file, paired with its part name.

    Names come from the STEP/XCAF product structure when available (parts
    exported from an assembly typically carry their CAD part name); solids
    that cannot be named this way -- including files with no assembly
    structure at all -- fall back to ``"body_0"``, ``"body_1"``, etc.
    Geometry is translated to metres regardless of the unit declared in the
    file, since OpenSim models are expressed in SI units.

    Parameters
    ----------
    step_path : str or pathlib.Path
        Path to a ``.step``/``.stp`` file.

    Returns
    -------
    list[tuple[str, TopoDS_Shape]]
        ``(name, solid)`` pairs, one per solid found in the file.

    Raises
    ------
    ValueError
        If the file cannot be read or contains no solids.
    RuntimeError
        If ``pythonocc-core`` is not installed.
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Pnt, gp_Trsf
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
        from OCC.Extend.DataExchange import read_step_file_with_names_colors
    except ImportError as error:
        raise _import_occ_error() from error

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP file: {step_path}")
    scale_to_metres = _file_length_unit_scale(reader)

    def _to_metres(shape: Any) -> Any:
        if scale_to_metres == 1.0:
            return shape
        trsf = gp_Trsf()
        trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale_to_metres)
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()

    def _iter_solids(shape: Any) -> list[Any]:
        if shape is None or shape.IsNull():
            return []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        found = []
        while explorer.More():
            found.append(topods.Solid(explorer.Current()))
            explorer.Next()
        return found

    solids: list[tuple[str, Any]] = []
    try:
        named_shapes = read_step_file_with_names_colors(str(step_path))
    except Exception:
        named_shapes = {}

    for shape, label in named_shapes.items():
        name = str(label[0]).strip() if isinstance(label, (list, tuple)) and label else str(label).strip()
        part_solids = _iter_solids(shape)
        for index, solid in enumerate(part_solids):
            solid_name = name or "body"
            if len(part_solids) > 1:
                solid_name = f"{solid_name}_{index}"
            solids.append((solid_name, _to_metres(solid)))

    if not solids:
        reader.TransferRoots()
        for index, solid in enumerate(_iter_solids(reader.OneShape())):
            solids.append((f"body_{index}", _to_metres(solid)))

    if not solids:
        raise ValueError(f"No solids found in STEP file: {step_path}")

    return solids


def _solid_mass_properties(solid: Any, density: float) -> tuple[float, tuple[float, float, float], tuple[float, ...]]:
    """Compute mass, centre of mass, and central inertia tensor of a solid.

    Parameters
    ----------
    solid : TopoDS_Shape
        Solid to analyze, as returned by :func:`read_step_solids`.
    density : float
        Density in kg/m^3.

    Returns
    -------
    tuple[float, tuple[float, float, float], tuple[float, ...]]
        ``(mass, center_of_mass, inertia)`` where ``center_of_mass`` is an
        ``(x, y, z)`` triple in the solid's original (global) frame and
        ``inertia`` is ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`` about that centre.
    """
    try:
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
    except ImportError as error:
        raise _import_occ_error() from error

    # First pass (reference at the global origin) just to locate the centroid.
    origin_props = GProp_GProps()
    brepgprop.VolumeProperties(solid, origin_props)
    centroid = origin_props.CentreOfMass()
    center_of_mass = (centroid.X(), centroid.Y(), centroid.Z())

    # Second pass, referenced at the centroid, yields the *central* inertia
    # tensor directly (no manual parallel-axis correction needed). OCC always
    # computes these assuming unit density, so mass/inertia are scaled here.
    central_props = GProp_GProps(centroid)
    brepgprop.VolumeProperties(solid, central_props)
    volume = central_props.Mass()
    mass = volume * density
    matrix = central_props.MatrixOfInertia()
    inertia = tuple(
        density * value
        for value in (
            matrix.Value(1, 1),
            matrix.Value(2, 2),
            matrix.Value(3, 3),
            matrix.Value(1, 2),
            matrix.Value(1, 3),
            matrix.Value(2, 3),
        )
    )
    return mass, center_of_mass, inertia


def _combine_mass_properties(
    entries: list[tuple[float, tuple[float, float, float], tuple[float, ...]]],
) -> tuple[float, tuple[float, float, float], tuple[float, ...]]:
    """Combine several solids' mass properties into one rigid body's.

    Parameters
    ----------
    entries : list[tuple[float, tuple[float, float, float], tuple[float, ...]]]
        ``(mass, center_of_mass, inertia)`` triples, one per solid, as
        returned by :func:`_solid_mass_properties`; ``center_of_mass`` and
        ``inertia`` must share a common frame (e.g. the STEP file's global
        axes).

    Returns
    -------
    tuple[float, tuple[float, float, float], tuple[float, ...]]
        ``(total_mass, combined_center_of_mass, combined_inertia)``, with
        ``combined_inertia`` -- ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)`` -- taken
        about ``combined_center_of_mass`` by shifting each solid's tensor
        there with the parallel-axis theorem before summing.
    """
    total_mass = sum(mass for mass, _, _ in entries)
    combined_com = tuple(
        sum(mass * com[axis] for mass, com, _ in entries) / total_mass for axis in range(3)
    )

    ixx = iyy = izz = ixy = ixz = iyz = 0.0
    for mass, com, inertia in entries:
        dx, dy, dz = (com[axis] - combined_com[axis] for axis in range(3))
        i_xx, i_yy, i_zz, i_xy, i_xz, i_yz = inertia
        ixx += i_xx + mass * (dy**2 + dz**2)
        iyy += i_yy + mass * (dx**2 + dz**2)
        izz += i_zz + mass * (dx**2 + dy**2)
        ixy += i_xy - mass * dx * dy
        ixz += i_xz - mass * dx * dz
        iyz += i_yz - mass * dy * dz

    return total_mass, combined_com, (ixx, iyy, izz, ixy, ixz, iyz)


# Simbody's visualizer protocol refuses to draw a DecorativeMesh with more
# than this many vertices (VisualizerProtocol::drawPolygonalMesh), so a
# tessellation denser than this is unusable by :meth:`OpenSimModel.show`.
_MAX_VISUALIZER_MESH_VERTICES = 65535


def _mesh_vertex_count(shape: Any) -> int:
    """Return the STL vertex count (3 per triangle, unshared) across ``shape``'s faces.

    STL has no vertex indexing, so each triangle contributes its own 3
    vertices -- this is what the Simbody visualizer counts against its
    :data:`_MAX_VISUALIZER_MESH_VERTICES` cap, not OCC's deduplicated
    per-face triangulation node count.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location

    total = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(explorer.Current(), location)
        if triangulation is not None:
            total += 3 * triangulation.NbTriangles()
        explorer.Next()
    return total


def _split_stl_by_facet_count(stl_path: Path, max_vertices: int) -> list[Path]:
    """Split an ASCII STL into sibling files with at most ``max_vertices // 3`` facets each.

    Used when a single tessellation has more facets than Simbody's
    visualizer can display in one ``DecorativeMesh``: some parts (many small
    faces, e.g. numerous holes/fillets) can't be brought under that cap by
    coarsening deflection alone, since each face still needs a minimum
    triangle count. Splitting loses no geometry -- it only distributes the
    same triangles across multiple mesh files/geometries on the same body.
    """
    facets = re.findall(r"facet normal.*?endfacet\n?", stl_path.read_text(), re.S)
    max_facets = max(1, max_vertices // 3)
    if len(facets) <= max_facets:
        return [stl_path]

    paths = []
    for index in range(0, len(facets), max_facets):
        chunk = facets[index : index + max_facets]
        chunk_path = (
            stl_path if index == 0 else stl_path.with_stem(f"{stl_path.stem}_{index // max_facets + 1}")
        )
        chunk_path.write_text(f"solid {stl_path.stem}\n" + "".join(chunk) + "endsolid\n")
        paths.append(chunk_path)
    return paths


def _write_solid_mesh(
    solid: Any,
    center_of_mass: tuple[float, float, float],
    destination_path: str | Path,
    linear_deflection: float,
    angular_deflection: float,
) -> list[Path]:
    """Tessellate a solid and write it as one or more STL meshes centred on its origin.

    The solid is translated by ``-center_of_mass`` before tessellation, so
    the resulting mesh is expressed in the body-local frame with its origin
    at the solid's centre of mass -- matching the ``mass_center = (0, 0, 0)``
    convention used when building the corresponding ``opensim.Body``.

    If the resulting tessellation has more vertices than Simbody's
    visualizer can display in a single mesh, it is split across sibling
    files (see :func:`_split_stl_by_facet_count`); this only affects how
    the *display* geometry is chunked, not the mass properties, which are
    computed separately from the original solid.

    Parameters
    ----------
    solid : TopoDS_Shape
        Solid to mesh, as returned by :func:`read_step_solids`.
    center_of_mass : tuple[float, float, float]
        Centre of mass in the solid's original (global) frame, as returned
        by :func:`solid_mass_properties`.
    destination_path : str or pathlib.Path
        Output ``.stl`` file path.
    linear_deflection, angular_deflection : float
        Tessellation tolerances (metres, radians): smaller values produce
        finer, larger meshes.

    Returns
    -------
    list[pathlib.Path]
        Paths of the STL file(s) written, all alongside ``destination_path``.
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.gp import gp_Trsf, gp_Vec
        from OCC.Core.StlAPI import StlAPI_Writer
    except ImportError as error:
        raise _import_occ_error() from error

    translation = gp_Trsf()
    translation.SetTranslation(gp_Vec(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2]))
    local_solid = BRepBuilderAPI_Transform(solid, translation, True).Shape()
    BRepMesh_IncrementalMesh(local_solid, linear_deflection, False, angular_deflection, True)

    destination_path = Path(destination_path)
    writer = StlAPI_Writer()
    writer.Write(local_solid, str(destination_path))

    if _mesh_vertex_count(local_solid) <= _MAX_VISUALIZER_MESH_VERTICES:
        return [destination_path]
    return _split_stl_by_facet_count(destination_path, _MAX_VISUALIZER_MESH_VERTICES)
