"""STL generation for simple primitive shapes (box, cylinder, sphere).

Internal helper for :mod:`opensim_models.operators`'s ``add_box_body``,
``add_cylinder_body`` and ``add_sphere_body``: every shape is written
centred on the origin (matching the ``mass_center = (0, 0, 0)`` convention
used for the ``opensim.Body`` the mesh is attached to), in metres.
"""

from __future__ import annotations

import math
from pathlib import Path


def _write_stl(destination_path: Path, name: str, triangles: list[tuple[tuple, tuple]]) -> None:
    """Write ``triangles`` (``(normal, (v0, v1, v2))`` tuples) as an ASCII STL."""
    lines = [f"solid {name}"]
    for normal, (v0, v1, v2) in triangles:
        lines.append(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}")
        lines.append("    outer loop")
        for vertex in (v0, v1, v2):
            lines.append(f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    destination_path.write_text("\n".join(lines) + "\n")


def write_box_mesh(
    destination_path: Path, size_x: float, size_y: float, size_z: float
) -> None:
    """Write a watertight, axis-aligned box STL (metres), centred on the origin."""
    hx, hy, hz = size_x / 2.0, size_y / 2.0, size_z / 2.0

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

    triangles = []
    for normal, quad in faces:
        points = [corner(signs) for signs in quad]
        triangles.append((normal, (points[0], points[1], points[2])))
        triangles.append((normal, (points[0], points[2], points[3])))
    _write_stl(destination_path, "box", triangles)


def write_cylinder_mesh(
    destination_path: Path, radius: float, height: float, *, segments: int = 32
) -> None:
    """Write a watertight cylinder STL (metres), centred on the origin.

    The cylinder's axis runs along the local Y axis (matching
    ``opensim.Cylinder``'s own convention), spanning ``[-height/2, height/2]``.
    """
    if segments < 3:
        raise ValueError("segments must be at least 3")
    half_height = height / 2.0
    angles = [2.0 * math.pi * i / segments for i in range(segments)]
    top_ring = [(radius * math.cos(a), half_height, radius * math.sin(a)) for a in angles]
    bottom_ring = [(radius * math.cos(a), -half_height, radius * math.sin(a)) for a in angles]
    top_center = (0.0, half_height, 0.0)
    bottom_center = (0.0, -half_height, 0.0)

    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        # Side wall: two triangles per segment, outward-facing normal.
        normal = (math.cos(angles[i]), 0.0, math.sin(angles[i]))
        triangles.append((normal, (bottom_ring[i], bottom_ring[j], top_ring[j])))
        triangles.append((normal, (bottom_ring[i], top_ring[j], top_ring[i])))
        # Caps, facing outward along +/-Y.
        triangles.append(((0.0, 1.0, 0.0), (top_center, top_ring[i], top_ring[j])))
        triangles.append(((0.0, -1.0, 0.0), (bottom_center, bottom_ring[j], bottom_ring[i])))
    _write_stl(destination_path, "cylinder", triangles)


def write_sphere_mesh(
    destination_path: Path, radius: float, *, segments: int = 24, rings: int = 12
) -> None:
    """Write a watertight UV-sphere STL (metres), centred on the origin."""
    if segments < 3 or rings < 2:
        raise ValueError("segments must be at least 3 and rings at least 2")

    def point(ring_index: int, segment_index: int) -> tuple[float, float, float]:
        theta = math.pi * ring_index / rings  # 0 (north pole) .. pi (south pole)
        phi = 2.0 * math.pi * segment_index / segments
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.cos(theta)
        z = radius * math.sin(theta) * math.sin(phi)
        return (x, y, z)

    def normal_at(p: tuple[float, float, float]) -> tuple[float, float, float]:
        length = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) or 1.0
        return (p[0] / length, p[1] / length, p[2] / length)

    triangles = []
    for ring_index in range(rings):
        for segment_index in range(segments):
            next_segment = (segment_index + 1) % segments
            p_top_left = point(ring_index, segment_index)
            p_top_right = point(ring_index, next_segment)
            p_bottom_left = point(ring_index + 1, segment_index)
            p_bottom_right = point(ring_index + 1, next_segment)

            if ring_index > 0:  # top ring degenerates to a single pole point
                triangles.append(
                    (normal_at(p_top_left), (p_top_left, p_bottom_left, p_bottom_right))
                )
            if ring_index < rings - 1:  # bottom ring degenerates to a single pole point
                triangles.append(
                    (normal_at(p_top_left), (p_top_left, p_bottom_right, p_top_right))
                )
    _write_stl(destination_path, "sphere", triangles)
