import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Screen
from opensim_models.models.screen.screen import (
    _MESH_FILENAME,
    _MESHES_DIR,
    _PLEXIGLASS_DENSITY_KG_M3,
    _THICKNESS_MM,
)

opensim = pytest.importorskip("opensim")


def joint_offset_frames(screen):
    joint = screen.model.getJointSet().get("screen_panel_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    child = opensim.PhysicalOffsetFrame.safeDownCast(joint.getChildFrame())
    return parent, child


def expected_mass_kg(width_mm, height_mm):
    volume_m3 = (width_mm / 1000.0) * (height_mm / 1000.0) * (_THICKNESS_MM / 1000.0)
    return volume_m3 * _PLEXIGLASS_DENSITY_KG_M3


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_default_size_comes_from_22in_16_9_diagonal():
    screen = Screen()

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    assert screen.width_mm is None
    assert screen.height_mm is None
    body = screen.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_explicit_size_overrides_diagonal_and_ratio():
    screen = Screen(width_mm=500.0, height_mm=300.0)

    assert screen.width_mm == 500.0
    assert screen.height_mm == 300.0
    body = screen.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(expected_mass_kg(500.0, 300.0), rel=1e-6)


def test_only_one_of_width_height_falls_back_to_diagonal():
    screen = Screen(width_mm=500.0)

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    body = screen.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_neither_sizing_method_usable_raises():
    with pytest.raises(ValueError, match="inches and ratio"):
        Screen(inches=None, ratio=None)


def test_invalid_ratio_string_raises():
    with pytest.raises(ValueError, match="aspect ratio"):
        Screen(ratio="16-9")
    with pytest.raises(ValueError, match="aspect ratio"):
        Screen(ratio="-16:9")


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def test_center_and_angle_set_the_joint_offset():
    screen = Screen(center_x=1.0, center_y=2.0, center_z=3.0, angle_deg=45.0)

    parent, _ = joint_offset_frames(screen)
    translation = parent.get_translation()
    orientation = parent.get_orientation()
    assert (translation[0], translation[1], translation[2]) == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert orientation[0] == pytest.approx(math.radians(45.0 - 90.0))


def test_default_angle_stands_the_panel_upright():
    screen = Screen()

    parent, _ = joint_offset_frames(screen)
    assert parent.get_orientation()[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------


def test_model_has_a_single_body_welded_to_ground():
    screen = Screen()

    assert screen.model.getBodySet().getSize() == 1
    assert screen.model.getJointSet().getSize() == 1
    joint = screen.model.getJointSet().get("screen_panel_joint")
    assert opensim.WeldJoint.safeDownCast(joint) is not None


def test_instances_are_backed_by_independent_opensim_models():
    small = Screen(width_mm=100.0, height_mm=100.0)
    large = Screen(width_mm=900.0, height_mm=900.0)

    assert small.model is not large.model
    assert small.model.getBodySet().get("screen_panel").get_mass() != pytest.approx(
        large.model.getBodySet().get("screen_panel").get_mass()
    )


def test_copy_preserves_type_and_parameters_without_a_screen_override():
    screen = Screen(width_mm=500.0, height_mm=300.0, center_x=1.0)

    duplicate = screen.copy()

    assert type(duplicate) is Screen
    assert duplicate.width_mm == 500.0
    assert duplicate.center_x == 1.0
    assert duplicate.model is not screen.model


# ---------------------------------------------------------------------------
# Setters rebuild the panel and its mesh
# ---------------------------------------------------------------------------


def test_setters_expose_getters_as_properties():
    screen = Screen()

    screen.set_width_mm(500.0)
    screen.set_height_mm(300.0)
    screen.set_inches(24.0)
    screen.set_ratio("4:3")
    screen.set_center_x(1.0)
    screen.set_center_y(2.0)
    screen.set_center_z(3.0)
    screen.set_angle_deg(30.0)

    assert screen.width_mm == 500.0
    assert screen.height_mm == 300.0
    assert screen.inches == 24.0
    assert screen.ratio == "4:3"
    assert screen.center_x == 1.0
    assert screen.center_y == 2.0
    assert screen.center_z == 3.0
    assert screen.angle_deg == 30.0


def test_switching_back_to_diagonal_sizing_after_explicit_size():
    screen = Screen(width_mm=500.0, height_mm=300.0)

    screen.set_width_mm(None)
    screen.set_height_mm(None)
    screen.set_inches(22.0)
    screen.set_ratio("16:9")

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    body = screen.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_size_setter_regenerates_the_mesh_on_disk():
    screen = Screen(width_mm=500.0, height_mm=300.0)
    mesh_path = _MESHES_DIR / _MESH_FILENAME
    first_contents = mesh_path.read_text()

    screen.set_width_mm(700.0)
    screen.set_height_mm(400.0)

    assert mesh_path.read_text() != first_contents
    assert mesh_path.read_text().count("facet normal") == 12


def test_screen_registers_its_own_mesh_directory_for_show():
    screen = Screen()

    assert _MESHES_DIR in screen.geometry_directories
    assert screen.visualizer is None
