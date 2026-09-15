import tempfile
from pathlib import Path

import numpy as np
import pytest

from opensim_models import OpenSimModel

# DEFAULT_MODEL_PATH is User's internal default, reused here as a real .osim
# fixture to test OpenSimModel's generic facade rather than User specifically.
from opensim_models.models.user.user import DEFAULT_MODEL_PATH

opensim = pytest.importorskip("opensim")

REAL_MODEL = DEFAULT_MODEL_PATH


def make_model(**kwargs):
    return OpenSimModel(model_path=REAL_MODEL, **kwargs)


def build_single_body_model(directory, body_name, joint_name):
    """Build a tiny synthetic one-body OpenSim model file for merge tests
    that must not depend on any specific bundled model."""
    raw = opensim.Model()
    body = opensim.Body(body_name, 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    raw.addBody(body)
    joint = opensim.FreeJoint(joint_name, raw.getGround(), body)
    raw.addJoint(joint)
    raw.finalizeConnections()
    raw.initSystem()
    path = Path(directory) / f"{body_name}.osim"
    raw.printToXML(str(path))
    return OpenSimModel(model_path=path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_blank_model_has_no_components():
    model = OpenSimModel(model_path=None)

    assert model.model_path is None
    assert model.bodies.getSize() == 0
    assert model.joints.getSize() == 0
    assert model.muscles.getSize() == 0
    assert model.markers.getSize() == 0
    assert model.coordinates.getSize() == 0
    assert model.state is not None


def test_model_loaded_from_file_exposes_expected_components():
    model = make_model()

    assert model.bodies.getSize() == 22
    assert model.joints.getSize() == 22
    assert model.muscles.getSize() == 80
    assert model.markers.getSize() == 66


def test_missing_model_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenSimModel(model_path="does/not/exist.osim")


def test_locked_coordinates_are_unlocked_on_load():
    model = make_model()

    assert model.coordinate_locked("subtalar_angle_l") is False
    assert model.coordinate_locked("mtp_angle_r") is False
    assert model.coordinate_locked("wrist_flex_r") is False


def test_instances_loaded_from_the_same_file_are_independent():
    first = make_model()
    second = make_model()

    assert first.model is not second.model
    first.set_coordinate_degrees("hip_flexion_r", 25.0)
    assert second.coordinate_degrees("hip_flexion_r") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Named-component accessors
# ---------------------------------------------------------------------------


def test_named_component_accessors_return_matching_objects():
    model = make_model()

    assert model.body("pelvis").getName() == "pelvis"
    assert model.joint("hip_r").getName() == "hip_r"
    assert model.muscle("glmax1_r").getName() == "glmax1_r"
    assert model.marker("RASI").getName() == "RASI"
    assert model.coordinate("hip_flexion_r").getName() == "hip_flexion_r"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_coordinate_degrees_round_trip_through_radians():
    model = make_model()

    model.set_coordinate_degrees("hip_flexion_r", 25.0)

    assert model.coordinate_degrees("hip_flexion_r") == pytest.approx(25.0)


def test_set_coordinate_degrees_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.set_coordinate_degrees("hip_flexion_r", float("nan"))


def test_coordinate_locking_can_be_toggled_and_is_enforced():
    model = make_model()

    model.set_coordinate_locked("knee_angle_r", True)
    assert model.coordinate_locked("knee_angle_r") is True
    with pytest.raises(ValueError, match="locked"):
        model.set_coordinate_degrees("knee_angle_r", 10.0)

    model.set_coordinate_locked("knee_angle_r", False)
    model.set_coordinate_degrees("knee_angle_r", 10.0)
    assert model.coordinate_degrees("knee_angle_r") == pytest.approx(10.0)


def test_coordinate_range_can_be_read_and_updated():
    model = make_model()

    model.set_coordinate_range("knee_angle_r", -100.0, 5.0)

    assert model.coordinate_range("knee_angle_r") == pytest.approx((-100.0, 5.0))
    with pytest.raises(ValueError, match="min_degrees"):
        model.set_coordinate_range("knee_angle_r", 5.0, -100.0)


def test_coordinate_range_rejects_non_finite_bounds():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.set_coordinate_range("knee_angle_r", float("nan"), 5.0)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def test_marker_location_can_be_read_and_updated():
    model = make_model()

    model.set_marker_location("RTOE", 0.1, 0.02, 0.03)

    assert model.marker_location("RTOE") == pytest.approx((0.1, 0.02, 0.03))


def test_set_marker_location_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.set_marker_location("RTOE", float("inf"), 0.0, 0.0)


# ---------------------------------------------------------------------------
# Muscles
# ---------------------------------------------------------------------------


def test_muscle_mechanical_parameters_can_be_read_and_updated():
    model = make_model()

    model.set_muscle_max_isometric_force("addbrev_r", 1500.0)
    model.set_muscle_optimal_fiber_length("addbrev_r", 0.15)
    model.set_muscle_tendon_slack_length("addbrev_r", 0.1)
    model.set_muscle_pennation_angle("addbrev_r", 10.0)

    assert model.muscle_max_isometric_force("addbrev_r") == pytest.approx(1500.0)
    assert model.muscle_optimal_fiber_length("addbrev_r") == pytest.approx(0.15)
    assert model.muscle_tendon_slack_length("addbrev_r") == pytest.approx(0.1)
    assert model.muscle_pennation_angle("addbrev_r") == pytest.approx(10.0)


@pytest.mark.parametrize(
    "setter, value",
    [
        ("set_muscle_max_isometric_force", 0.0),
        ("set_muscle_optimal_fiber_length", -0.1),
        ("set_muscle_tendon_slack_length", float("nan")),
    ],
)
def test_muscle_setters_reject_non_positive_or_non_finite_values(setter, value):
    model = make_model()

    with pytest.raises(ValueError):
        getattr(model, setter)("addbrev_r", value)


def test_muscle_pennation_angle_must_be_within_valid_range():
    model = make_model()

    with pytest.raises(ValueError, match="pennation angle"):
        model.set_muscle_pennation_angle("addbrev_r", 90.0)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_scale_bodies_applies_positive_factors():
    model = make_model()
    baseline = model.marker_location("RTOE")

    model.scale_bodies({"calcn_r": (1.2, 1.2, 1.2), "toes_r": (1.2, 1.2, 1.2)})

    assert model.marker_location("RTOE") != pytest.approx(baseline)


def test_scale_bodies_ignores_unknown_body_names():
    model = make_model()

    model.scale_bodies({"not_a_real_body": (1.5, 1.5, 1.5)})

    assert model.bodies.getSize() == 22


def test_scale_bodies_rejects_non_positive_factors():
    model = make_model()

    with pytest.raises(ValueError):
        model.scale_bodies({"pelvis": (1.0, -1.0, 1.0)})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_a_reloadable_osim_file_with_current_posture():
    model = make_model()
    model.set_coordinate_degrees("hip_flexion_r", 25.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = model.export(Path(tmp_dir) / "exported.osim")

        assert destination.is_file()
        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        hip_flexion_r = reloaded.getCoordinateSet().get("hip_flexion_r")
        assert hip_flexion_r.get_default_value() == pytest.approx(np.deg2rad(25.0))


def test_export_copies_referenced_meshes_into_a_geometry_folder():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        geometry_dir = tmp_path / "assets"
        geometry_dir.mkdir()
        mesh_file = geometry_dir / "part.stl"
        mesh_file.write_bytes(b"solid fake\nendsolid fake\n")

        model = build_single_body_model(tmp_path, "part_body", "part_joint")
        model.body("part_body").attachGeometry(opensim.Mesh("part.stl"))
        model.add_geometry_directory(geometry_dir)

        export_dir = tmp_path / "exported"
        export_dir.mkdir()
        destination = model.export(export_dir / "model.osim")

        exported_mesh = destination.parent / "Geometry" / "part.stl"
        assert exported_mesh.is_file()
        assert exported_mesh.read_bytes() == mesh_file.read_bytes()


def test_export_without_referenced_meshes_creates_no_geometry_folder():
    model = make_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = model.export(Path(tmp_dir) / "exported.osim")

        assert not (destination.parent / "Geometry").exists()


# ---------------------------------------------------------------------------
# Geometry / visualizer
# ---------------------------------------------------------------------------


def test_geometry_directories_start_empty_and_accept_registrations():
    model = OpenSimModel(model_path=None)

    assert model.geometry_directories == ()
    assert model.visualizer is None

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.add_geometry_directory(tmp_dir)

        assert model.geometry_directories == (Path(tmp_dir),)


# ---------------------------------------------------------------------------
# Composing models: add_model / remove_model / __add__ / __radd__
# ---------------------------------------------------------------------------


def test_add_model_renames_colliding_components_with_a_prefix():
    base = make_model()
    extra = make_model()  # loaded from the same file: every name collides

    base.add_model(extra, name="extra")

    assert base.bodies.getSize() == 44
    assert base.joints.getSize() == 44
    assert base.muscles.getSize() == 80 * 2
    assert base.markers.getSize() == 66 * 2
    assert base.bodies.getIndex("extra_pelvis") >= 0
    # the original operand is left untouched
    assert extra.bodies.getSize() == 22
    assert extra.bodies.getIndex("extra_pelvis") == -1


def test_add_model_default_prefix_is_the_operand_class_name():
    base = make_model()
    extra = make_model()

    base.add_model(extra)

    assert base.bodies.getIndex("opensimmodel_pelvis") >= 0


def test_add_model_does_not_rename_non_colliding_components():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra = build_single_body_model(tmp_dir, "gadget", "gadget_to_ground")

        base.add_model(extra, name="extra")

        assert base.bodies.getSize() == 2
        assert base.bodies.getIndex("widget") >= 0
        assert base.bodies.getIndex("gadget") >= 0  # kept as-is, no collision


def test_add_model_merged_model_initializes_and_exports():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")

        base.add_model(extra, name="extra")
        destination = Path(tmp_dir) / "combined.osim"
        base.export(destination)

        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        assert reloaded.getBodySet().getSize() == 2


def test_add_model_carries_over_geometry_directories():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = OpenSimModel(model_path=None)
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra.add_geometry_directory(tmp_dir)

        base.add_model(extra)

        assert Path(tmp_dir) in base.geometry_directories


def test_add_model_then_remove_model_restores_original_state():
    base = OpenSimModel(model_path=None)
    extra = make_model()

    base.add_model(extra)
    base.remove_model(extra)

    assert base.bodies.getSize() == 0
    assert base.joints.getSize() == 0
    assert base.muscles.getSize() == 0
    assert base.markers.getSize() == 0


def test_remove_model_rejects_a_model_that_was_never_added():
    base = OpenSimModel(model_path=None)
    stranger = make_model()

    with pytest.raises(ValueError):
        base.remove_model(stranger)


@pytest.mark.parametrize(
    "operation",
    [
        lambda model: model.add_model("not a model"),
        lambda model: model.remove_model("not a model"),
        lambda model: model + "not a model",
        lambda model: "not a model" + model,
    ],
)
def test_merge_operations_reject_non_model_operands(operation):
    model = OpenSimModel(model_path=None)

    with pytest.raises(TypeError):
        operation(model)


def test_add_operator_returns_a_new_model_without_mutating_operands():
    first = make_model()
    second = make_model()

    combined = first + second

    assert isinstance(combined, OpenSimModel)
    assert combined is not first
    assert combined is not second
    assert combined.bodies.getSize() == first.bodies.getSize() + second.bodies.getSize()
    assert first.bodies.getSize() == 22
    assert second.bodies.getSize() == 22


def test_radd_delegates_to_the_left_operand():
    first = make_model()
    second = make_model()

    combined = first.__radd__(second)

    assert combined.bodies.getSize() == first.bodies.getSize() + second.bodies.getSize()


def test_chained_add_merges_three_models():
    first = make_model()
    second = make_model()
    third = make_model()

    combined = first + second + third

    assert combined.bodies.getSize() == 22 * 3
