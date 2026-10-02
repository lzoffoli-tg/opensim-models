import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel, operators
from opensim_models.ensemble import OpenSimEnsemble

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
    assert len(model.bodies) == 0
    assert len(model.joints) == 0
    assert len(model.muscles) == 0
    assert len(model.markers) == 0
    assert len(model.coordinates) == 0
    assert model.state is not None


def test_model_loaded_from_file_exposes_expected_components():
    model = make_model()

    assert len(model.bodies) == 22
    assert len(model.joints) == 22
    assert len(model.muscles) == 80
    assert len(model.markers) == 66


def test_missing_model_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenSimModel(model_path="does/not/exist.osim")


def test_locked_coordinates_are_unlocked_on_load():
    model = make_model()

    assert model.coordinate("subtalar_angle_l").locked is False
    assert model.coordinate("mtp_angle_r").locked is False
    assert model.coordinate("wrist_flex_r").locked is False


def test_instances_loaded_from_the_same_file_are_independent():
    first = make_model()
    second = make_model()

    assert first.model is not second.model
    first.coordinate("hip_flexion_r").set_value_degrees(25.0)
    assert second.coordinate("hip_flexion_r").value_degrees == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_returns_an_instance_of_the_same_type():
    model = make_model()

    assert type(model.copy()) is type(model)


def test_copy_is_backed_by_an_independent_opensim_model():
    model = make_model()

    duplicate = model.copy()

    assert duplicate.model is not model.model
    duplicate.coordinate("hip_flexion_r").set_value_degrees(25.0)
    assert model.coordinate("hip_flexion_r").value_degrees == pytest.approx(0.0)


def test_copy_preserves_posture_and_speed():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    duplicate = model.copy()

    assert duplicate.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert duplicate.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_copy_preserves_a_coupled_dependent_coordinate():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    duplicate = model.copy()

    assert duplicate.coordinate("knee_angle_r_beta").value_degrees == pytest.approx(90.0)


def test_copy_carries_over_geometry_directories_independently():
    model = make_model()
    model.add_geometry_directory("some/dir")

    duplicate = model.copy()
    duplicate.add_geometry_directory("other/dir")

    assert Path("other/dir") not in model.geometry_directories


# ---------------------------------------------------------------------------
# update_state()
# ---------------------------------------------------------------------------


def test_coordinate_setters_do_not_auto_realize():
    model = make_model()
    tibia = model.body("tibia_r")
    origin = opensim.Vec3(0, 0, 0)
    tibia.raw.findStationLocationInGround(model.state, origin)  # baseline: works

    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    # The raw value is set immediately (no realize needed to read it back)...
    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    # ...but nothing derived is recomputed: OpenSim's own cache-versioning
    # catches this and raises rather than silently returning a stale value.
    with pytest.raises(RuntimeError):
        tibia.raw.findStationLocationInGround(model.state, origin)

    model.update_state()
    tibia.raw.findStationLocationInGround(model.state, origin)  # works again


def test_update_state_realizes_position_and_velocity():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    model.update_state()

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_update_state_equilibrates_muscles_without_crashing():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    model.update_state()  # must not raise or crash the process

    assert len(model.muscles) > 0


# ---------------------------------------------------------------------------
# Named-component accessors
# ---------------------------------------------------------------------------


def test_named_component_accessors_return_matching_objects():
    model = make_model()

    assert model.body("pelvis").name == "pelvis"
    assert model.joint("hip_r").name == "hip_r"
    assert model.muscle("glmax1_r").name == "glmax1_r"
    assert model.marker("RASI").name == "RASI"
    assert model.coordinate("hip_flexion_r").name == "hip_flexion_r"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_coordinate_degrees_round_trip_through_radians():
    model = make_model()

    model.coordinate("hip_flexion_r").set_value_degrees(25.0)

    assert model.coordinate("hip_flexion_r").value_degrees == pytest.approx(25.0)


def test_set_coordinate_degrees_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.coordinate("hip_flexion_r").set_value_degrees(float("nan"))


def test_coordinate_locking_can_be_toggled_and_is_enforced():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    coordinate.set_locked(True)
    assert coordinate.locked is True
    with pytest.raises(ValueError, match="locked"):
        coordinate.set_value_degrees(10.0)

    coordinate.set_locked(False)
    coordinate.set_value_degrees(10.0)
    assert coordinate.value_degrees == pytest.approx(10.0)


def test_coordinate_range_can_be_read_and_updated():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    # Coordinate.set_range/.range are in the coordinate's native unit
    # (radians, for this rotational coordinate), not degrees -- convert at
    # the test boundary so the assertions can still be expressed in degrees.
    coordinate.set_range((np.radians(-100.0), np.radians(5.0)))

    assert tuple(np.degrees(coordinate.range)) == pytest.approx((-100.0, 5.0))
    with pytest.raises(ValueError, match="min must be less than max"):
        coordinate.set_range((np.radians(5.0), np.radians(-100.0)))


def test_coordinate_range_rejects_non_finite_bounds():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    with pytest.raises(ValueError, match="finite"):
        coordinate.set_range((float("nan"), np.radians(5.0)))


def test_coordinate_speed_degrees_round_trip_through_radians():
    model = make_model()

    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_set_coordinate_speed_degrees_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.coordinate("hip_flexion_r").set_speed_degrees(float("nan"))


def test_set_coordinate_speed_degrees_is_rejected_when_locked():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")
    coordinate.set_locked(True)

    with pytest.raises(ValueError, match="locked"):
        coordinate.set_speed_degrees(10.0)


# ---------------------------------------------------------------------------
# Body properties
# ---------------------------------------------------------------------------


def test_body_mass_can_be_read_and_updated():
    model = make_model()

    model.body("tibia_r").set_mass(5.0)

    assert model.body("tibia_r").mass == pytest.approx(5.0)


def test_set_body_mass_rejects_non_positive_values():
    model = make_model()
    tibia = model.body("tibia_r")

    with pytest.raises(ValueError, match="positive"):
        tibia.set_mass(0.0)
    with pytest.raises(ValueError, match="positive"):
        tibia.set_mass(-1.0)


def test_set_body_mass_preserves_posture():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    model.body("tibia_r").set_mass(5.0)

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Reinitializing the system (structural property changes)
# ---------------------------------------------------------------------------


def test_reinitialize_preserves_posture_and_speed():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    model.reinitialize()

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_reinitialize_updates_a_coupled_dependent_coordinate():
    model = make_model()
    # knee_angle_r_beta (the patella) is coupled to knee_angle_r via a
    # CoordinateCouplerConstraint in the bundled Rajagopal model.
    assert "knee_angle_r_beta" in model.coordinates

    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.reinitialize()

    assert model.coordinate("knee_angle_r_beta").value_degrees == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def test_marker_location_can_be_read_and_updated():
    model = make_model()

    model.marker("RTOE").set_location((0.1, 0.02, 0.03))

    assert model.marker("RTOE").location == pytest.approx((0.1, 0.02, 0.03))


def test_set_marker_location_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.marker("RTOE").set_location((float("inf"), 0.0, 0.0))


# ---------------------------------------------------------------------------
# Muscles
# ---------------------------------------------------------------------------


def test_muscle_mechanical_parameters_can_be_read_and_updated():
    model = make_model()
    muscle = model.muscle("addbrev_r")

    muscle.set_max_isometric_force(1500.0)
    muscle.set_optimal_fiber_length(0.15)
    muscle.set_tendon_slack_length(0.1)
    muscle.set_pennation_angle(10.0)

    assert muscle.max_isometric_force == pytest.approx(1500.0)
    assert muscle.optimal_fiber_length == pytest.approx(0.15)
    assert muscle.tendon_slack_length == pytest.approx(0.1)
    assert muscle.pennation_angle == pytest.approx(10.0)


@pytest.mark.parametrize(
    "setter, value",
    [
        ("set_max_isometric_force", 0.0),
        ("set_optimal_fiber_length", -0.1),
        ("set_tendon_slack_length", float("nan")),
    ],
)
def test_muscle_setters_reject_non_positive_or_non_finite_values(setter, value):
    model = make_model()
    muscle = model.muscle("addbrev_r")

    with pytest.raises(ValueError):
        getattr(muscle, setter)(value)


def test_muscle_pennation_angle_must_be_within_valid_range():
    model = make_model()

    with pytest.raises(ValueError, match="pennation angle"):
        model.muscle("addbrev_r").set_pennation_angle(90.0)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_scale_bodies_applies_positive_factors():
    model = make_model()
    baseline = model.marker("RTOE").location

    model.scale_bodies({"calcn_r": (1.2, 1.2, 1.2), "toes_r": (1.2, 1.2, 1.2)})

    assert model.marker("RTOE").location != pytest.approx(baseline)


def test_scale_bodies_ignores_unknown_body_names():
    model = make_model()

    model.scale_bodies({"not_a_real_body": (1.5, 1.5, 1.5)})

    assert len(model.bodies) == 22


def test_scale_bodies_rejects_non_positive_factors():
    model = make_model()

    with pytest.raises(ValueError):
        model.scale_bodies({"pelvis": (1.0, -1.0, 1.0)})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_a_reloadable_osim_file_with_current_posture():
    model = make_model()
    model.coordinate("hip_flexion_r").set_value_degrees(25.0)

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
        model.body("part_body").raw.attachGeometry(opensim.Mesh("part.stl"))
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

    assert len(base.bodies) == 44
    assert len(base.joints) == 44
    assert len(base.muscles) == 80 * 2
    assert len(base.markers) == 66 * 2
    assert "extra_pelvis" in base.bodies
    # the original operand is left untouched
    assert len(extra.bodies) == 22
    assert "extra_pelvis" not in extra.bodies


def test_add_model_default_prefix_is_the_operand_class_name():
    base = make_model()
    extra = make_model()

    base.add_model(extra)

    assert "opensimmodel_pelvis" in base.bodies


def test_add_model_does_not_rename_non_colliding_components():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra = build_single_body_model(tmp_dir, "gadget", "gadget_to_ground")

        base.add_model(extra, name="extra")

        assert len(base.bodies) == 2
        assert "widget" in base.bodies
        assert "gadget" in base.bodies  # kept as-is, no collision


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


def test_add_model_preserves_the_merged_operands_posture():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = OpenSimModel(model_path=None)
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        # Just need *some* coordinate name; .coordinates is a dict (no
        # positional index), so go straight to the raw CoordinateSet for
        # this one arbitrary, order-dependent lookup.
        coordinate_name = extra.model.getCoordinateSet().get(0).getName()
        extra.coordinate(coordinate_name).set_value_degrees(30.0)

        base.add_model(extra)

        assert base.coordinate(coordinate_name).value_degrees == pytest.approx(30.0)


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

    assert len(base.bodies) == 0
    assert len(base.joints) == 0
    assert len(base.muscles) == 0
    assert len(base.markers) == 0


def test_remove_model_preserves_the_remaining_posture():
    base = make_model()
    extra = OpenSimModel(model_path=None)
    base.coordinate("knee_angle_r").set_value_degrees(90.0)

    base.add_model(extra)
    base.remove_model(extra)

    assert base.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)


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


# `+` no longer merges operands into a new OpenSimModel: it now returns an
# OpenSimEnsemble, keeping both containers separate/independently editable
# (see opensim_models.ensemble). The permanent, in-place merge these tests
# used to exercise via `+` still exists, just moved to add_model() directly
# (covered above); these three tests instead verify the new operator's own
# contract -- it builds a non-mutating ensemble of the original containers,
# and ensemble.combined() still produces the same merged result on demand.


def test_add_operator_returns_a_new_ensemble_without_mutating_operands():
    first = make_model()
    second = make_model()

    combined = first + second

    assert isinstance(combined, OpenSimEnsemble)
    assert combined.containers == (first, second)
    assert len(combined.combined().bodies) == len(first.bodies) + len(second.bodies)
    assert len(first.bodies) == 22
    assert len(second.bodies) == 22


def test_radd_delegates_to_the_left_operand():
    first = make_model()
    second = make_model()

    # first.__radd__(second) supports `second + first` when second doesn't
    # implement __add__; the resulting ensemble keeps second's containers
    # first (matching `second + first`'s left-to-right order).
    combined = first.__radd__(second)

    assert isinstance(combined, OpenSimEnsemble)
    assert combined.containers == (second, first)
    assert len(combined.combined().bodies) == len(first.bodies) + len(second.bodies)


def test_chained_add_merges_three_models():
    first = make_model()
    second = make_model()
    third = make_model()

    combined = first + second + third

    assert isinstance(combined, OpenSimEnsemble)
    assert combined.containers == (first, second, third)  # flattened, not nested
    assert len(combined.combined().bodies) == 22 * 3


# ---------------------------------------------------------------------------
# rotate / translate
# ---------------------------------------------------------------------------


def build_one_body_model(position=(1.0, 0.0, 0.0)):
    model = OpenSimModel(model_path=None)
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=position)
    return model, body


def test_rotate_delegates_to_rotate_object():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    new_position = model.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_translate_delegates_to_translate_object():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    new_position = model.translate((0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_rotate_not_inplace_returns_a_rotated_copy_and_leaves_self_untouched():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    rotated_copy = model.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False)

    assert isinstance(rotated_copy, OpenSimModel)
    assert rotated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = rotated_copy.body("b1").raw.getPositionInGround(rotated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_translate_not_inplace_returns_a_translated_copy_and_leaves_self_untouched():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    translated_copy = model.translate((0.0, 2.0, 0.0), inplace=False)

    assert isinstance(translated_copy, OpenSimModel)
    assert translated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = translated_copy.body("b1").raw.getPositionInGround(translated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
