import tempfile
from pathlib import Path

import numpy as np
import pytest

from opensim_models import OpenSimModel, User

# User only exposes DEFAULT_DATASET/DEFAULT_MESHES_DIR/load_ansur/resolve_reference
# internally; tests reach into the implementation modules directly to exercise them.
from opensim_models.models.user._data import DEFAULT_DATASET, load_ansur, resolve_reference
from opensim_models.models.user.user import DEFAULT_MESHES_DIR

try:
    import opensim
except ImportError:
    opensim = None

requires_opensim = pytest.mark.skipif(
    opensim is None, reason="OpenSim bindings are not installed"
)

DATASET = DEFAULT_DATASET

POSTURE_SETTERS = [
    ("set_left_hip_flexionextension", "hip_flexion_l"),
    ("set_right_hip_flexionextension", "hip_flexion_r"),
    ("set_left_hip_adduction", "hip_adduction_l"),
    ("set_right_hip_adduction", "hip_adduction_r"),
    ("set_left_hip_rotation", "hip_rotation_l"),
    ("set_right_hip_rotation", "hip_rotation_r"),
    ("set_left_knee_flexionextension", "knee_angle_l"),
    ("set_right_knee_flexionextension", "knee_angle_r"),
    ("set_left_ankle_flexiondorsiflexion", "ankle_angle_l"),
    ("set_right_ankle_flexiondorsiflexion", "ankle_angle_r"),
    ("set_left_subtalar_inversion", "subtalar_angle_l"),
    ("set_right_subtalar_inversion", "subtalar_angle_r"),
    ("set_left_mtp_flexion", "mtp_angle_l"),
    ("set_right_mtp_flexion", "mtp_angle_r"),
    ("set_left_shoulder_flexion", "arm_flex_l"),
    ("set_right_shoulder_flexion", "arm_flex_r"),
    ("set_left_shoulder_adduction", "arm_add_l"),
    ("set_right_shoulder_adduction", "arm_add_r"),
    ("set_left_shoulder_rotation", "arm_rot_l"),
    ("set_right_shoulder_rotation", "arm_rot_r"),
    ("set_left_elbow_flexion", "elbow_flex_l"),
    ("set_right_elbow_flexion", "elbow_flex_r"),
    ("set_left_wrist_flexion", "wrist_flex_l"),
    ("set_right_wrist_flexion", "wrist_flex_r"),
    ("set_left_wrist_deviation", "wrist_dev_l"),
    ("set_right_wrist_deviation", "wrist_dev_r"),
    ("set_left_forearm_pronation", "pro_sup_l"),
    ("set_right_forearm_pronation", "pro_sup_r"),
    ("set_lumbar_extension", "lumbar_extension"),
    ("set_lumbar_bending", "lumbar_bending"),
    ("set_lumbar_rotation", "lumbar_rotation"),
]


def make_user(**kwargs):
    return User(dataset=DATASET, **kwargs)


# ---------------------------------------------------------------------------
# ANSUR data layer (no OpenSim binding required)
# ---------------------------------------------------------------------------


def test_loader_repairs_implicit_subject_id_and_gender():
    frame = load_ansur(DATASET)

    assert frame.columns[0] == "subject_id"
    assert len(frame) == 6068
    assert set(frame["Gender"]) == {"M", "F"}
    assert frame["stature_m"].dtype.kind == "f"


def test_default_percentile_is_used_for_gender_only():
    reference = resolve_reference("M", dataset=DATASET)

    assert reference.gender == "M"
    assert reference.percentile == 50.0
    assert reference.height_cm == pytest.approx(175.5, abs=0.1)


def test_explicit_percentile_applies_to_height_and_measurements():
    reference = resolve_reference("F", percentile=75.0, dataset=DATASET)
    female = load_ansur(DATASET).query("Gender == 'F'")

    assert reference.percentile == 75.0
    assert reference.height_m == pytest.approx(
        np.percentile(female["stature_m"], 75.0, method="linear")
    )
    assert reference.values["footlength"] == pytest.approx(
        np.percentile(female["footlength"], 75.0, method="linear")
    )


def test_height_resolves_a_common_target_percentile():
    reference = resolve_reference("M", height=175.0, dataset=DATASET)
    male = load_ansur(DATASET).query("Gender == 'M'")
    expected_percentile = (
        np.searchsorted(np.sort(male["stature_m"].to_numpy()), 1.75, side="right")
        * 100
        / len(male)
    )

    assert reference.percentile == pytest.approx(
        min(99.9, max(0.1, expected_percentile))
    )
    assert reference.height_cm == pytest.approx(
        np.percentile(male["stature_m"], reference.percentile, method="linear") * 100
    )


@pytest.mark.parametrize("gender", ["X", "male", ""])
def test_gender_is_validated(gender):
    with pytest.raises(ValueError, match="gender"):
        resolve_reference(gender, dataset=DATASET)


def test_height_outside_dataset_range_is_rejected():
    with pytest.raises(ValueError, match="outside ANSUR range"):
        resolve_reference("F", height=250.0, dataset=DATASET)


# ---------------------------------------------------------------------------
# Constructing a User (requires the OpenSim bindings)
# ---------------------------------------------------------------------------


@requires_opensim
def test_default_percentile_is_used_when_only_gender_is_given():
    male = make_user(gender="M")
    female = make_user(gender="F")

    assert male.gender == "M"
    assert male.percentile == 50.0
    assert female.gender == "F"
    assert female.percentile == 50.0


@requires_opensim
def test_explicit_percentile_is_honored():
    user = make_user(gender="M", percentile=75.0)

    assert user.percentile == 75.0


@requires_opensim
def test_explicit_height_resolves_its_empirical_percentile():
    user = make_user(gender="M", height=175.0)
    reference = resolve_reference("M", height=175.0, dataset=DATASET)

    assert user.percentile == pytest.approx(reference.percentile)
    assert user.height == pytest.approx(reference.height_cm)


@requires_opensim
def test_anthropometry_property_matches_the_resolved_reference():
    user = make_user(gender="F", percentile=75.0)

    assert user.anthropometry.gender == "F"
    assert user.anthropometry.percentile == 75.0
    assert user.anthropometry.height_cm == pytest.approx(user.height)


@requires_opensim
def test_gender_only_accepts_m_or_f():
    with pytest.raises(ValueError, match="gender"):
        make_user(gender="X")


@requires_opensim
def test_instances_are_backed_by_independent_opensim_models():
    user_50 = make_user(gender="M", percentile=50)
    user_75 = make_user(gender="M", percentile=75)

    assert user_50.model is not user_75.model


# ---------------------------------------------------------------------------
# Anthropometric scaling of the bundled OpenSim model
# ---------------------------------------------------------------------------


@requires_opensim
def test_model_components_match_the_bundled_rajagopal_model():
    user = make_user(gender="M", percentile=75.0)

    assert user.bodies.getSize() == 22
    assert user.joints.getSize() == 22
    assert user.muscles.getSize() == 80
    assert user.markers.getSize() == 66
    assert user.coordinates.getSize() == 39


@requires_opensim
def test_scaling_moves_markers_away_from_the_unscaled_baseline():
    baseline_user = make_user(gender="M", percentile=50.0)
    scaled_user = make_user(gender="M", percentile=95.0)

    baseline = baseline_user.marker_location("RTOE")
    scaled = scaled_user.marker_location("RTOE")

    assert scaled != pytest.approx(baseline)


@requires_opensim
def test_originally_locked_coordinates_are_unlocked_for_full_configurability():
    user = make_user(gender="F")

    assert user.coordinate_locked("subtalar_angle_l") is False
    assert user.coordinate_locked("mtp_angle_r") is False
    assert user.coordinate_locked("wrist_flex_r") is False


# ---------------------------------------------------------------------------
# Posture setters (exhaustive: every left/right/lumbar coordinate alias)
# ---------------------------------------------------------------------------


@requires_opensim
@pytest.mark.parametrize("setter_name, coordinate_name", POSTURE_SETTERS)
def test_every_posture_setter_writes_its_target_coordinate(setter_name, coordinate_name):
    user = make_user(gender="F")

    getattr(user, setter_name)(12.0)

    assert user.coordinate_degrees(coordinate_name) == pytest.approx(12.0)


@requires_opensim
def test_posture_setters_do_not_cross_talk_between_sides():
    user = make_user(gender="M")

    user.set_left_knee_flexionextension(10.0)
    user.set_right_hip_flexionextension(25.0)

    assert user.coordinate_degrees("knee_angle_l") == pytest.approx(10.0)
    assert user.coordinate_degrees("knee_angle_r") == pytest.approx(0.0)
    assert user.coordinate_degrees("hip_flexion_r") == pytest.approx(25.0)
    assert user.coordinate_degrees("hip_flexion_l") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Inherited OpenSimModel facade, exercised through User
# ---------------------------------------------------------------------------


@requires_opensim
def test_inherited_coordinate_locking_is_enforced():
    user = make_user(gender="M")

    user.set_coordinate_locked("knee_angle_r", True)
    with pytest.raises(ValueError, match="locked"):
        user.set_coordinate_degrees("knee_angle_r", 10.0)


@requires_opensim
def test_inherited_marker_and_muscle_accessors_work():
    user = make_user(gender="M")

    user.set_marker_location("RTOE", 0.1, 0.02, 0.03)
    user.set_muscle_max_isometric_force("addbrev_r", 1500.0)

    assert user.marker_location("RTOE") == pytest.approx((0.1, 0.02, 0.03))
    assert user.muscle_max_isometric_force("addbrev_r") == pytest.approx(1500.0)


@requires_opensim
def test_export_writes_a_reloadable_osim_file_with_current_posture():
    user = make_user(gender="M", percentile=75.0)
    user.set_right_hip_flexionextension(25.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = user.export(Path(tmp_dir) / "exported.osim")

        assert destination.is_file()
        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        hip_flexion_r = reloaded.getCoordinateSet().get("hip_flexion_r")
        assert hip_flexion_r.get_default_value() == pytest.approx(np.deg2rad(25.0))


# ---------------------------------------------------------------------------
# Rendering assets and composition with other OpenSimModel instances
# ---------------------------------------------------------------------------


@requires_opensim
def test_user_registers_its_own_mesh_directory_for_show():
    user = make_user(gender="M")

    assert DEFAULT_MESHES_DIR in user.geometry_directories
    assert user.visualizer is None


@requires_opensim
def test_adding_two_users_returns_a_generic_opensimmodel_not_a_user():
    combined = make_user(gender="M") + make_user(gender="F")

    assert type(combined) is OpenSimModel
    assert combined.bodies.getSize() == 44
