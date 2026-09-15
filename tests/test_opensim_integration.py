import tempfile
from pathlib import Path

import numpy as np
import pytest

from opensim_user import OpensimUser, OpensimViewer

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "assets" / "ansur_ref.csv"
MODEL = ROOT / "assets" / "rajagopalaiulrich2023.osim"


opensim = pytest.importorskip("opensim")


def make_user(**kwargs):
    return OpensimUser(dataset=DATASET, model_path=MODEL, **kwargs)


def test_model_components_and_scaled_marker():
    user = make_user(gender="M", percentile=75.0)

    assert user.bodies.getSize() == 22
    assert user.joints.getSize() == 22
    assert user.muscles.getSize() == 80
    assert user.markers.getSize() == 66
    assert user.marker("RTOE").get_location().get(0) != pytest.approx(0.205)


def test_instances_are_independent_and_posture_uses_degrees():
    first = make_user(gender="M")
    second = make_user(gender="M")

    first.set_right_hip_flexionextension(25.0)
    first.set_left_knee_flexionextension(10.0)

    assert first.model is not second.model
    assert first.coordinate_degrees("hip_flexion_r") == pytest.approx(25.0)
    assert second.coordinate_degrees("hip_flexion_r") == pytest.approx(0.0)
    assert first.coordinate_degrees("knee_angle_l") == pytest.approx(10.0)


def test_all_main_posture_aliases_write_coordinates():
    user = make_user(gender="F")
    setters = [
        (user.set_right_hip_flexionextension, "hip_flexion_r"),
        (user.set_left_hip_adduction, "hip_adduction_l"),
        (user.set_right_ankle_flexiondorsiflexion, "ankle_angle_r"),
        (user.set_left_forearm_pronation, "pro_sup_l"),
    ]
    for setter, coordinate in setters:
        setter(12.0)
        assert user.coordinate_degrees(coordinate) == pytest.approx(12.0)


def test_originally_locked_coordinates_are_unlocked_for_full_configurability():
    user = make_user(gender="F")

    assert user.coordinate_locked("subtalar_angle_l") is False
    assert user.coordinate_locked("mtp_angle_r") is False
    assert user.coordinate_locked("wrist_flex_r") is False

    user.set_left_subtalar_inversion(12.0)
    user.set_right_mtp_flexion(12.0)
    user.set_right_wrist_flexion(12.0)

    assert user.coordinate_degrees("subtalar_angle_l") == pytest.approx(12.0)
    assert user.coordinate_degrees("mtp_angle_r") == pytest.approx(12.0)
    assert user.coordinate_degrees("wrist_flex_r") == pytest.approx(12.0)


def test_coordinate_locking_can_be_toggled_and_is_enforced():
    user = make_user(gender="F")

    user.set_coordinate_locked("knee_angle_r", True)
    assert user.coordinate_locked("knee_angle_r") is True
    with pytest.raises(ValueError, match="locked"):
        user.set_coordinate_degrees("knee_angle_r", 10.0)

    user.set_coordinate_locked("knee_angle_r", False)
    user.set_coordinate_degrees("knee_angle_r", 10.0)
    assert user.coordinate_degrees("knee_angle_r") == pytest.approx(10.0)


def test_coordinate_range_can_be_read_and_updated():
    user = make_user(gender="M")

    user.set_coordinate_range("knee_angle_r", -100.0, 5.0)

    assert user.coordinate_range("knee_angle_r") == pytest.approx((-100.0, 5.0))
    with pytest.raises(ValueError, match="min_degrees"):
        user.set_coordinate_range("knee_angle_r", 5.0, -100.0)


def test_marker_location_can_be_read_and_updated():
    user = make_user(gender="M")

    user.set_marker_location("RTOE", 0.1, 0.02, 0.03)

    assert user.marker_location("RTOE") == pytest.approx((0.1, 0.02, 0.03))


def test_muscle_mechanical_parameters_can_be_read_and_updated():
    user = make_user(gender="M")

    user.set_muscle_max_isometric_force("addbrev_r", 1500.0)
    user.set_muscle_optimal_fiber_length("addbrev_r", 0.15)
    user.set_muscle_tendon_slack_length("addbrev_r", 0.1)
    user.set_muscle_pennation_angle("addbrev_r", 10.0)

    assert user.muscle_max_isometric_force("addbrev_r") == pytest.approx(1500.0)
    assert user.muscle_optimal_fiber_length("addbrev_r") == pytest.approx(0.15)
    assert user.muscle_tendon_slack_length("addbrev_r") == pytest.approx(0.1)
    assert user.muscle_pennation_angle("addbrev_r") == pytest.approx(10.0)


def test_export_writes_a_reloadable_osim_file_with_current_posture():
    user = make_user(gender="M", percentile=75.0)
    user.set_right_hip_flexionextension(25.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = user.export(Path(tmp_dir) / "exported.osim")

        assert destination.is_file()
        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        coordinates = reloaded.getCoordinateSet()
        hip_flexion_r = coordinates.get("hip_flexion_r")
        assert hip_flexion_r.get_default_value() == pytest.approx(np.deg2rad(25.0))


def test_viewer_accepts_user_without_opening_window():
    user = make_user(gender="M")
    viewer = OpensimViewer(user, geometry_path=ROOT / "assets" / "meshes")

    assert viewer.model is user.model
    assert viewer.state is user.state
    assert viewer.visualizer is None
