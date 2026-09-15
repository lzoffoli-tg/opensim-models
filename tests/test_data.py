from pathlib import Path

import numpy as np
import pytest

from opensim_user.data import load_ansur, resolve_reference

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "assets" / "ansur_ref.csv"


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
