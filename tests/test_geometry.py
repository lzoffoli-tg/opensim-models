"""opensim_models.operators.euclidean_distance -- a pure geometry helper with
no OpenSim model involved at all, so (unlike every other test file in this
suite) this one does not need the opensim bindings installed to run."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import operators


def test_euclidean_distance_matches_a_known_3_4_5_triangle():
    assert operators.euclidean_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_euclidean_distance_is_zero_for_coincident_points():
    assert operators.euclidean_distance((1.0, -2.0, 3.0), (1.0, -2.0, 3.0)) == pytest.approx(0.0)


def test_euclidean_distance_is_symmetric():
    a = (1.0, 2.0, 3.0)
    b = (-4.0, 5.0, -6.0)
    assert operators.euclidean_distance(a, b) == pytest.approx(operators.euclidean_distance(b, a))


def test_euclidean_distance_accepts_lists_and_numpy_arrays():
    import numpy as np

    assert operators.euclidean_distance([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert operators.euclidean_distance(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    ) == pytest.approx(2.0)


def test_euclidean_distance_returns_a_plain_float():
    result = operators.euclidean_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "position1,position2",
    [
        ((0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0)),
    ],
)
def test_euclidean_distance_rejects_wrong_sized_positions(position1, position2):
    with pytest.raises(ValueError, match="exactly 3 values"):
        operators.euclidean_distance(position1, position2)


def test_euclidean_distance_rejects_non_finite_position1():
    with pytest.raises(ValueError, match="finite"):
        operators.euclidean_distance((float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0))


def test_euclidean_distance_rejects_non_finite_position2():
    with pytest.raises(ValueError, match="finite"):
        operators.euclidean_distance((0.0, 0.0, 0.0), (0.0, float("nan"), 0.0))
