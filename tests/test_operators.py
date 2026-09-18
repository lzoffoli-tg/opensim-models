import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel, operators

opensim = pytest.importorskip("opensim")


def make_model():
    return OpenSimModel(model_path=None)


def add_free_body(model, name):
    """Add a body with a FreeJoint to ground, as a single structural change."""
    with model.structural_change():
        body = operators.add_body(model, name, mass=2.0, inertia=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        operators.add_joint(
            model,
            opensim.FreeJoint(f"{name}_to_ground", model.model.getGround(), body),
        )
    return body


# ---------------------------------------------------------------------------
# Generic add_component/remove_component
# ---------------------------------------------------------------------------


def test_add_component_rejects_unknown_kind():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0, 0, 0))

    with pytest.raises(ValueError, match="Unknown component kind"):
        operators.add_component(model, "not_a_kind", marker)


def test_remove_component_rejects_unknown_name():
    model = make_model()

    with pytest.raises(ValueError, match="No marker named"):
        operators.remove_component(model, "marker", "does_not_exist")


# ---------------------------------------------------------------------------
# Bodies and joints
# ---------------------------------------------------------------------------


def test_add_body_and_joint_as_a_batch():
    model = make_model()

    body = add_free_body(model, "b1")

    assert model.bodies.getSize() == 1
    assert model.joints.getSize() == 1
    assert model.body("b1").getMass() == pytest.approx(2.0)
    assert body.getName() == "b1"


def test_remove_body_and_joint_as_a_batch():
    model = make_model()
    add_free_body(model, "b1")

    with model.structural_change():
        operators.remove_joint(model, "b1_to_ground")
        operators.remove_body(model, "b1")

    assert model.bodies.getSize() == 0
    assert model.joints.getSize() == 0


def test_structural_change_preserves_posture_of_surviving_coordinates():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    coordinate_name = next(
        model.coordinates.get(i).getName()
        for i in range(model.coordinates.getSize())
        if model.coordinates.get(i).getName().startswith("b1_to_ground")
    )
    model.set_coordinate_degrees(coordinate_name, 30.0)

    with model.structural_change():
        operators.remove_joint(model, "b2_to_ground")
        operators.remove_body(model, "b2")

    assert model.coordinate_degrees(coordinate_name) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Forces and muscles
# ---------------------------------------------------------------------------


def _make_muscle(model, body, name="mus1"):
    muscle = opensim.Millard2012EquilibriumMuscle(name, 500.0, 0.1, 0.2, 0.0)
    muscle.addNewPathPoint("p1", model.model.getGround(), opensim.Vec3(0, 0, 0))
    muscle.addNewPathPoint("p2", body, opensim.Vec3(0, 0, 0))
    return muscle


def test_add_muscle_and_remove_muscle():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(model, _make_muscle(model, body), reinitialize=True)
    assert model.muscles.getSize() == 1

    operators.remove_muscle(model, "mus1", reinitialize=True)
    assert model.muscles.getSize() == 0


def test_add_muscle_is_visible_through_the_force_set():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(model, _make_muscle(model, body), reinitialize=True)

    assert model.model.getForceSet().getIndex("mus1") >= 0


# ---------------------------------------------------------------------------
# Markers, constraints
# ---------------------------------------------------------------------------


def test_add_marker_and_remove_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0, 0, 0))

    operators.add_marker(model, marker, reinitialize=True)
    assert model.markers.getSize() == 1

    operators.remove_marker(model, "mk1", reinitialize=True)
    assert model.markers.getSize() == 0


def test_add_constraint_and_remove_constraint():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    independent = next(
        model.coordinates.get(i).getName()
        for i in range(model.coordinates.getSize())
        if model.coordinates.get(i).getName().startswith("b1_to_ground")
    )
    dependent = next(
        model.coordinates.get(i).getName()
        for i in range(model.coordinates.getSize())
        if model.coordinates.get(i).getName().startswith("b2_to_ground")
    )

    coupler = opensim.CoordinateCouplerConstraint()
    coupler.setName("coupler1")
    coupler.setIndependentCoordinateNames(opensim.ArrayStr(independent, 1))
    coupler.setDependentCoordinateName(dependent)
    coupler.setFunction(opensim.LinearFunction(1.0, 0.0))

    operators.add_constraint(model, coupler, reinitialize=True)
    assert model.model.getConstraintSet().getSize() == 1

    operators.remove_constraint(model, "coupler1", reinitialize=True)
    assert model.model.getConstraintSet().getSize() == 0


# ---------------------------------------------------------------------------
# add_body mesh attachment
# ---------------------------------------------------------------------------


def test_add_body_attaches_an_existing_mesh_file():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_path = Path(tmp_dir) / "part.stl"
        from opensim_models._primitives import write_box_mesh

        write_box_mesh(mesh_path, 0.1, 0.1, 0.1)

        with model.structural_change():
            body = operators.add_body(model, "b1", mass=1.0, mesh_files=mesh_path)
            operators.add_weld_joint(model, "b1_joint", body)

        assert body.getPropertyByName("attached_geometry").size() == 1
        assert Path(tmp_dir) in model.geometry_directories


# ---------------------------------------------------------------------------
# Named joint constructors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adder, expected_dof",
    [
        (operators.add_free_joint, 6),
        (operators.add_pin_joint, 1),
        (operators.add_ball_joint, 3),
        (operators.add_slider_joint, 1),
        (operators.add_weld_joint, 0),
    ],
)
def test_named_joint_constructors_create_the_expected_number_of_coordinates(
    adder, expected_dof
):
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        adder(model, "b1_joint", body)

    assert model.coordinates.getSize() == expected_dof


def test_named_joint_constructor_places_the_joint_at_position_and_orientation():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_pin_joint(
            model,
            "b1_joint",
            body,
            position=(1.0, 2.0, 3.0),
            orientation_deg=(90.0, 0.0, 0.0),
        )

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx((1.0, 2.0, 3.0))
    assert parent.get_orientation()[0] == pytest.approx(math.radians(90.0))


def test_add_joint_rejects_unknown_kind_when_called_generically():
    model = make_model()
    body = operators.add_body(model, "b1", mass=1.0)

    with pytest.raises(ValueError, match="Unknown component kind"):
        operators.add_component(model, "not_a_joint_kind", body)


# ---------------------------------------------------------------------------
# Primitive-shaped bodies
# ---------------------------------------------------------------------------


def test_add_box_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, joint = operators.add_box_body(
        model, "box1", (0.2, 0.3, 0.4), density=1200.0, reinitialize=True
    )

    expected_mass = 0.2 * 0.3 * 0.4 * 1200.0
    assert body.getMass() == pytest.approx(expected_mass)
    moments = body.getInertia().getMoments()
    assert moments.get(0) == pytest.approx(expected_mass / 12.0 * (0.3**2 + 0.4**2))
    assert moments.get(1) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.4**2))
    assert moments.get(2) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.3**2))
    assert opensim.WeldJoint.safeDownCast(joint) is not None


def test_add_cylinder_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_cylinder_body(model, "cyl1", 0.05, 0.3, reinitialize=True)

    expected_mass = math.pi * 0.05**2 * 0.3 * 1000.0
    assert body.getMass() == pytest.approx(expected_mass)
    moments = body.getInertia().getMoments()
    assert moments.get(1) == pytest.approx(expected_mass * 0.05**2 / 2.0)


def test_add_sphere_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_sphere_body(model, "sph1", 0.1, reinitialize=True)

    expected_mass = 4.0 / 3.0 * math.pi * 0.1**3 * 1000.0
    assert body.getMass() == pytest.approx(expected_mass)
    moments = body.getInertia().getMoments()
    expected_moment = 2.0 / 5.0 * expected_mass * 0.1**2
    assert moments.get(0) == pytest.approx(expected_moment)
    assert moments.get(1) == pytest.approx(expected_moment)
    assert moments.get(2) == pytest.approx(expected_moment)


def test_primitive_body_is_placed_at_the_given_position_and_orientation():
    model = make_model()

    _, joint = operators.add_box_body(
        model,
        "box1",
        (0.1, 0.1, 0.1),
        position=(1.0, 2.0, 3.0),
        orientation_deg=(0.0, 45.0, 0.0),
        reinitialize=True,
    )

    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx((1.0, 2.0, 3.0))
    assert parent.get_orientation()[1] == pytest.approx(math.radians(45.0))


def test_primitive_body_default_joint_type_is_weld():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, reinitialize=True)

    assert model.coordinates.getSize() == 0


def test_primitive_body_joint_type_can_be_changed():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, joint_type="free", reinitialize=True)

    assert model.coordinates.getSize() == 6


def test_primitive_body_rejects_unknown_joint_type():
    model = make_model()

    with pytest.raises(ValueError, match="Unknown joint_type"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), joint_type="bogus")

    assert model.bodies.getSize() == 0


def test_primitive_body_mesh_requires_mesh_dir():
    model = make_model()

    with pytest.raises(ValueError, match="mesh_dir"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), mesh=True)

    assert model.bodies.getSize() == 0


def test_primitive_body_can_generate_an_actual_mesh_file():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        body, _ = operators.add_box_body(
            model, "box1", (0.1, 0.2, 0.3), mesh=True, mesh_dir=tmp_dir, reinitialize=True
        )

        assert (Path(tmp_dir) / "box1.stl").is_file()
        assert body.getPropertyByName("attached_geometry").size() == 1


def test_primitive_bodies_can_be_batched_together():
    model = make_model()

    with model.structural_change():
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), position=(0.0, 0.0, 0.0))
        operators.add_sphere_body(model, "sph1", 0.05, position=(1.0, 0.0, 0.0))

    assert model.bodies.getSize() == 2
    assert model.joints.getSize() == 2
