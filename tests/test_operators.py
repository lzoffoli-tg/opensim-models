import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Box, OpenSimModel, operators

opensim = pytest.importorskip("opensim")


def make_model():
    return OpenSimModel(model_path=None)


def add_free_body(model, name):
    """Add a body with a FreeJoint to ground, as a single structural change."""
    with model.structural_change():
        body = operators.add_body(model, name, mass=2.0, inertia=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        operators.add_joint(
            model,
            opensim.FreeJoint(f"{name}_to_ground", model.model.getGround(), body.raw),
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

    assert len(model.bodies) == 1
    assert len(model.joints) == 1
    assert model.body("b1").mass == pytest.approx(2.0)
    assert body.name == "b1"


def test_remove_body_and_joint_as_a_batch():
    model = make_model()
    add_free_body(model, "b1")

    with model.structural_change():
        operators.remove_joint(model, "b1_to_ground")
        operators.remove_body(model, "b1")

    assert len(model.bodies) == 0
    assert len(model.joints) == 0


def test_structural_change_preserves_posture_of_surviving_coordinates():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    coordinate_name = next(
        name for name in model.coordinates if name.startswith("b1_to_ground")
    )
    model.coordinate(coordinate_name).set_value_degrees(30.0)

    with model.structural_change():
        operators.remove_joint(model, "b2_to_ground")
        operators.remove_body(model, "b2")

    assert model.coordinate(coordinate_name).value_degrees == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# attach_component
# ---------------------------------------------------------------------------


def test_attach_component_reattaches_to_a_new_parent_at_com():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    joint = operators.attach_component(
        model, child.raw, to=parent.raw, reinitialize=True,
    )

    assert joint.name == "child_to_parent"
    assert len(model.joints) == 2
    assert "child_to_ground" not in model.joints
    assert "parent" in joint.raw.getParentFrame().getName()


def test_attach_component_tilts_a_slider_axis_with_parent_orientation_deg():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    joint = operators.attach_component(
        model, child.raw, to=parent.raw,
        child_point=(0.0, 0.0, 0.0), parent_point=(0.0, 0.0, 0.0),
        parent_orientation_deg=(0.0, 0.0, 45.0),
        joint_type="slider", reinitialize=True,
    )

    coordinate = next(iter(joint.coordinates.values()))
    coordinate.set_value(1.0, enforce_constraints=False)
    model.update_state()
    model.model.realizePosition(model.state)

    position = child.raw.getPositionInGround(model.state)
    assert (position.get(0), position.get(1)) == pytest.approx(
        (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))
    )


def test_attach_component_resolves_explicit_points():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    operators.attach_component(
        model, child.raw, to=parent.raw,
        child_point=(0.0, 0.0, 0.0), parent_point=(0.0, 0.5, 0.0),
        reinitialize=True,
    )

    model.model.realizePosition(model.state)
    position = child.raw.getPositionInGround(model.state)
    assert (position.get(0), position.get(1), position.get(2)) == pytest.approx((0.0, 0.5, 0.0))


def test_attach_component_supports_a_standalone_component_once_merged():
    model = make_model()
    parent = add_free_body(model, "parent")
    box = Box(0.1, 0.1, 0.1, origin=(9.0, 9.0, 9.0))

    merged = model + box
    merged_parent = merged.body("parent")
    merged_box = merged.body("box")

    merged.attach_component(merged_box, to=merged_parent, reinitialize=True)

    merged.model.realizePosition(merged.state)
    box_position = merged_box.raw.getPositionInGround(merged.state)
    parent_position = merged_parent.raw.getPositionInGround(merged.state)
    assert (box_position.get(0), box_position.get(1), box_position.get(2)) == pytest.approx(
        (parent_position.get(0), parent_position.get(1), parent_position.get(2))
    )


def test_attach_component_rejects_a_component_not_yet_in_the_model():
    model = make_model()
    parent = add_free_body(model, "parent")
    lone_box = Box(0.1, 0.1, 0.1)

    with pytest.raises(ValueError, match="not connected by any joint"):
        operators.attach_component(model, lone_box.raw, to=parent.raw)


def test_attach_component_rejects_an_unknown_joint_type():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    with pytest.raises(ValueError, match="Unknown joint_type"):
        operators.attach_component(model, child.raw, to=parent.raw, joint_type="bogus")


# ---------------------------------------------------------------------------
# Forces and muscles
# ---------------------------------------------------------------------------


def test_add_muscle_and_remove_muscle():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, reinitialize=True,
    )
    assert len(model.muscles) == 1

    operators.remove_muscle(model, "mus1", reinitialize=True)
    assert len(model.muscles) == 0


def test_add_muscle_is_visible_through_the_force_set():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, reinitialize=True,
    )

    assert model.model.getForceSet().getIndex("mus1") >= 0


def test_add_muscle_with_via_points_creates_a_longer_path():
    model = make_model()
    body = add_free_body(model, "b1")

    muscle = operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, via_points=[(model.model.getGround(), (0.05, 0, 0))],
        reinitialize=True,
    )

    assert muscle.raw.getGeometryPath().getCurrentPath(model.state).getSize() == 3


# ---------------------------------------------------------------------------
# Markers, constraints
# ---------------------------------------------------------------------------


def test_add_marker_and_remove_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0, 0, 0))

    operators.add_marker(model, marker, reinitialize=True)
    assert len(model.markers) == 1

    operators.remove_marker(model, "mk1", reinitialize=True)
    assert len(model.markers) == 0


def test_add_coordinate_coupler_constraint_and_remove_constraint():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    independent = next(
        name for name in model.coordinates if name.startswith("b1_to_ground")
    )
    dependent = next(
        name for name in model.coordinates if name.startswith("b2_to_ground")
    )

    operators.add_coordinate_coupler_constraint(
        model, "coupler1", independent, dependent, opensim.LinearFunction(1.0, 0.0),
        reinitialize=True,
    )
    assert model.model.getConstraintSet().getSize() == 1

    operators.remove_constraint(model, "coupler1", reinitialize=True)
    assert model.model.getConstraintSet().getSize() == 0


def test_add_weld_constraint_removes_relative_motion():
    model = make_model()
    body1 = add_free_body(model, "b1")
    body2 = add_free_body(model, "b2")

    operators.add_weld_constraint(model, "weld1", body1.raw, body2.raw, reinitialize=True)

    assert model.model.getConstraintSet().getSize() == 1
    assert model.model.getConstraintSet().get("weld1") is not None


def test_add_point_constraint_connects_a_body_to_ground():
    # Confirmed directly (OpenSim 4.6): a PointConstraint between two
    # non-ground bodies crashes the process natively in initSystem() --
    # see add_point_constraint's docstring. Ground-to-body is the
    # confirmed-safe case, so that's what this exercises.
    model = make_model()
    body = add_free_body(model, "b1")

    constraint = operators.add_point_constraint(
        model, "point1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        reinitialize=True,
    )

    assert constraint.name == "point1"
    assert model.model.getConstraintSet().getSize() == 1


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

        assert body.raw.getPropertyByName("attached_geometry").size() == 1
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
        joint = adder(model, "b1_joint", body)

    assert len(joint.coordinates) == expected_dof


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
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())
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
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
    assert moments.get(0) == pytest.approx(expected_mass / 12.0 * (0.3**2 + 0.4**2))
    assert moments.get(1) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.4**2))
    assert moments.get(2) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.3**2))
    assert opensim.WeldJoint.safeDownCast(joint.raw) is not None


def test_add_cylinder_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_cylinder_body(model, "cyl1", 0.05, 0.3, reinitialize=True)

    expected_mass = math.pi * 0.05**2 * 0.3 * 1000.0
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
    assert moments.get(1) == pytest.approx(expected_mass * 0.05**2 / 2.0)


def test_add_sphere_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_sphere_body(model, "sph1", 0.1, reinitialize=True)

    expected_mass = 4.0 / 3.0 * math.pi * 0.1**3 * 1000.0
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
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

    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx((1.0, 2.0, 3.0))
    assert parent.get_orientation()[1] == pytest.approx(math.radians(45.0))


def test_primitive_body_default_joint_type_is_weld():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, reinitialize=True)

    assert len(model.coordinates) == 0


def test_primitive_body_joint_type_can_be_changed():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, joint_type="free", reinitialize=True)

    assert len(model.coordinates) == 6


def test_primitive_body_rejects_unknown_joint_type():
    model = make_model()

    with pytest.raises(ValueError, match="Unknown joint_type"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), joint_type="bogus")

    assert len(model.bodies) == 0


def test_primitive_body_mesh_requires_mesh_dir():
    model = make_model()

    with pytest.raises(ValueError, match="mesh_dir"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), mesh=True)

    assert len(model.bodies) == 0


def test_primitive_body_can_generate_an_actual_mesh_file():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        body, _ = operators.add_box_body(
            model, "box1", (0.1, 0.2, 0.3), mesh=True, mesh_dir=tmp_dir, reinitialize=True
        )

        assert (Path(tmp_dir) / "box1.stl").is_file()
        assert body.raw.getPropertyByName("attached_geometry").size() == 1


def test_primitive_bodies_can_be_batched_together():
    model = make_model()

    with model.structural_change():
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), position=(0.0, 0.0, 0.0))
        operators.add_sphere_body(model, "sph1", 0.05, position=(1.0, 0.0, 0.0))

    assert len(model.bodies) == 2
    assert len(model.joints) == 2


# ---------------------------------------------------------------------------
# rotate_object
# ---------------------------------------------------------------------------


def test_rotate_object_rotates_a_marker_about_the_origin():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    new_position = operators.rotate_object(marker, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert model.marker("mk1").location == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_rotates_a_physical_offset_frame_translation_and_orientation():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.rotate_object(parent, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )
    assert parent.get_orientation()[2] == pytest.approx(math.radians(90.0))


def test_rotate_object_pivots_about_an_external_point():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.rotate_object(parent, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0), 180.0)

    assert new_position == pytest.approx((3.0, 0.0, 0.0), abs=1e-9)


def test_rotate_object_origin_can_be_another_component():
    model = make_model()
    pivot_marker = opensim.Marker(
        "pivot", model.model.getGround(), opensim.Vec3(2.0, 0.0, 0.0)
    )
    operators.add_marker(model, pivot_marker, reinitialize=True)
    moving_marker = opensim.Marker(
        "moving", model.model.getGround(), opensim.Vec3(3.0, 0.0, 0.0)
    )
    operators.add_marker(model, moving_marker, reinitialize=True)

    new_position = operators.rotate_object(
        moving_marker, pivot_marker, (0.0, 0.0, 1.0), 180.0
    )

    assert new_position == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_rotate_object_on_a_body_is_read_only():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.rotate_object(body, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


def test_rotate_object_rejects_unresolvable_object():
    with pytest.raises(TypeError, match="Cannot resolve a ground-frame position"):
        operators.rotate_object(
            "not an opensim component", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0
        )


def test_rotate_object_rejects_zero_direction():
    with pytest.raises(ValueError, match="direction must be a non-zero vector"):
        operators.rotate_object((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 90.0)


def test_rotate_object_works_standalone_on_plain_coordinates():
    new_position = operators.rotate_object((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_rotates_the_whole_model_about_its_root_joint():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    new_position = operators.rotate_object(model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_rejects_a_model_with_no_ground_joint():
    model = make_model()

    with pytest.raises(ValueError, match="no joint attached to ground"):
        operators.rotate_object(model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)


# ---------------------------------------------------------------------------
# rotate_object(..., inplace=False)
# ---------------------------------------------------------------------------


def test_rotate_object_not_inplace_leaves_the_whole_model_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    rotated_copy = operators.rotate_object(
        model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = rotated_copy.body("b1").raw.getPositionInGround(rotated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_not_inplace_leaves_a_marker_untouched():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    rotated_copy = operators.rotate_object(
        marker, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not marker
    assert tuple(marker.get_location().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(rotated_copy.get_location().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_not_inplace_leaves_a_physical_offset_frame_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    rotated_copy = operators.rotate_object(
        parent, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not parent
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(rotated_copy.get_translation().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_not_inplace_on_a_body_still_returns_a_position():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.rotate_object(
        body, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


# ---------------------------------------------------------------------------
# translate_object
# ---------------------------------------------------------------------------


def test_translate_object_translates_a_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    new_position = operators.translate_object(marker, (0.0, 2.0, 3.0))

    assert new_position == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)
    assert model.marker("mk1").location == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)


def test_translate_object_translates_a_physical_offset_frame():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.translate_object(parent, (0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )
    # a translation does not rotate anything
    orientation = parent.get_orientation()
    assert (orientation[0], orientation[1], orientation[2]) == pytest.approx(
        (0.0, 0.0, 0.0)
    )


def test_translate_object_on_a_body_is_read_only():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.translate_object(body, (1.0, 0.0, 0.0))

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


def test_translate_object_rejects_unresolvable_object():
    with pytest.raises(TypeError, match="Cannot resolve a ground-frame position"):
        operators.translate_object("not an opensim component", (0.0, 0.0, 0.0))


def test_translate_object_rejects_malformed_direction():
    with pytest.raises(ValueError, match="direction must have exactly 3 values"):
        operators.translate_object((1.0, 0.0, 0.0), (0.0, 0.0))


def test_translate_object_works_standalone_on_plain_coordinates():
    new_position = operators.translate_object((1.0, 0.0, 0.0), (0.0, 2.0, 3.0))

    assert new_position == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)


def test_translate_object_translates_the_whole_model_about_its_root_joint():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    new_position = operators.translate_object(model, (0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_translate_object_rejects_a_model_with_no_ground_joint():
    model = make_model()

    with pytest.raises(ValueError, match="no joint attached to ground"):
        operators.translate_object(model, (0.0, 2.0, 0.0))


# ---------------------------------------------------------------------------
# translate_object(..., inplace=False)
# ---------------------------------------------------------------------------


def test_translate_object_not_inplace_leaves_the_whole_model_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    translated_copy = operators.translate_object(model, (0.0, 2.0, 0.0), inplace=False)

    assert translated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = translated_copy.body("b1").raw.getPositionInGround(translated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)


def test_translate_object_not_inplace_leaves_a_marker_untouched():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    translated_copy = operators.translate_object(marker, (0.0, 2.0, 3.0), inplace=False)

    assert translated_copy is not marker
    assert tuple(marker.get_location().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(translated_copy.get_location().to_numpy()) == pytest.approx(
        (1.0, 2.0, 3.0), abs=1e-9
    )


def test_translate_object_not_inplace_leaves_a_physical_offset_frame_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    translated_copy = operators.translate_object(parent, (0.0, 2.0, 0.0), inplace=False)

    assert translated_copy is not parent
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(translated_copy.get_translation().to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_translate_object_not_inplace_on_a_body_still_returns_a_position():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.translate_object(body, (1.0, 0.0, 0.0), inplace=False)

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )
