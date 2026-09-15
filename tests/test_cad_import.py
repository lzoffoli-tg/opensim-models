import tempfile
from pathlib import Path

import pytest

from opensim_models import OpenSimModel

pytest.importorskip("opensim")
OCC = pytest.importorskip("OCC")


def _make_box_step(directory: Path, name: str, dx: float, dy: float, dz: float) -> Path:
    """Write a single-solid STEP file, in millimetres, for a dx x dy x dz metre box.

    Declaring the file in millimetres (the typical unit for mechanical CAD
    exports) exercises OpenSimModel.from_step's unit conversion to metres.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    # Interface_Static only registers STEP parameters once a reader/writer
    # exists, so SetCVal must run after STEPControl_Writer() is constructed.
    Interface_Static.SetCVal("write.step.unit", "MM")
    box = BRepPrimAPI_MakeBox(dx * 1000.0, dy * 1000.0, dz * 1000.0).Shape()

    writer.Transfer(box, STEPControl_AsIs)
    step_path = directory / f"{name}.step"
    writer.Write(str(step_path))
    return step_path


def test_from_step_builds_one_body_with_analytic_mass_and_inertia():
    dx, dy, dz = 0.2, 0.3, 0.4
    density = 1200.0
    volume = dx * dy * dz
    expected_mass = volume * density
    # Analytic central inertia of a solid cuboid: I_xx = m/12 * (dy^2 + dz^2), etc.
    expected_ixx = expected_mass / 12.0 * (dy**2 + dz**2)
    expected_iyy = expected_mass / 12.0 * (dx**2 + dz**2)
    expected_izz = expected_mass / 12.0 * (dx**2 + dy**2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_box_step(tmp_path, "box", dx, dy, dz)

        model = OpenSimModel.from_step(step_path, density=density)

        assert model.bodies.getSize() == 1
        body = model.bodies.get(0)
        assert body.getMass() == pytest.approx(expected_mass, rel=1e-3)

        inertia = body.getInertia().getMoments()
        assert inertia.get(0) == pytest.approx(expected_ixx, rel=1e-2)
        assert inertia.get(1) == pytest.approx(expected_iyy, rel=1e-2)
        assert inertia.get(2) == pytest.approx(expected_izz, rel=1e-2)

        assert model.joints.getSize() == 1
        mesh_files = list(model.geometry_directories[0].glob("*.stl"))
        assert len(mesh_files) == 1


def test_from_step_without_free_joints_adds_unconnected_bodies():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_box_step(tmp_path, "box", 0.1, 0.1, 0.1)

        model = OpenSimModel.from_step(step_path, add_free_joints=False)

        assert model.bodies.getSize() == 1
        assert model.joints.getSize() == 0


def test_from_step_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenSimModel.from_step("does/not/exist.step")


def test_from_step_rejects_step_file_without_solids():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = tmp_path / "empty.step"
        step_path.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        )

        with pytest.raises(ValueError):
            OpenSimModel.from_step(step_path)
