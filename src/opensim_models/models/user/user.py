from __future__ import annotations

import math
from pathlib import Path

from ...model import OpenSimModel, _register_geometry_search_path, import_opensim
from ._data import DEFAULT_DATASET, resolve_reference
from ._mapping import segment_scale_factors

__all__ = ["User"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL_PATH = _ASSETS_DIR / "rajagopalaiulrich2023.osim"
DEFAULT_MESHES_DIR = _ASSETS_DIR / "meshes"

# Friendly name -> OpenSim joint name, for every joint in the model. Each
# joint's child frame sits at the anatomical joint centre by OpenSim/Rajagopal
# convention.
_JOINT_CENTER_NAMES = {
    "pelvis": "ground_pelvis",
    "left_hip": "hip_l",
    "right_hip": "hip_r",
    "left_knee": "walker_knee_l",
    "right_knee": "walker_knee_r",
    "left_patella": "patellofemoral_l",
    "right_patella": "patellofemoral_r",
    "left_ankle": "ankle_l",
    "right_ankle": "ankle_r",
    "left_subtalar": "subtalar_l",
    "right_subtalar": "subtalar_r",
    "left_mtp": "mtp_l",
    "right_mtp": "mtp_r",
    "torso": "back",
    "left_shoulder": "acromial_l",
    "right_shoulder": "acromial_r",
    "left_elbow": "elbow_l",
    "right_elbow": "elbow_r",
    "left_radioulnar": "radioulnar_l",
    "right_radioulnar": "radioulnar_r",
    "left_wrist": "radius_hand_l",
    "right_wrist": "radius_hand_r",
}

_PELVIS_TRANSLATION_COORDINATES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")

# Friendly name -> OpenSim marker name. The foot has no 1st-metatarsal
# marker in this model (only the 5th): no landmark is exposed for it rather
# than guessing one.
_FOOT_MARKER_NAMES = {
    "left_heel": "LCAL",
    "right_heel": "RCAL",
    "left_toe": "LTOE",
    "right_toe": "RTOE",
    "left_mt5": "LMT5",
    "right_mt5": "RMT5",
}

# Friendly name -> OpenSim coordinate name, mirroring every set_* posture
# setter below (same name, minus the "set_" prefix) as a read-only property.
_POSTURE_COORDINATE_NAMES = {
    "left_hip_flexionextension": "hip_flexion_l",
    "right_hip_flexionextension": "hip_flexion_r",
    "left_hip_adduction": "hip_adduction_l",
    "right_hip_adduction": "hip_adduction_r",
    "left_hip_rotation": "hip_rotation_l",
    "right_hip_rotation": "hip_rotation_r",
    "left_knee_flexionextension": "knee_angle_l",
    "right_knee_flexionextension": "knee_angle_r",
    "left_ankle_flexiondorsiflexion": "ankle_angle_l",
    "right_ankle_flexiondorsiflexion": "ankle_angle_r",
    "left_subtalar_inversion": "subtalar_angle_l",
    "right_subtalar_inversion": "subtalar_angle_r",
    "left_mtp_flexion": "mtp_angle_l",
    "right_mtp_flexion": "mtp_angle_r",
    "left_shoulder_flexion": "arm_flex_l",
    "right_shoulder_flexion": "arm_flex_r",
    "left_shoulder_adduction": "arm_add_l",
    "right_shoulder_adduction": "arm_add_r",
    "left_shoulder_rotation": "arm_rot_l",
    "right_shoulder_rotation": "arm_rot_r",
    "left_elbow_flexion": "elbow_flex_l",
    "right_elbow_flexion": "elbow_flex_r",
    "left_wrist_flexion": "wrist_flex_l",
    "right_wrist_flexion": "wrist_flex_r",
    "left_wrist_deviation": "wrist_dev_l",
    "right_wrist_deviation": "wrist_dev_r",
    "left_forearm_pronation": "pro_sup_l",
    "right_forearm_pronation": "pro_sup_r",
    "lumbar_extension": "lumbar_extension",
    "lumbar_bending": "lumbar_bending",
    "lumbar_rotation": "lumbar_rotation",
}


class User(OpenSimModel):
    """An ANSUR-based anthropometric user backed by an OpenSim model.

    Parameters
    ----------
    gender : str
        Sex code, either ``"M"`` or ``"F"``.
    height : float or None, optional
        Requested stature in centimetres. If provided, every measurement is
        resolved directly from this stature and takes precedence over
        ``percentile``. Heights outside the ANSUR range are extrapolated
        (with a :class:`UserWarning`) rather than rejected.
    percentile : float, optional
        Common ANSUR percentile when ``height`` is not provided. Defaults to
        ``50.0``.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR reference CSV.
    model_path : str or pathlib.Path, optional
        Path to the OpenSim model file.
    """

    def __init__(
        self,
        gender: str,
        height: float | None = None,
        percentile: float = 50.0,
        *,
        dataset: str | Path = DEFAULT_DATASET,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        """Resolve anthropometry, load the base model, and scale it."""
        self._reference = resolve_reference(gender, height, percentile, dataset)
        # Register the mesh directory before the model file is loaded: the
        # bodies' attached Mesh geometry resolves its file immediately while
        # the model is being built, not lazily when show() runs. self isn't
        # a full OpenSimModel yet (super().__init__ hasn't run), so this
        # can't go through the self.add_geometry_directory instance method.
        _register_geometry_search_path(import_opensim(), DEFAULT_MESHES_DIR)
        super().__init__(model_path)
        self.add_geometry_directory(DEFAULT_MESHES_DIR)
        baseline = resolve_reference(self.gender, percentile=50.0, dataset=dataset)
        factors = segment_scale_factors(self._reference, baseline)
        self.scale_bodies(self._expand_bilateral_bodies(factors))

    @property
    def gender(self):
        """Return the normalized sex code used for the reference."""
        return self._reference.gender

    @property
    def height(self):
        """Return resolved stature in centimetres."""
        return self._reference.height_cm

    @property
    def percentile(self):
        """Return the effective ANSUR percentile."""
        return self._reference.percentile

    @property
    def anthropometry(self):
        """Return all resolved anthropometric measurements."""
        return self._reference

    def _joint_center(self, joint_name: str) -> tuple[float, float, float]:
        self.model.realizePosition(self.state)
        frame = self.joint(joint_name).getChildFrame()
        position = frame.getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def joint_centers(self) -> dict[str, tuple[float, float, float]]:
        """Return every joint centre in the ground frame, in metres.

        Reflects the model's current posture (realized automatically; no
        need to call :meth:`update_state` first).
        """
        return {name: self._joint_center(joint) for name, joint in _JOINT_CENTER_NAMES.items()}

    @property
    def pelvis(self) -> tuple[float, float, float]:
        """Return the pelvis joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("ground_pelvis")

    @property
    def left_hip(self) -> tuple[float, float, float]:
        """Return the left hip joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("hip_l")

    @property
    def right_hip(self) -> tuple[float, float, float]:
        """Return the right hip joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("hip_r")

    @property
    def left_knee(self) -> tuple[float, float, float]:
        """Return the left knee joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("walker_knee_l")

    @property
    def right_knee(self) -> tuple[float, float, float]:
        """Return the right knee joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("walker_knee_r")

    @property
    def left_patella(self) -> tuple[float, float, float]:
        """Return the left patella joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("patellofemoral_l")

    @property
    def right_patella(self) -> tuple[float, float, float]:
        """Return the right patella joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("patellofemoral_r")

    @property
    def left_ankle(self) -> tuple[float, float, float]:
        """Return the left ankle joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("ankle_l")

    @property
    def right_ankle(self) -> tuple[float, float, float]:
        """Return the right ankle joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("ankle_r")

    @property
    def left_subtalar(self) -> tuple[float, float, float]:
        """Return the left subtalar joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("subtalar_l")

    @property
    def right_subtalar(self) -> tuple[float, float, float]:
        """Return the right subtalar joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("subtalar_r")

    @property
    def left_mtp(self) -> tuple[float, float, float]:
        """Return the left mtp joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("mtp_l")

    @property
    def right_mtp(self) -> tuple[float, float, float]:
        """Return the right mtp joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("mtp_r")

    @property
    def torso(self) -> tuple[float, float, float]:
        """Return the torso joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("back")

    @property
    def left_shoulder(self) -> tuple[float, float, float]:
        """Return the left shoulder joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("acromial_l")

    @property
    def right_shoulder(self) -> tuple[float, float, float]:
        """Return the right shoulder joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("acromial_r")

    @property
    def left_elbow(self) -> tuple[float, float, float]:
        """Return the left elbow joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("elbow_l")

    @property
    def right_elbow(self) -> tuple[float, float, float]:
        """Return the right elbow joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("elbow_r")

    @property
    def left_radioulnar(self) -> tuple[float, float, float]:
        """Return the left radioulnar joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("radioulnar_l")

    @property
    def right_radioulnar(self) -> tuple[float, float, float]:
        """Return the right radioulnar joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("radioulnar_r")

    @property
    def left_wrist(self) -> tuple[float, float, float]:
        """Return the left wrist joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("radius_hand_l")

    @property
    def right_wrist(self) -> tuple[float, float, float]:
        """Return the right wrist joint centre in the ground frame, in metres, at the model's current posture."""
        return self._joint_center("radius_hand_r")

    def _marker_location(self, marker_name: str) -> tuple[float, float, float]:
        self.model.realizePosition(self.state)
        position = self.marker(marker_name).getLocationInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def foot_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return every foot landmark marker in the ground frame, in metres.

        Reflects the model's current posture (realized automatically; no
        need to call :meth:`update_state` first). There is no 1st-metatarsal
        marker in this model (only the 5th, see :attr:`left_mt5`).
        """
        return {name: self._marker_location(marker) for name, marker in _FOOT_MARKER_NAMES.items()}

    @property
    def left_heel(self) -> tuple[float, float, float]:
        """Return the left heel marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("LCAL")

    @property
    def right_heel(self) -> tuple[float, float, float]:
        """Return the right heel marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("RCAL")

    @property
    def left_toe(self) -> tuple[float, float, float]:
        """Return the left toe marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("LTOE")

    @property
    def right_toe(self) -> tuple[float, float, float]:
        """Return the right toe marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("RTOE")

    @property
    def left_mt5(self) -> tuple[float, float, float]:
        """Return the left 5th-metatarsal marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("LMT5")

    @property
    def right_mt5(self) -> tuple[float, float, float]:
        """Return the right 5th-metatarsal marker in the ground frame, in metres, at the model's current posture."""
        return self._marker_location("RMT5")

    @property
    def com(self) -> tuple[float, float, float]:
        """Return the whole-body centre of mass in the ground frame, in metres.

        Reflects the model's current posture (realized automatically; no
        need to call :meth:`update_state` first).
        """
        self.model.realizePosition(self.state)
        position = self.model.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def cop(self) -> tuple[float, float, float]:
        """Return the ground projection of the centre of mass, in metres.

        This is a kinematic projection of :attr:`com` straight down onto the
        ground plane (``y = 0``), not a dynamically computed centre of
        pressure from contact forces.
        """
        x, _, z = self.com
        return (x, 0.0, z)

    def set_position(
        self, reference: tuple[float, float, float], x: float, y: float, z: float
    ) -> None:
        """Translate the whole model so that ``reference`` ends up at ``(x, y, z)``.

        ``reference`` is any ground-frame point belonging to this model --
        e.g. :attr:`com`, :attr:`cop`, a :attr:`joint_centers` entry, or one
        of the named joint-centre properties (``left_ankle``, ``pelvis``,
        ...). The whole model is translated rigidly through its root
        (``pelvis_tx``/``pelvis_ty``/``pelvis_tz``): relative posture and
        joint angles are unaffected. Calls :meth:`update_state` internally,
        so derived quantities are immediately up to date.

        Parameters
        ----------
        reference : tuple[float, float, float]
            Current ground-frame position, in metres, of the point to move
            (read it right before calling, since it depends on the current
            posture).
        x, y, z : float
            Target ground-frame coordinates for that point, in metres.

        Raises
        ------
        ValueError
            If any coordinate is not finite.
        """
        if not all(math.isfinite(value) for value in (x, y, z, *reference)):
            raise ValueError("position must be finite")
        targets = (x, y, z)
        for name, target, current in zip(_PELVIS_TRANSLATION_COORDINATES, targets, reference):
            coordinate = self.coordinate(name)
            coordinate.setValue(self.state, coordinate.getValue(self.state) + (target - current), False)
        self.update_state()

    @property
    def left_foot_length(self) -> float:
        """Return left foot length in metres, from the ANSUR footlength measurement."""
        return self._reference.values["footlength"] / 1000.0

    @property
    def right_foot_length(self) -> float:
        """Return right foot length in metres, from the ANSUR footlength measurement."""
        return self._reference.values["footlength"] / 1000.0

    @property
    def left_foot_height(self) -> float:
        """Return left foot height above the ground in metres.

        Approximated as the left ankle joint centre's height, at the
        model's current posture.
        """
        return self._joint_center("ankle_l")[1]

    @property
    def right_foot_height(self) -> float:
        """Return right foot height above the ground in metres.

        Approximated as the right ankle joint centre's height, at the
        model's current posture.
        """
        return self._joint_center("ankle_r")[1]

    @property
    def left_thigh_length(self) -> float:
        """Return the left hip-to-knee distance in metres, at the current posture."""
        return math.dist(self._joint_center("hip_l"), self._joint_center("walker_knee_l"))

    @property
    def right_thigh_length(self) -> float:
        """Return the right hip-to-knee distance in metres, at the current posture."""
        return math.dist(self._joint_center("hip_r"), self._joint_center("walker_knee_r"))

    @property
    def left_shank_length(self) -> float:
        """Return the left knee-to-ankle distance in metres, at the current posture."""
        return math.dist(self._joint_center("walker_knee_l"), self._joint_center("ankle_l"))

    @property
    def right_shank_length(self) -> float:
        """Return the right knee-to-ankle distance in metres, at the current posture."""
        return math.dist(self._joint_center("walker_knee_r"), self._joint_center("ankle_r"))

    @property
    def torso_height(self) -> float:
        """Return the hip-centre-to-shoulder-centre distance in metres.

        The hip centre and shoulder centre are each the midpoint between
        the left and right hip (respectively shoulder) joint centres, at
        the model's current posture.
        """
        left_hip, right_hip = self._joint_center("hip_l"), self._joint_center("hip_r")
        left_shoulder, right_shoulder = (
            self._joint_center("acromial_l"),
            self._joint_center("acromial_r"),
        )
        hip_center = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
        shoulder_center = tuple((a + b) / 2 for a, b in zip(left_shoulder, right_shoulder))
        return math.dist(hip_center, shoulder_center)

    @property
    def shoulder_width(self) -> float:
        """Return the left-to-right shoulder joint centre distance in metres.

        Distance between the ``left_shoulder`` and ``right_shoulder`` joint
        centres, at the model's current posture. See
        :attr:`biacromial_breadth` for the ANSUR-measured equivalent.
        """
        return math.dist(self._joint_center("acromial_l"), self._joint_center("acromial_r"))

    @property
    def biacromial_breadth(self) -> float:
        """Return the biacromial breadth in metres, from ANSUR."""
        return self._reference.values["biacromialbreadth"] / 1000.0

    @property
    def left_arm_length(self) -> float:
        """Return the left shoulder-to-elbow distance in metres, at the current posture."""
        return math.dist(self._joint_center("acromial_l"), self._joint_center("elbow_l"))

    @property
    def right_arm_length(self) -> float:
        """Return the right shoulder-to-elbow distance in metres, at the current posture."""
        return math.dist(self._joint_center("acromial_r"), self._joint_center("elbow_r"))

    @property
    def left_forearm_length(self) -> float:
        """Return the left elbow-to-wrist distance in metres, at the current posture."""
        return math.dist(self._joint_center("elbow_l"), self._joint_center("radius_hand_l"))

    @property
    def right_forearm_length(self) -> float:
        """Return the right elbow-to-wrist distance in metres, at the current posture."""
        return math.dist(self._joint_center("elbow_r"), self._joint_center("radius_hand_r"))

    @property
    def left_palm_length(self) -> float:
        """Return left palm length in metres, from the ANSUR palmlength measurement."""
        return self._reference.values["palmlength"] / 1000.0

    @property
    def right_palm_length(self) -> float:
        """Return right palm length in metres, from the ANSUR palmlength measurement."""
        return self._reference.values["palmlength"] / 1000.0

    @property
    def left_arm_circumference(self) -> float:
        """Return left flexed-biceps circumference in metres, from ANSUR."""
        return self._reference.values["bicepscircumferenceflexed"] / 1000.0

    @property
    def right_arm_circumference(self) -> float:
        """Return right flexed-biceps circumference in metres, from ANSUR."""
        return self._reference.values["bicepscircumferenceflexed"] / 1000.0

    @property
    def left_forearm_circumference(self) -> float:
        """Return left flexed-forearm circumference in metres, from ANSUR."""
        return self._reference.values["forearmcircumferenceflexed"] / 1000.0

    @property
    def right_forearm_circumference(self) -> float:
        """Return right flexed-forearm circumference in metres, from ANSUR."""
        return self._reference.values["forearmcircumferenceflexed"] / 1000.0

    @property
    def neck_circumference(self) -> float:
        """Return neck circumference in metres, from ANSUR."""
        return self._reference.values["neckcircumference"] / 1000.0

    @property
    def chest_circumference(self) -> float:
        """Return chest circumference in metres, from ANSUR."""
        return self._reference.values["chestcircumference"] / 1000.0

    @property
    def chest_depth(self) -> float:
        """Return chest (sagittal) depth in metres, from ANSUR."""
        return self._reference.values["chestdepth"] / 1000.0

    @property
    def chest_width(self) -> float:
        """Return chest width (breadth) in metres, from ANSUR."""
        return self._reference.values["chestbreadth"] / 1000.0

    @property
    def waist_circumference(self) -> float:
        """Return waist circumference in metres, from ANSUR."""
        return self._reference.values["waistcircumference"] / 1000.0

    @property
    def waist_depth(self) -> float:
        """Return waist (sagittal) depth in metres, from ANSUR."""
        return self._reference.values["waistdepth"] / 1000.0

    @property
    def waist_width(self) -> float:
        """Return waist width (breadth) in metres, from ANSUR."""
        return self._reference.values["waistbreadth"] / 1000.0

    @property
    def hip_circumference(self) -> float:
        """Return hip (buttock) circumference in metres, from ANSUR."""
        return self._reference.values["buttockcircumference"] / 1000.0

    @property
    def hip_depth(self) -> float:
        """Return hip (buttock, sagittal) depth in metres, from ANSUR."""
        return self._reference.values["buttockdepth"] / 1000.0

    @property
    def hip_width(self) -> float:
        """Return hip width (breadth) in metres, from ANSUR."""
        return self._reference.values["hipbreadth"] / 1000.0

    @property
    def left_thigh_circumference(self) -> float:
        """Return left thigh circumference in metres, from ANSUR."""
        return self._reference.values["thighcircumference"] / 1000.0

    @property
    def right_thigh_circumference(self) -> float:
        """Return right thigh circumference in metres, from ANSUR."""
        return self._reference.values["thighcircumference"] / 1000.0

    @property
    def left_calf_circumference(self) -> float:
        """Return left calf circumference in metres, from ANSUR."""
        return self._reference.values["calfcircumference"] / 1000.0

    @property
    def right_calf_circumference(self) -> float:
        """Return right calf circumference in metres, from ANSUR."""
        return self._reference.values["calfcircumference"] / 1000.0

    @property
    def left_thigh_depth(self) -> float:
        """Return left thigh depth in metres, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against, so
        this derives a diameter from the thigh circumference instead
        (``circumference / pi``).
        """
        return self._reference.values["thighcircumference"] / 1000.0 / math.pi

    @property
    def right_thigh_depth(self) -> float:
        """Return right thigh depth in metres, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against, so
        this derives a diameter from the thigh circumference instead
        (``circumference / pi``).
        """
        return self._reference.values["thighcircumference"] / 1000.0 / math.pi

    @property
    def left_thigh_width(self) -> float:
        """Return left thigh width in metres, assuming a circular cross-section.

        Equal to :attr:`left_thigh_depth`: a circle has a single diameter.
        """
        return self.left_thigh_depth

    @property
    def right_thigh_width(self) -> float:
        """Return right thigh width in metres, assuming a circular cross-section.

        Equal to :attr:`right_thigh_depth`: a circle has a single diameter.
        """
        return self.right_thigh_depth

    @property
    def left_calf_depth(self) -> float:
        """Return left calf depth in metres, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``).
        """
        return self._reference.values["calfcircumference"] / 1000.0 / math.pi

    @property
    def right_calf_depth(self) -> float:
        """Return right calf depth in metres, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``).
        """
        return self._reference.values["calfcircumference"] / 1000.0 / math.pi

    @property
    def left_calf_width(self) -> float:
        """Return left calf width in metres, assuming a circular cross-section.

        Equal to :attr:`left_calf_depth`: a circle has a single diameter.
        """
        return self.left_calf_depth

    @property
    def right_calf_width(self) -> float:
        """Return right calf width in metres, assuming a circular cross-section.

        Equal to :attr:`right_calf_depth`: a circle has a single diameter.
        """
        return self.right_calf_depth

    def _expand_bilateral_bodies(self, factors: dict[str, tuple[float, float, float]]):
        expanded: dict[str, tuple[float, float, float]] = {}
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if self.bodies.contains(candidate):
                    expanded[candidate] = values
        return expanded

    def set_left_hip_flexionextension(self, degrees: float):
        """Set left hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_l", degrees)

    def set_right_hip_flexionextension(self, degrees: float):
        """Set right hip flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_flexion_r", degrees)

    def set_left_hip_adduction(self, degrees: float):
        """Set left hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_l", degrees)

    def set_right_hip_adduction(self, degrees: float):
        """Set right hip adduction/abduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_adduction_r", degrees)

    def set_left_hip_rotation(self, degrees: float):
        """Set left hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_l", degrees)

    def set_right_hip_rotation(self, degrees: float):
        """Set right hip rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("hip_rotation_r", degrees)

    def set_left_knee_flexionextension(self, degrees: float):
        """Set left knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_l", degrees)

    def set_right_knee_flexionextension(self, degrees: float):
        """Set right knee flexion/extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("knee_angle_r", degrees)

    def set_left_ankle_flexiondorsiflexion(self, degrees: float):
        """Set left ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_l", degrees)

    def set_right_ankle_flexiondorsiflexion(self, degrees: float):
        """Set right ankle flexion/dorsiflexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("ankle_angle_r", degrees)

    def set_left_subtalar_inversion(self, degrees: float):
        """Set left subtalar inversion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks subtalar coordinates.
        """
        self.set_coordinate_degrees("subtalar_angle_l", degrees)

    def set_right_subtalar_inversion(self, degrees: float):
        """Set right subtalar inversion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks subtalar coordinates.
        """
        self.set_coordinate_degrees("subtalar_angle_r", degrees)

    def set_left_mtp_flexion(self, degrees: float):
        """Set left metatarsophalangeal flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks metatarsophalangeal coordinates.
        """
        self.set_coordinate_degrees("mtp_angle_l", degrees)

    def set_right_mtp_flexion(self, degrees: float):
        """Set right metatarsophalangeal flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.

        Raises
        ------
        ValueError
            The supplied model locks metatarsophalangeal coordinates.
        """
        self.set_coordinate_degrees("mtp_angle_r", degrees)

    def set_left_shoulder_flexion(self, degrees: float):
        """Set left shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_l", degrees)

    def set_right_shoulder_flexion(self, degrees: float):
        """Set right shoulder flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_flex_r", degrees)

    def set_left_shoulder_adduction(self, degrees: float):
        """Set left shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_l", degrees)

    def set_right_shoulder_adduction(self, degrees: float):
        """Set right shoulder adduction coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_add_r", degrees)

    def set_left_shoulder_rotation(self, degrees: float):
        """Set left shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_l", degrees)

    def set_right_shoulder_rotation(self, degrees: float):
        """Set right shoulder rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("arm_rot_r", degrees)

    def set_left_elbow_flexion(self, degrees: float):
        """Set left elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_l", degrees)

    def set_right_elbow_flexion(self, degrees: float):
        """Set right elbow flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("elbow_flex_r", degrees)

    def set_left_wrist_flexion(self, degrees: float):
        """Set left wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_l", degrees)

    def set_right_wrist_flexion(self, degrees: float):
        """Set right wrist flexion coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_flex_r", degrees)

    def set_left_wrist_deviation(self, degrees: float):
        """Set left wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_l", degrees)

    def set_right_wrist_deviation(self, degrees: float):
        """Set right wrist deviation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("wrist_dev_r", degrees)

    def set_left_forearm_pronation(self, degrees: float):
        """Set left forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_l", degrees)

    def set_right_forearm_pronation(self, degrees: float):
        """Set right forearm pronation/supination coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("pro_sup_r", degrees)

    def set_lumbar_extension(self, degrees: float):
        """Set lumbar extension coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_extension", degrees)

    def set_lumbar_bending(self, degrees: float):
        """Set lumbar lateral bending coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_bending", degrees)

    def set_lumbar_rotation(self, degrees: float):
        """Set lumbar rotation coordinate in degrees.

        Parameters
        ----------
        degrees : float
            Angle in degrees.
        """
        self.set_coordinate_degrees("lumbar_rotation", degrees)

    @property
    def left_hip_flexionextension(self) -> float:
        """Return the current left hip flexion/extension coordinate in degrees.

        Read-only counterpart of ``set_left_hip_flexionextension``.
        """
        return self.coordinate_degrees("hip_flexion_l")

    @property
    def right_hip_flexionextension(self) -> float:
        """Return the current right hip flexion/extension coordinate in degrees.

        Read-only counterpart of ``set_right_hip_flexionextension``.
        """
        return self.coordinate_degrees("hip_flexion_r")

    @property
    def left_hip_adduction(self) -> float:
        """Return the current left hip adduction coordinate in degrees.

        Read-only counterpart of ``set_left_hip_adduction``.
        """
        return self.coordinate_degrees("hip_adduction_l")

    @property
    def right_hip_adduction(self) -> float:
        """Return the current right hip adduction coordinate in degrees.

        Read-only counterpart of ``set_right_hip_adduction``.
        """
        return self.coordinate_degrees("hip_adduction_r")

    @property
    def left_hip_rotation(self) -> float:
        """Return the current left hip rotation coordinate in degrees.

        Read-only counterpart of ``set_left_hip_rotation``.
        """
        return self.coordinate_degrees("hip_rotation_l")

    @property
    def right_hip_rotation(self) -> float:
        """Return the current right hip rotation coordinate in degrees.

        Read-only counterpart of ``set_right_hip_rotation``.
        """
        return self.coordinate_degrees("hip_rotation_r")

    @property
    def left_knee_flexionextension(self) -> float:
        """Return the current left knee flexion/extension coordinate in degrees.

        Read-only counterpart of ``set_left_knee_flexionextension``.
        """
        return self.coordinate_degrees("knee_angle_l")

    @property
    def right_knee_flexionextension(self) -> float:
        """Return the current right knee flexion/extension coordinate in degrees.

        Read-only counterpart of ``set_right_knee_flexionextension``.
        """
        return self.coordinate_degrees("knee_angle_r")

    @property
    def left_ankle_flexiondorsiflexion(self) -> float:
        """Return the current left ankle flexion/dorsiflexion coordinate in degrees.

        Read-only counterpart of ``set_left_ankle_flexiondorsiflexion``.
        """
        return self.coordinate_degrees("ankle_angle_l")

    @property
    def right_ankle_flexiondorsiflexion(self) -> float:
        """Return the current right ankle flexion/dorsiflexion coordinate in degrees.

        Read-only counterpart of ``set_right_ankle_flexiondorsiflexion``.
        """
        return self.coordinate_degrees("ankle_angle_r")

    @property
    def left_subtalar_inversion(self) -> float:
        """Return the current left subtalar inversion coordinate in degrees.

        Read-only counterpart of ``set_left_subtalar_inversion``.
        """
        return self.coordinate_degrees("subtalar_angle_l")

    @property
    def right_subtalar_inversion(self) -> float:
        """Return the current right subtalar inversion coordinate in degrees.

        Read-only counterpart of ``set_right_subtalar_inversion``.
        """
        return self.coordinate_degrees("subtalar_angle_r")

    @property
    def left_mtp_flexion(self) -> float:
        """Return the current left mtp flexion coordinate in degrees.

        Read-only counterpart of ``set_left_mtp_flexion``.
        """
        return self.coordinate_degrees("mtp_angle_l")

    @property
    def right_mtp_flexion(self) -> float:
        """Return the current right mtp flexion coordinate in degrees.

        Read-only counterpart of ``set_right_mtp_flexion``.
        """
        return self.coordinate_degrees("mtp_angle_r")

    @property
    def left_shoulder_flexion(self) -> float:
        """Return the current left shoulder flexion coordinate in degrees.

        Read-only counterpart of ``set_left_shoulder_flexion``.
        """
        return self.coordinate_degrees("arm_flex_l")

    @property
    def right_shoulder_flexion(self) -> float:
        """Return the current right shoulder flexion coordinate in degrees.

        Read-only counterpart of ``set_right_shoulder_flexion``.
        """
        return self.coordinate_degrees("arm_flex_r")

    @property
    def left_shoulder_adduction(self) -> float:
        """Return the current left shoulder adduction coordinate in degrees.

        Read-only counterpart of ``set_left_shoulder_adduction``.
        """
        return self.coordinate_degrees("arm_add_l")

    @property
    def right_shoulder_adduction(self) -> float:
        """Return the current right shoulder adduction coordinate in degrees.

        Read-only counterpart of ``set_right_shoulder_adduction``.
        """
        return self.coordinate_degrees("arm_add_r")

    @property
    def left_shoulder_rotation(self) -> float:
        """Return the current left shoulder rotation coordinate in degrees.

        Read-only counterpart of ``set_left_shoulder_rotation``.
        """
        return self.coordinate_degrees("arm_rot_l")

    @property
    def right_shoulder_rotation(self) -> float:
        """Return the current right shoulder rotation coordinate in degrees.

        Read-only counterpart of ``set_right_shoulder_rotation``.
        """
        return self.coordinate_degrees("arm_rot_r")

    @property
    def left_elbow_flexion(self) -> float:
        """Return the current left elbow flexion coordinate in degrees.

        Read-only counterpart of ``set_left_elbow_flexion``.
        """
        return self.coordinate_degrees("elbow_flex_l")

    @property
    def right_elbow_flexion(self) -> float:
        """Return the current right elbow flexion coordinate in degrees.

        Read-only counterpart of ``set_right_elbow_flexion``.
        """
        return self.coordinate_degrees("elbow_flex_r")

    @property
    def left_wrist_flexion(self) -> float:
        """Return the current left wrist flexion coordinate in degrees.

        Read-only counterpart of ``set_left_wrist_flexion``.
        """
        return self.coordinate_degrees("wrist_flex_l")

    @property
    def right_wrist_flexion(self) -> float:
        """Return the current right wrist flexion coordinate in degrees.

        Read-only counterpart of ``set_right_wrist_flexion``.
        """
        return self.coordinate_degrees("wrist_flex_r")

    @property
    def left_wrist_deviation(self) -> float:
        """Return the current left wrist deviation coordinate in degrees.

        Read-only counterpart of ``set_left_wrist_deviation``.
        """
        return self.coordinate_degrees("wrist_dev_l")

    @property
    def right_wrist_deviation(self) -> float:
        """Return the current right wrist deviation coordinate in degrees.

        Read-only counterpart of ``set_right_wrist_deviation``.
        """
        return self.coordinate_degrees("wrist_dev_r")

    @property
    def left_forearm_pronation(self) -> float:
        """Return the current left forearm pronation coordinate in degrees.

        Read-only counterpart of ``set_left_forearm_pronation``.
        """
        return self.coordinate_degrees("pro_sup_l")

    @property
    def right_forearm_pronation(self) -> float:
        """Return the current right forearm pronation coordinate in degrees.

        Read-only counterpart of ``set_right_forearm_pronation``.
        """
        return self.coordinate_degrees("pro_sup_r")

    @property
    def lumbar_extension(self) -> float:
        """Return the current lumbar extension coordinate in degrees.

        Read-only counterpart of ``set_lumbar_extension``.
        """
        return self.coordinate_degrees("lumbar_extension")

    @property
    def lumbar_bending(self) -> float:
        """Return the current lumbar bending coordinate in degrees.

        Read-only counterpart of ``set_lumbar_bending``.
        """
        return self.coordinate_degrees("lumbar_bending")

    @property
    def lumbar_rotation(self) -> float:
        """Return the current lumbar rotation coordinate in degrees.

        Read-only counterpart of ``set_lumbar_rotation``.
        """
        return self.coordinate_degrees("lumbar_rotation")
