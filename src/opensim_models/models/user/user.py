from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from ...model import OpenSimModel, _register_geometry_search_path, import_opensim
from ._data import DEFAULT_DATASET, resolve_reference
from ._joint_center_table import JOINT_CENTER_NAMES as _JOINT_CENTER_NAMES
from ._joint_centers_generated import _JointCenterMixin
from ._mapping import segment_scale_factors
from ._posture_generated import _PostureMixin
from ._posture_table import POSTURE_COORDINATE_NAMES as _POSTURE_COORDINATE_NAMES

__all__ = ["User"]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL_PATH = _ASSETS_DIR / "rajagopalaiulrich2023.osim"
DEFAULT_MESHES_DIR = _ASSETS_DIR / "meshes"

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


class User(OpenSimModel, _PostureMixin, _JointCenterMixin):
    """An ANSUR-based anthropometric user backed by an OpenSim model.

    Built from the Rajagopal-Lai-Uhlrich full-body OpenSim model
    (``rajagopalaiulrich2023.osim``, bundled under ``assets/``), scaled
    per-body to the requested anthropometry using ANSUR II reference data
    (see "Scaling antropometrico" in the package README). Every joint angle
    is exposed through a ``set_<joint>_<motion>``/``<joint>_<motion>``
    setter/getter pair (see the posture methods further below), and every
    major joint centre, segment length, and ANSUR-derived body measurement
    is exposed as a read-only property.

    Parameters
    ----------
    gender : str | Literal['M', 'F']
        Sex code used to select the ANSUR II reference population. Must be
        exactly ``"M"`` or ``"F"`` (case-sensitive, no other values
        accepted).
    height_cm : float or None, optional
        Requested stature, in centimetres. If provided, every anthropometric
        measurement (stature included) is resolved directly from this
        height via a per-measurement PCHIP regression against the ANSUR
        subjects of the requested ``gender``, and takes precedence over
        ``percentile`` (which is still computed afterwards, purely for
        information -- see :attr:`percentile`). A height outside the ANSUR
        range observed for that gender is not rejected: it is extrapolated
        (PCHIP with linear tails) and raises a :class:`UserWarning`, with
        extrapolation reliability dropping the further the requested height
        is from the observed range.
    percentile : float, optional
        Common ANSUR percentile applied to every numeric measurement when
        ``height`` is not provided. Defaults to ``50.0``. Must lie in the
        inclusive interval ``[0.1, 99.9]``; this applies ``numpy.percentile``
        independently per measurement (no single reference subject is
        selected, and no interpolation is performed across different
        measurements).
    dataset : str or pathlib.Path, optional
        Path to the ANSUR II reference CSV to resolve measurements from.
        Defaults to the dataset bundled with this package.
    model_path : str or pathlib.Path, optional
        Path to the OpenSim ``.osim`` model file to load and scale.
        Defaults to the bundled Rajagopal-Lai-Uhlrich model.

    Raises
    ------
    ValueError
        If ``gender`` is not ``"M"``/``"F"``, if ``height`` is given but is
        not a positive finite number, or if ``percentile`` is not finite or
        falls outside ``[0.1, 99.9]``.
    """

    def __init__(
        self,
        gender: str | Literal["M", "F"],
        height_cm: float | None = None,
        percentile: float = 50.0,
        *,
        dataset: str | Path = DEFAULT_DATASET,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        """Resolve anthropometry, load the base model, and scale it.

        See the class docstring for the full description of ``gender``,
        ``height``, ``percentile``, ``dataset``, and ``model_path``, and of
        the conditions under which a :class:`ValueError` or
        :class:`UserWarning` is raised. After resolving the anthropometric
        reference, this loads ``model_path`` (unlocking every coordinate --
        see ``OpenSimModel._unlock_coordinates``), registers the bundled
        mesh directory for rendering, and scales every body via
        :meth:`scale_bodies` using per-segment factors derived by comparing
        the resolved reference against the 50th-percentile baseline for the
        same ``gender`` (see "Scaling antropometrico" in the README).
        """
        self._reference = resolve_reference(gender, height_cm, percentile, dataset)
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
        """Return the normalized sex code used to resolve this user's anthropometry.

        Returns
        -------
        str
            ``"M"`` or ``"F"``, matching the ``gender`` passed to the
            constructor (normalization only affects case/whitespace, not
            the two accepted values themselves).
        """
        return self._reference.gender

    @property
    def height(self):
        """Return this user's resolved stature, in centimetres.

        If the constructor was called with ``height=...``, this is exactly
        that value (even if it was outside the ANSUR range and triggered
        extrapolation). Otherwise it is the ANSUR stature at the resolved
        ``percentile`` for this user's ``gender``.

        Returns
        -------
        float
            Stature, in centimetres.
        """
        return self._reference.height_cm

    @property
    def percentile(self):
        """Return the ANSUR percentile associated with this user.

        When the constructor was called with ``percentile=...`` (the
        default path), this is exactly that value, applied independently to
        every ANSUR measurement. When it was called with ``height=...``
        instead, every measurement is resolved directly from the requested
        height (not from this percentile), and this property instead
        reports the *empirical* percentile of that height within the ANSUR
        population of this user's ``gender`` -- informational only, see the
        constructor's ``height`` parameter.

        Returns
        -------
        float
            Percentile in the inclusive range ``[0.1, 99.9]``.
        """
        return self._reference.percentile

    @property
    def anthropometry(self):
        """Return every anthropometric measurement resolved for this user.

        Returns
        -------
        opensim_models.models.user._data.AnthropometricReference
            Internal reference object exposing ``gender``, ``percentile``,
            ``values`` (a ``dict`` mapping every numeric ANSUR II column
            resolved for this user to its value in the dataset's original
            units -- predominantly millimetres, with ``values["stature_m"]``
            in metres; see "Dati ANSUR risolti" in the README), and the
            convenience properties ``height_m``/``height_cm`` (stature only,
            derived from ``values["stature_m"]``). This type is an internal
            implementation detail (not part of the package's re-exported
            public surface), but the object itself and its attributes
            remain freely readable once a :class:`User` has been built.
        """
        return self._reference

    def _joint_center(self, joint_name: str) -> tuple[float, float, float]:
        self.model.realizePosition(self.state)
        frame = self.joint(joint_name).raw.getChildFrame()
        position = frame.getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def joint_centers(self) -> dict[str, tuple[float, float, float]]:
        """Return every named joint centre at once, in the ground frame.

        Equivalent to calling every individual joint-centre property below
        (``pelvis``, ``left_hip``, ``right_knee``, ...) and collecting the
        results into one dictionary, keyed by the same friendly names.
        Reflects the model's current posture: each position is computed via
        ``getPositionInGround`` with an automatic ``realizePosition``, so it
        is correct immediately after a posture setter, with no need to call
        :meth:`update_state` first.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly joint name (``"pelvis"``, ``"left_hip"``,
            ``"right_ankle"``, ...) to that joint's centre as an ``(x, y,
            z)`` position in the ground frame, in metres.
        """
        return {
            name: self._joint_center(joint)
            for name, joint in _JOINT_CENTER_NAMES.items()
        }

    def _marker_location(self, marker_name: str) -> tuple[float, float, float]:
        self.model.realizePosition(self.state)
        position = self.marker(marker_name).raw.getLocationInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def foot_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return every foot landmark marker at once, in the ground frame.

        Equivalent to calling each of ``left_heel``/``right_heel``,
        ``left_toe``/``right_toe``, and ``left_mt5``/``right_mt5`` and
        collecting the results into one dictionary. Reflects the model's
        current posture (``realizePosition`` is called automatically; no
        need to call :meth:`update_state` first). There is no 1st-metatarsal
        marker in this model (only the 5th, see :attr:`left_mt5`), so none
        is exposed here either.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly marker name (``"left_heel"``,
            ``"right_toe"``, ``"left_mt5"``, ...) to that marker's location
            as an ``(x, y, z)`` position in the ground frame, in metres.
        """
        return {
            name: self._marker_location(marker)
            for name, marker in _FOOT_MARKER_NAMES.items()
        }

    @property
    def left_heel(self) -> tuple[float, float, float]:
        """Return the left heel marker location, in the ground frame.

        Reads the OpenSim marker ``LCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LCAL")

    @property
    def right_heel(self) -> tuple[float, float, float]:
        """Return the right heel marker location, in the ground frame.

        Reads the OpenSim marker ``RCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RCAL")

    @property
    def left_toe(self) -> tuple[float, float, float]:
        """Return the left toe marker location, in the ground frame.

        Reads the OpenSim marker ``LTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LTOE")

    @property
    def right_toe(self) -> tuple[float, float, float]:
        """Return the right toe marker location, in the ground frame.

        Reads the OpenSim marker ``RTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RTOE")

    @property
    def left_mt5(self) -> tuple[float, float, float]:
        """Return the left 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``LMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LMT5")

    @property
    def right_mt5(self) -> tuple[float, float, float]:
        """Return the right 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``RMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RMT5")

    @property
    def com(self) -> tuple[float, float, float]:
        """Return the whole-body centre of mass, in the ground frame.

        Computed natively by OpenSim (``Model.calcMassCenterPosition``),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need to call :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        self.model.realizePosition(self.state)
        position = self.model.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def cop(self) -> tuple[float, float, float]:
        """Return the ground projection of the centre of mass.

        This is a purely kinematic projection of :attr:`com` straight down
        onto the ground plane (``y = 0``): the ``x``/``z`` coordinates are
        copied unchanged from :attr:`com` and ``y`` is forced to ``0.0``.
        Despite the name, this is **not** a dynamically computed centre of
        pressure derived from contact/ground-reaction forces.

        Returns
        -------
        tuple[float, float, float]
            ``(x, 0.0, z)`` position in metres, in the ground frame, where
            ``x``/``z`` match :attr:`com`.
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
        for name, target, current in zip(
            _PELVIS_TRANSLATION_COORDINATES, targets, reference
        ):
            coordinate = self.coordinate(name)
            coordinate.set_value(
                coordinate.value + (target - current), enforce_constraints=False
            )
        self.update_state()

    @property
    def left_foot_length(self) -> float:
        """Return left foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._reference.values["footlength"] / 1000.0

    @property
    def right_foot_length(self) -> float:
        """Return right foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre. ANSUR does not record left/right foot length separately, so
        this is identical to :attr:`left_foot_length`.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._reference.values["footlength"] / 1000.0

    @property
    def left_foot_height(self) -> float:
        """Return the left foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`left_ankle` (the
        left ankle joint centre), at the model's current posture -- this is
        an approximation of the sole's height, not a direct measurement of
        the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_l")[1]

    @property
    def right_foot_height(self) -> float:
        """Return the right foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`right_ankle` (the
        right ankle joint centre), at the model's current posture -- this
        is an approximation of the sole's height, not a direct measurement
        of the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_r")[1]

    @property
    def left_thigh_length(self) -> float:
        """Return the left thigh length, at the model's current posture.

        Euclidean distance between :attr:`left_hip` and :attr:`left_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_l"), self._joint_center("walker_knee_l")
        )

    @property
    def right_thigh_length(self) -> float:
        """Return the right thigh length, at the model's current posture.

        Euclidean distance between :attr:`right_hip` and :attr:`right_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_r"), self._joint_center("walker_knee_r")
        )

    @property
    def left_shank_length(self) -> float:
        """Return the left shank length, at the model's current posture.

        Euclidean distance between :attr:`left_knee` and :attr:`left_ankle`
        in the ground frame; since both depend on posture, this changes as
        the knee/ankle angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_l"), self._joint_center("ankle_l")
        )

    @property
    def right_shank_length(self) -> float:
        """Return the right shank length, at the model's current posture.

        Euclidean distance between :attr:`right_knee` and
        :attr:`right_ankle` in the ground frame; since both depend on
        posture, this changes as the knee/ankle angles change (it is not a
        fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_r"), self._joint_center("ankle_r")
        )

    @property
    def torso_height(self) -> float:
        """Return the trunk height, at the model's current posture.

        Euclidean distance between a "hip centre" and a "shoulder centre",
        each the midpoint between the corresponding left/right joint
        centres (:attr:`left_hip`/:attr:`right_hip` and
        :attr:`left_shoulder`/:attr:`right_shoulder`). Since all four
        inputs depend on posture, this changes with the current hip/spine
        posture (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-centre-to-shoulder-centre distance, in metres.
        """
        left_hip, right_hip = self._joint_center("hip_l"), self._joint_center("hip_r")
        left_shoulder, right_shoulder = (
            self._joint_center("acromial_l"),
            self._joint_center("acromial_r"),
        )
        hip_center = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
        shoulder_center = tuple(
            (a + b) / 2 for a, b in zip(left_shoulder, right_shoulder)
        )
        return math.dist(hip_center, shoulder_center)

    @property
    def shoulder_width(self) -> float:
        """Return the left-to-right shoulder joint centre distance.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`right_shoulder`, at the model's current posture -- a
        geometric measurement, not read from ANSUR. See
        :attr:`biacromial_breadth` for the ANSUR-measured equivalent; the
        two are independent and generally do not coincide numerically (see
        the README's "Centri articolari e misure derivate" section for why).

        Returns
        -------
        float
            Shoulder-to-shoulder distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("acromial_r")
        )

    @property
    def biacromial_breadth(self) -> float:
        """Return the biacromial breadth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``biacromialbreadth``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. See :attr:`shoulder_width` for the
        posture-dependent geometric equivalent derived from the model's
        shoulder joint centres.

        Returns
        -------
        float
            Biacromial breadth, in metres.
        """
        return self._reference.values["biacromialbreadth"] / 1000.0

    @property
    def left_arm_length(self) -> float:
        """Return the left upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`left_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("elbow_l")
        )

    @property
    def right_arm_length(self) -> float:
        """Return the right upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`right_shoulder` and
        :attr:`right_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_r"), self._joint_center("elbow_r")
        )

    @property
    def left_forearm_length(self) -> float:
        """Return the left forearm length, at the model's current posture.

        Euclidean distance between :attr:`left_elbow` and :attr:`left_wrist`
        in the ground frame; since both depend on posture, this changes as
        the elbow/wrist/forearm angles change (it is not a fixed ANSUR
        measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_l"), self._joint_center("radius_hand_l")
        )

    @property
    def right_forearm_length(self) -> float:
        """Return the right forearm length, at the model's current posture.

        Euclidean distance between :attr:`right_elbow` and
        :attr:`right_wrist` in the ground frame; since both depend on
        posture, this changes as the elbow/wrist/forearm angles change (it
        is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_r"), self._joint_center("radius_hand_r")
        )

    @property
    def left_palm_length(self) -> float:
        """Return left palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers (ANSUR II and the
        underlying OpenSim model do not provide individual finger lengths,
        see the README).

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._reference.values["palmlength"] / 1000.0

    @property
    def right_palm_length(self) -> float:
        """Return right palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers. ANSUR does not
        record left/right palm length separately, so this is identical to
        :attr:`left_palm_length`.

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._reference.values["palmlength"] / 1000.0

    @property
    def left_arm_circumference(self) -> float:
        """Return left flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._reference.values["bicepscircumferenceflexed"] / 1000.0

    @property
    def right_arm_circumference(self) -> float:
        """Return right flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_arm_circumference`.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._reference.values["bicepscircumferenceflexed"] / 1000.0

    @property
    def left_forearm_circumference(self) -> float:
        """Return left flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._reference.values["forearmcircumferenceflexed"] / 1000.0

    @property
    def right_forearm_circumference(self) -> float:
        """Return right flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.
        ANSUR does not record left/right circumference separately, so this
        is identical to :attr:`left_forearm_circumference`.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._reference.values["forearmcircumferenceflexed"] / 1000.0

    @property
    def neck_circumference(self) -> float:
        """Return neck circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``neckcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Neck circumference, in metres.
        """
        return self._reference.values["neckcircumference"] / 1000.0

    @property
    def chest_circumference(self) -> float:
        """Return chest circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Chest circumference, in metres.
        """
        return self._reference.values["chestcircumference"] / 1000.0

    @property
    def chest_depth(self) -> float:
        """Return chest (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest depth (front-to-back), in metres.
        """
        return self._reference.values["chestdepth"] / 1000.0

    @property
    def chest_width(self) -> float:
        """Return chest width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest width (side-to-side), in metres.
        """
        return self._reference.values["chestbreadth"] / 1000.0

    @property
    def waist_circumference(self) -> float:
        """Return waist circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Waist circumference, in metres.
        """
        return self._reference.values["waistcircumference"] / 1000.0

    @property
    def waist_depth(self) -> float:
        """Return waist (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist depth (front-to-back), in metres.
        """
        return self._reference.values["waistdepth"] / 1000.0

    @property
    def waist_width(self) -> float:
        """Return waist width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist width (side-to-side), in metres.
        """
        return self._reference.values["waistbreadth"] / 1000.0

    @property
    def hip_circumference(self) -> float:
        """Return hip (buttock) circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Hip/buttock circumference, in metres.
        """
        return self._reference.values["buttockcircumference"] / 1000.0

    @property
    def hip_depth(self) -> float:
        """Return hip (buttock, sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip/buttock depth (front-to-back), in metres.
        """
        return self._reference.values["buttockdepth"] / 1000.0

    @property
    def hip_width(self) -> float:
        """Return hip width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``hipbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip width (side-to-side), in metres.
        """
        return self._reference.values["hipbreadth"] / 1000.0

    @property
    def left_thigh_circumference(self) -> float:
        """Return left thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._reference.values["thighcircumference"] / 1000.0

    @property
    def right_thigh_circumference(self) -> float:
        """Return right thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_thigh_circumference`.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._reference.values["thighcircumference"] / 1000.0

    @property
    def left_calf_circumference(self) -> float:
        """Return left calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._reference.values["calfcircumference"] / 1000.0

    @property
    def right_calf_circumference(self) -> float:
        """Return right calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_calf_circumference`.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._reference.values["calfcircumference"] / 1000.0

    @property
    def left_thigh_depth(self) -> float:
        """Return left thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self._reference.values["thighcircumference"] / 1000.0 / math.pi

    @property
    def right_thigh_depth(self) -> float:
        """Return right thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self._reference.values["thighcircumference"] / 1000.0 / math.pi

    @property
    def left_thigh_width(self) -> float:
        """Return left thigh width, assuming a circular cross-section.

        Equal to :attr:`left_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.left_thigh_depth

    @property
    def right_thigh_width(self) -> float:
        """Return right thigh width, assuming a circular cross-section.

        Equal to :attr:`right_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.right_thigh_depth

    @property
    def left_calf_depth(self) -> float:
        """Return left calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self._reference.values["calfcircumference"] / 1000.0 / math.pi

    @property
    def right_calf_depth(self) -> float:
        """Return right calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self._reference.values["calfcircumference"] / 1000.0 / math.pi

    @property
    def left_calf_width(self) -> float:
        """Return left calf width, assuming a circular cross-section.

        Equal to :attr:`left_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.left_calf_depth

    @property
    def right_calf_width(self) -> float:
        """Return right calf width, assuming a circular cross-section.

        Equal to :attr:`right_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.right_calf_depth

    def _expand_bilateral_bodies(self, factors: dict[str, tuple[float, float, float]]):
        expanded: dict[str, tuple[float, float, float]] = {}
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if candidate in self.bodies:
                    expanded[candidate] = values
        return expanded
