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
    gender : str
        Sex code used to select the ANSUR II reference population. Must be
        exactly ``"M"`` or ``"F"`` (case-sensitive, no other values
        accepted).
    height : float or None, optional
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
        gender: str,
        height: float | None = None,
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
        return {name: self._joint_center(joint) for name, joint in _JOINT_CENTER_NAMES.items()}

    @property
    def pelvis(self) -> tuple[float, float, float]:
        """Return the pelvis joint centre, in the ground frame.

        This is the child frame origin of the ``ground_pelvis`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("ground_pelvis")

    @property
    def left_hip(self) -> tuple[float, float, float]:
        """Return the left hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("hip_l")

    @property
    def right_hip(self) -> tuple[float, float, float]:
        """Return the right hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("hip_r")

    @property
    def left_knee(self) -> tuple[float, float, float]:
        """Return the left knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("walker_knee_l")

    @property
    def right_knee(self) -> tuple[float, float, float]:
        """Return the right knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("walker_knee_r")

    @property
    def left_patella(self) -> tuple[float, float, float]:
        """Return the left patellofemoral joint centre, in the ground frame.

        This is the child frame origin of the ``patellofemoral_l`` joint.
        The patella's own coordinate (``knee_angle_l_beta``) is kinematically
        coupled to ``knee_angle_l`` via a ``CoordinateCouplerConstraint``;
        this position reflects the model's current posture, but that
        coupling is only resolved by ``realizePosition``/``assemble``-level
        updates, not by a plain :meth:`update_state` after a mid-edit knee
        angle change (see :meth:`set_left_knee_flexionextension` and the
        README's "Limiti noti di update_state()").

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("patellofemoral_l")

    @property
    def right_patella(self) -> tuple[float, float, float]:
        """Return the right patellofemoral joint centre, in the ground frame.

        This is the child frame origin of the ``patellofemoral_r`` joint.
        The patella's own coordinate (``knee_angle_r_beta``) is kinematically
        coupled to ``knee_angle_r`` via a ``CoordinateCouplerConstraint``;
        this position reflects the model's current posture, but that
        coupling is only resolved by ``realizePosition``/``assemble``-level
        updates, not by a plain :meth:`update_state` after a mid-edit knee
        angle change (see :meth:`set_right_knee_flexionextension` and the
        README's "Limiti noti di update_state()").

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("patellofemoral_r")

    @property
    def left_ankle(self) -> tuple[float, float, float]:
        """Return the left ankle joint centre, in the ground frame.

        This is the child frame origin of the ``ankle_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first). Also used
        as the approximation for :attr:`left_foot_height`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("ankle_l")

    @property
    def right_ankle(self) -> tuple[float, float, float]:
        """Return the right ankle joint centre, in the ground frame.

        This is the child frame origin of the ``ankle_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first). Also used
        as the approximation for :attr:`right_foot_height`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("ankle_r")

    @property
    def left_subtalar(self) -> tuple[float, float, float]:
        """Return the left subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("subtalar_l")

    @property
    def right_subtalar(self) -> tuple[float, float, float]:
        """Return the right subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("subtalar_r")

    @property
    def left_mtp(self) -> tuple[float, float, float]:
        """Return the left metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("mtp_l")

    @property
    def right_mtp(self) -> tuple[float, float, float]:
        """Return the right metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("mtp_r")

    @property
    def torso(self) -> tuple[float, float, float]:
        """Return the torso (back) joint centre, in the ground frame.

        This is the child frame origin of the ``back`` joint (the lumbar
        joint connecting pelvis and torso), reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("back")

    @property
    def left_shoulder(self) -> tuple[float, float, float]:
        """Return the left shoulder joint centre, in the ground frame.

        This is the child frame origin of the ``acromial_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first). See
        :attr:`shoulder_width` and :attr:`biacromial_breadth` for two
        different measures derived from this and :attr:`right_shoulder`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("acromial_l")

    @property
    def right_shoulder(self) -> tuple[float, float, float]:
        """Return the right shoulder joint centre, in the ground frame.

        This is the child frame origin of the ``acromial_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first). See
        :attr:`shoulder_width` and :attr:`biacromial_breadth` for two
        different measures derived from this and :attr:`left_shoulder`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("acromial_r")

    @property
    def left_elbow(self) -> tuple[float, float, float]:
        """Return the left elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("elbow_l")

    @property
    def right_elbow(self) -> tuple[float, float, float]:
        """Return the right elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("elbow_r")

    @property
    def left_radioulnar(self) -> tuple[float, float, float]:
        """Return the left radioulnar joint centre, in the ground frame.

        This is the child frame origin of the ``radioulnar_l`` joint (the
        joint whose coordinate is driven by :meth:`set_left_forearm_pronation`),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("radioulnar_l")

    @property
    def right_radioulnar(self) -> tuple[float, float, float]:
        """Return the right radioulnar joint centre, in the ground frame.

        This is the child frame origin of the ``radioulnar_r`` joint (the
        joint whose coordinate is driven by :meth:`set_right_forearm_pronation`),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("radioulnar_r")

    @property
    def left_wrist(self) -> tuple[float, float, float]:
        """Return the left wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("radius_hand_l")

    @property
    def right_wrist(self) -> tuple[float, float, float]:
        """Return the right wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center("radius_hand_r")

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
        return {name: self._marker_location(marker) for name, marker in _FOOT_MARKER_NAMES.items()}

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
        for name, target, current in zip(_PELVIS_TRANSLATION_COORDINATES, targets, reference):
            coordinate = self.coordinate(name)
            coordinate.set_value(coordinate.value + (target - current), enforce_constraints=False)
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
        return math.dist(self._joint_center("hip_l"), self._joint_center("walker_knee_l"))

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
        return math.dist(self._joint_center("hip_r"), self._joint_center("walker_knee_r"))

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
        return math.dist(self._joint_center("walker_knee_l"), self._joint_center("ankle_l"))

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
        return math.dist(self._joint_center("walker_knee_r"), self._joint_center("ankle_r"))

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
        shoulder_center = tuple((a + b) / 2 for a, b in zip(left_shoulder, right_shoulder))
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
        return math.dist(self._joint_center("acromial_l"), self._joint_center("acromial_r"))

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
        return math.dist(self._joint_center("acromial_l"), self._joint_center("elbow_l"))

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
        return math.dist(self._joint_center("acromial_r"), self._joint_center("elbow_r"))

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
        return math.dist(self._joint_center("elbow_l"), self._joint_center("radius_hand_l"))

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
        return math.dist(self._joint_center("elbow_r"), self._joint_center("radius_hand_r"))

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

    def set_left_hip_flexionextension(self, degrees: float):
        """Set the left hip flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``hip_flexion_l``. Positive values
        flex the hip (thigh swings forward/upward relative to the pelvis);
        negative values extend it (thigh swings backward). This direction
        is inferred from the coordinate's own range in the bundled model
        (-30 deg of extension vs. 120 deg of flexion), which mirrors the
        real anatomical asymmetry between hip flexion and extension.

        Like every posture setter, this only writes the raw coordinate
        value (see "Setter grezzi sullo stato" in the project README):
        call :meth:`update_state` before reading anything derived from the
        new posture (joint centres, ``com``, muscle lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target hip flexion/extension angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-30, 120]`` (``clamped`` is
            ``True`` in the .osim file): a value outside that interval is
            not rejected, it is silently pulled back to the nearest bound
            instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built, regardless of what the
            base .osim file specifies -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_flexion_l").set_value_degrees(degrees)

    def set_right_hip_flexionextension(self, degrees: float):
        """Set the right hip flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``hip_flexion_r``. Positive values
        flex the hip (thigh swings forward/upward relative to the pelvis);
        negative values extend it (thigh swings backward). This direction
        is inferred from the coordinate's own range in the bundled model
        (-30 deg of extension vs. 120 deg of flexion), which mirrors the
        real anatomical asymmetry between hip flexion and extension.

        Like every posture setter, this only writes the raw coordinate
        value (see "Setter grezzi sullo stato" in the project README):
        call :meth:`update_state` before reading anything derived from the
        new posture (joint centres, ``com``, muscle lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target hip flexion/extension angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-30, 120]`` (``clamped`` is
            ``True`` in the .osim file): a value outside that interval is
            not rejected, it is silently pulled back to the nearest bound
            instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built, regardless of what the
            base .osim file specifies -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_flexion_r").set_value_degrees(degrees)

    def set_left_hip_adduction(self, degrees: float):
        """Set the left hip adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``hip_adduction_l``. Positive values
        adduct the hip (thigh swings toward the midline); negative values
        abduct it (thigh swings away from the midline) -- consistent both
        with the coordinate's own name and with its asymmetric range in
        the bundled model (-50 deg of abduction vs. 30 deg of adduction,
        matching the real ROM asymmetry between the two).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip adduction/abduction angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-50, 30]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_adduction_l").set_value_degrees(degrees)

    def set_right_hip_adduction(self, degrees: float):
        """Set the right hip adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``hip_adduction_r``. Positive values
        adduct the hip (thigh swings toward the midline); negative values
        abduct it (thigh swings away from the midline) -- consistent both
        with the coordinate's own name and with its asymmetric range in
        the bundled model (-50 deg of abduction vs. 30 deg of adduction,
        matching the real ROM asymmetry between the two).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip adduction/abduction angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-50, 30]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_adduction_r").set_value_degrees(degrees)

    def set_left_hip_rotation(self, degrees: float):
        """Set the left hip (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``hip_rotation_l``, rotating the
        thigh about its own long axis. Unlike the flexion/adduction
        coordinates above, neither the coordinate's name nor its range
        pins down a sign here: the range is symmetric (``[-40, 40]``
        degrees) and nothing in this package's code, README, or tests
        states whether positive is internal (medial) or external
        (lateral) rotation. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip rotation angle, in degrees. The underlying OpenSim
            coordinate is clamped to ``[-40, 40]``: a value outside that
            interval is silently pulled back to the nearest bound instead
            of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_rotation_l").set_value_degrees(degrees)

    def set_right_hip_rotation(self, degrees: float):
        """Set the right hip (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``hip_rotation_r``, rotating the
        thigh about its own long axis. Unlike the flexion/adduction
        coordinates above, neither the coordinate's name nor its range
        pins down a sign here: the range is symmetric (``[-40, 40]``
        degrees) and nothing in this package's code, README, or tests
        states whether positive is internal (medial) or external
        (lateral) rotation. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip rotation angle, in degrees. The underlying OpenSim
            coordinate is clamped to ``[-40, 40]``: a value outside that
            interval is silently pulled back to the nearest bound instead
            of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("hip_rotation_r").set_value_degrees(degrees)

    def set_left_knee_flexionextension(self, degrees: float):
        """Set the left knee flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``knee_angle_l``. Positive values
        flex the knee (shank swings backward relative to the thigh); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 140 degrees) -- a knee cannot extend
        past straight in this model, so positive necessarily means
        flexion.

        The patella (:attr:`left_patella`) is kinematically coupled to
        this coordinate through a ``CoordinateCouplerConstraint`` in the
        base model. :meth:`update_state` does **not** resolve that
        coupling (a documented limitation, see the README): call
        :meth:`reinitialize` instead if the patella position needs to
        reflect the new knee angle for something other than ``export()``.

        Parameters
        ----------
        degrees : float
            Target knee flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[0, 140]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("knee_angle_l").set_value_degrees(degrees)

    def set_right_knee_flexionextension(self, degrees: float):
        """Set the right knee flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``knee_angle_r``. Positive values
        flex the knee (shank swings backward relative to the thigh); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 140 degrees) -- a knee cannot extend
        past straight in this model, so positive necessarily means
        flexion.

        The patella (:attr:`right_patella`) is kinematically coupled to
        this coordinate through a ``CoordinateCouplerConstraint`` in the
        base model. :meth:`update_state` does **not** resolve that
        coupling (a documented limitation, see the README): call
        :meth:`reinitialize` instead if the patella position needs to
        reflect the new knee angle for something other than ``export()``.

        Parameters
        ----------
        degrees : float
            Target knee flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[0, 140]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("knee_angle_r").set_value_degrees(degrees)

    def set_left_ankle_flexiondorsiflexion(self, degrees: float):
        """Set the left ankle flexion/dorsiflexion angle, in degrees.

        Writes the OpenSim coordinate ``ankle_angle_l``, which combines
        plantarflexion and dorsiflexion into a single degree of freedom.
        Unlike the hip/knee coordinates above, the sign here is not pinned
        down by this package: the range is symmetric (``[-50, 50]``
        degrees) and the coordinate's own name does not commit to a
        direction. This package's example notebook
        (``example_usage.ipynb``) labels increasing raw values of this
        coordinate as increasing "Plantar Flexion", which suggests
        positive = plantarflexion (foot points away from the shin) and
        negative = dorsiflexion (foot points toward the shin) -- but that
        is example code, not a test assertion, so treat the direction as
        unverified and check the base model documentation if it matters.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target ankle flexion/dorsiflexion angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-50, 50]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("ankle_angle_l").set_value_degrees(degrees)

    def set_right_ankle_flexiondorsiflexion(self, degrees: float):
        """Set the right ankle flexion/dorsiflexion angle, in degrees.

        Writes the OpenSim coordinate ``ankle_angle_r``, which combines
        plantarflexion and dorsiflexion into a single degree of freedom.
        Unlike the hip/knee coordinates above, the sign here is not pinned
        down by this package: the range is symmetric (``[-50, 50]``
        degrees) and the coordinate's own name does not commit to a
        direction. This package's example notebook
        (``example_usage.ipynb``) labels increasing raw values of this
        coordinate as increasing "Plantar Flexion", which suggests
        positive = plantarflexion (foot points away from the shin) and
        negative = dorsiflexion (foot points toward the shin) -- but that
        is example code, not a test assertion, so treat the direction as
        unverified and check the base model documentation if it matters.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target ankle flexion/dorsiflexion angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-50, 50]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("ankle_angle_r").set_value_degrees(degrees)

    def set_left_subtalar_inversion(self, degrees: float):
        """Set the left subtalar inversion/eversion angle, in degrees.

        Writes the OpenSim coordinate ``subtalar_angle_l``. Positive
        values invert the foot (sole turns to face the body's midline);
        negative values evert it (sole turns outward) -- per the
        coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates`` and
        the README's "Modificare la postura" section), overriding whatever
        the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target subtalar inversion/eversion angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-35, 35]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("subtalar_angle_l").set_locked(True)``).
        """
        self.coordinate("subtalar_angle_l").set_value_degrees(degrees)

    def set_right_subtalar_inversion(self, degrees: float):
        """Set the right subtalar inversion/eversion angle, in degrees.

        Writes the OpenSim coordinate ``subtalar_angle_r``. Positive
        values invert the foot (sole turns to face the body's midline);
        negative values evert it (sole turns outward) -- per the
        coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates`` and
        the README's "Modificare la postura" section), overriding whatever
        the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target subtalar inversion/eversion angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-35, 35]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("subtalar_angle_r").set_locked(True)``).
        """
        self.coordinate("subtalar_angle_r").set_value_degrees(degrees)

    def set_left_mtp_flexion(self, degrees: float):
        """Set the left metatarsophalangeal (toe) flexion angle, in degrees.

        Writes the OpenSim coordinate ``mtp_angle_l``. Positive values
        flex the toes (curl downward/plantarward); negative values extend
        them (dorsiflexion, toes point up, as in late stance/toe-off) --
        per the coordinate's own name and its asymmetric range in the
        bundled model (45 deg of extension vs. 30 deg of flexion), which
        matches the real MTP joint's larger dorsiflexion ROM.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target MTP flexion/extension angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-45, 30]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction.
        """
        self.coordinate("mtp_angle_l").set_value_degrees(degrees)

    def set_right_mtp_flexion(self, degrees: float):
        """Set the right metatarsophalangeal (toe) flexion angle, in degrees.

        Writes the OpenSim coordinate ``mtp_angle_r``. Positive values
        flex the toes (curl downward/plantarward); negative values extend
        them (dorsiflexion, toes point up, as in late stance/toe-off) --
        per the coordinate's own name and its asymmetric range in the
        bundled model (45 deg of extension vs. 30 deg of flexion), which
        matches the real MTP joint's larger dorsiflexion ROM.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target MTP flexion/extension angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-45, 30]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (see
            ``coordinate("mtp_angle_r").set_locked(True)``).
        """
        self.coordinate("mtp_angle_r").set_value_degrees(degrees)

    def set_left_shoulder_flexion(self, degrees: float):
        """Set the left shoulder flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``arm_flex_l``. Positive values flex
        the shoulder (arm swings forward and up, toward overhead);
        negative values extend it (arm swings backward) -- per the
        coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture (joint centres, ``com``, muscle
        lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target shoulder flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-90, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_flex_l").set_value_degrees(degrees)

    def set_right_shoulder_flexion(self, degrees: float):
        """Set the right shoulder flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``arm_flex_r``. Positive values flex
        the shoulder (arm swings forward and up, toward overhead);
        negative values extend it (arm swings backward) -- per the
        coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-90, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_flex_r").set_value_degrees(degrees)

    def set_left_shoulder_adduction(self, degrees: float):
        """Set the left shoulder adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``arm_add_l``. Positive values
        adduct the shoulder (arm swings toward/across the body); negative
        values abduct it (arm swings away from the body, overhead) -- per
        the coordinate's own name and its asymmetric range in the bundled
        model (120 deg of abduction vs. 90 deg of adduction), matching the
        real ROM asymmetry between the two (abduction reaches much further
        than adduction past the midline).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder adduction/abduction angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-120, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_add_l").set_value_degrees(degrees)

    def set_right_shoulder_adduction(self, degrees: float):
        """Set the right shoulder adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``arm_add_r``. Positive values
        adduct the shoulder (arm swings toward/across the body); negative
        values abduct it (arm swings away from the body, overhead) -- per
        the coordinate's own name and its asymmetric range in the bundled
        model (120 deg of abduction vs. 90 deg of adduction), matching the
        real ROM asymmetry between the two (abduction reaches much further
        than adduction past the midline).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder adduction/abduction angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-120, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_add_r").set_value_degrees(degrees)

    def set_left_shoulder_rotation(self, degrees: float):
        """Set the left shoulder (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``arm_rot_l``, rotating the upper
        arm about its own long axis. Neither the coordinate's name nor its
        range pins down a sign here: the range is symmetric (``[-90, 90]``
        degrees) and nothing in this package's code, README, or tests
        states whether positive is internal (medial) or external
        (lateral) rotation. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder rotation angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-90, 90]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_rot_l").set_value_degrees(degrees)

    def set_right_shoulder_rotation(self, degrees: float):
        """Set the right shoulder (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``arm_rot_r``, rotating the upper
        arm about its own long axis. Neither the coordinate's name nor its
        range pins down a sign here: the range is symmetric (``[-90, 90]``
        degrees) and nothing in this package's code, README, or tests
        states whether positive is internal (medial) or external
        (lateral) rotation. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder rotation angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-90, 90]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("arm_rot_r").set_value_degrees(degrees)

    def set_left_elbow_flexion(self, degrees: float):
        """Set the left elbow flexion angle, in degrees.

        Writes the OpenSim coordinate ``elbow_flex_l``. Positive values
        flex the elbow (forearm swings toward the upper arm); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 150 degrees) -- an elbow cannot
        extend past straight in this model, so positive necessarily means
        flexion.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target elbow flexion angle, in degrees. The underlying OpenSim
            coordinate is clamped to ``[0, 150]``: a value outside that
            interval is silently pulled back to the nearest bound instead
            of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("elbow_flex_l").set_value_degrees(degrees)

    def set_right_elbow_flexion(self, degrees: float):
        """Set the right elbow flexion angle, in degrees.

        Writes the OpenSim coordinate ``elbow_flex_r``. Positive values
        flex the elbow (forearm swings toward the upper arm); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 150 degrees) -- an elbow cannot
        extend past straight in this model, so positive necessarily means
        flexion.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target elbow flexion angle, in degrees. The underlying OpenSim
            coordinate is clamped to ``[0, 150]``: a value outside that
            interval is silently pulled back to the nearest bound instead
            of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("elbow_flex_r").set_value_degrees(degrees)

    def set_left_wrist_flexion(self, degrees: float):
        """Set the left wrist flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``wrist_flex_l``. Positive values
        flex the wrist (palmar flexion, palm moves toward the forearm);
        negative values extend it (dorsiflexion, back of the hand moves
        toward the forearm) -- per the coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-70, 70]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_flex_l").set_locked(True)``).
        """
        self.coordinate("wrist_flex_l").set_value_degrees(degrees)

    def set_right_wrist_flexion(self, degrees: float):
        """Set the right wrist flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``wrist_flex_r``. Positive values
        flex the wrist (palmar flexion, palm moves toward the forearm);
        negative values extend it (dorsiflexion, back of the hand moves
        toward the forearm) -- per the coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist flexion/extension angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-70, 70]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_flex_r").set_locked(True)``).
        """
        self.coordinate("wrist_flex_r").set_value_degrees(degrees)

    def set_left_wrist_deviation(self, degrees: float):
        """Set the left wrist radial/ulnar deviation angle, in degrees.

        Writes the OpenSim coordinate ``wrist_dev_l``, which deviates the
        hand sideways relative to the forearm. Neither the coordinate's
        name nor its range reliably pins down a sign here: the range is
        asymmetric in the bundled model (``[-25, 35]`` degrees), but
        nothing in this package's code, README, or tests states which
        direction (radial, toward the thumb, vs. ulnar, toward the little
        finger) is positive. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist deviation angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-25, 35]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_dev_l").set_locked(True)``).
        """
        self.coordinate("wrist_dev_l").set_value_degrees(degrees)

    def set_right_wrist_deviation(self, degrees: float):
        """Set the right wrist radial/ulnar deviation angle, in degrees.

        Writes the OpenSim coordinate ``wrist_dev_r``, which deviates the
        hand sideways relative to the forearm. Neither the coordinate's
        name nor its range reliably pins down a sign here: the range is
        asymmetric in the bundled model (``[-25, 35]`` degrees), but
        nothing in this package's code, README, or tests states which
        direction (radial, toward the thumb, vs. ulnar, toward the little
        finger) is positive. Treat the sign as unverified and check the
        base Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``OpenSimModel._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist deviation angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-25, 35]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_dev_r").set_locked(True)``).
        """
        self.coordinate("wrist_dev_r").set_value_degrees(degrees)

    def set_left_forearm_pronation(self, degrees: float):
        """Set the left forearm pronation/supination angle, in degrees.

        Writes the OpenSim coordinate ``pro_sup_l``. Positive values
        pronate the forearm (palm turns to face backward/downward); the
        coordinate's range starts exactly at 0 degrees and only goes
        positive (up to approximately 120 degrees), consistent with 0
        being the fully supinated reference posture and positive values
        moving toward pronation -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target forearm pronation/supination angle, in degrees. The
            underlying OpenSim coordinate is clamped to approximately
            ``[0, 119.75]`` (stored as ``2.09`` radians): a value outside
            that interval is silently pulled back to the nearest bound
            instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("pro_sup_l").set_value_degrees(degrees)

    def set_right_forearm_pronation(self, degrees: float):
        """Set the right forearm pronation/supination angle, in degrees.

        Writes the OpenSim coordinate ``pro_sup_r``. Positive values
        pronate the forearm (palm turns to face backward/downward); the
        coordinate's range starts exactly at 0 degrees and only goes
        positive (up to approximately 120 degrees), consistent with 0
        being the fully supinated reference posture and positive values
        moving toward pronation -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target forearm pronation/supination angle, in degrees. The
            underlying OpenSim coordinate is clamped to approximately
            ``[0, 119.75]`` (stored as ``2.09`` radians): a value outside
            that interval is silently pulled back to the nearest bound
            instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("pro_sup_r").set_value_degrees(degrees)

    def set_lumbar_extension(self, degrees: float):
        """Set the lumbar extension/flexion angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_extension``. This is the
        only lumbar coordinate without a left/right pair (the trunk has a
        single lumbar joint in this model). Positive values extend the
        trunk (leans backward); negative values flex it (leans forward)
        -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture (joint centres, ``com``, muscle
        lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target lumbar extension/flexion angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-90, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("lumbar_extension").set_value_degrees(degrees)

    def set_lumbar_bending(self, degrees: float):
        """Set the lumbar lateral bending angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_bending`` (no left/right
        pair: the trunk has a single lumbar joint in this model). This
        bends the trunk sideways; the model does not label which sign
        bends toward the left vs. the right, and the range is symmetric
        (``[-90, 90]`` degrees) so it gives no hint either. Treat the sign
        as unverified and check the base Rajagopal-Lai-Uhlrich model
        documentation directly if the direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target lumbar lateral bending angle, in degrees. The
            underlying OpenSim coordinate is clamped to ``[-90, 90]``: a
            value outside that interval is silently pulled back to the
            nearest bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("lumbar_bending").set_value_degrees(degrees)

    def set_lumbar_rotation(self, degrees: float):
        """Set the lumbar (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_rotation`` (no left/right
        pair: the trunk has a single lumbar joint in this model), rotating
        the trunk about its own long axis. The model does not label which
        sign rotates the trunk toward the left vs. the right, and the
        range is symmetric (``[-90, 90]`` degrees) so it gives no hint
        either. Treat the sign as unverified and check the base
        Rajagopal-Lai-Uhlrich model documentation directly if the
        direction matters for your use case.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target lumbar rotation angle, in degrees. The underlying
            OpenSim coordinate is clamped to ``[-90, 90]``: a value
            outside that interval is silently pulled back to the nearest
            bound instead of raising.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``OpenSimModel._unlock_coordinates``).
        """
        self.coordinate("lumbar_rotation").set_value_degrees(degrees)

    @property
    def left_hip_flexionextension(self) -> float:
        """Return the current left hip flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``hip_flexion_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_flexionextension` or any other
        writer of this coordinate). Positive values mean the hip is flexed
        (thigh forward/upward relative to the pelvis); negative values
        mean it is extended (thigh backward). The underlying OpenSim
        coordinate is clamped to ``[-30, 120]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current left hip flexion/extension angle, in degrees.
        """
        return self.coordinate("hip_flexion_l").value_degrees

    @property
    def right_hip_flexionextension(self) -> float:
        """Return the current right hip flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``hip_flexion_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_flexionextension` or any other
        writer of this coordinate). Positive values mean the hip is flexed
        (thigh forward/upward relative to the pelvis); negative values
        mean it is extended (thigh backward). The underlying OpenSim
        coordinate is clamped to ``[-30, 120]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current right hip flexion/extension angle, in degrees.
        """
        return self.coordinate("hip_flexion_r").value_degrees

    @property
    def left_hip_adduction(self) -> float:
        """Return the current left hip adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``hip_adduction_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_adduction` or any other writer of
        this coordinate). Positive values mean the hip is adducted (thigh
        toward the midline); negative values mean it is abducted (thigh
        away from the midline). The underlying OpenSim coordinate is
        clamped to ``[-50, 30]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current left hip adduction/abduction angle, in degrees.
        """
        return self.coordinate("hip_adduction_l").value_degrees

    @property
    def right_hip_adduction(self) -> float:
        """Return the current right hip adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``hip_adduction_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_adduction` or any other writer of
        this coordinate). Positive values mean the hip is adducted (thigh
        toward the midline); negative values mean it is abducted (thigh
        away from the midline). The underlying OpenSim coordinate is
        clamped to ``[-50, 30]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current right hip adduction/abduction angle, in degrees.
        """
        return self.coordinate("hip_adduction_r").value_degrees

    @property
    def left_hip_rotation(self) -> float:
        """Return the current left hip (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``hip_rotation_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_rotation` or any other writer of
        this coordinate). This is the thigh's rotation about its own long
        axis; whether positive is internal or external rotation is not
        pinned down by this package's code, README, or tests (see
        :meth:`set_left_hip_rotation`). The underlying OpenSim coordinate
        is clamped to ``[-40, 40]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current left hip rotation angle, in degrees.
        """
        return self.coordinate("hip_rotation_l").value_degrees

    @property
    def right_hip_rotation(self) -> float:
        """Return the current right hip (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``hip_rotation_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_rotation` or any other writer of
        this coordinate). This is the thigh's rotation about its own long
        axis; whether positive is internal or external rotation is not
        pinned down by this package's code, README, or tests (see
        :meth:`set_right_hip_rotation`). The underlying OpenSim coordinate
        is clamped to ``[-40, 40]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current right hip rotation angle, in degrees.
        """
        return self.coordinate("hip_rotation_r").value_degrees

    @property
    def left_knee_flexionextension(self) -> float:
        """Return the current left knee flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``knee_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_knee_flexionextension` or any other
        writer of this coordinate). Positive values mean the knee is
        flexed (shank swung backward relative to the thigh); the
        coordinate cannot go below 0 degrees (full extension) in this
        model. The underlying OpenSim coordinate is clamped to
        ``[0, 140]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it. Note that the coupled patella position
        (:attr:`left_patella`) does not automatically reflect a knee angle
        written mid-edit until :meth:`reinitialize` is called (see
        :meth:`set_left_knee_flexionextension`).

        Returns
        -------
        float
            Current left knee flexion/extension angle, in degrees.
        """
        return self.coordinate("knee_angle_l").value_degrees

    @property
    def right_knee_flexionextension(self) -> float:
        """Return the current right knee flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``knee_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_knee_flexionextension` or any other
        writer of this coordinate). Positive values mean the knee is
        flexed (shank swung backward relative to the thigh); the
        coordinate cannot go below 0 degrees (full extension) in this
        model. The underlying OpenSim coordinate is clamped to
        ``[0, 140]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it. Note that the coupled patella position
        (:attr:`right_patella`) does not automatically reflect a knee
        angle written mid-edit until :meth:`reinitialize` is called (see
        :meth:`set_right_knee_flexionextension`).

        Returns
        -------
        float
            Current right knee flexion/extension angle, in degrees.
        """
        return self.coordinate("knee_angle_r").value_degrees

    @property
    def left_ankle_flexiondorsiflexion(self) -> float:
        """Return the current left ankle flexion/dorsiflexion angle, in degrees.

        Reads the OpenSim coordinate ``ankle_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_ankle_flexiondorsiflexion` or any other
        writer of this coordinate). This combines plantarflexion and
        dorsiflexion into a single degree of freedom; which sign is which
        is not firmly pinned down by this package (see
        :meth:`set_left_ankle_flexiondorsiflexion` for the available
        evidence). The underlying OpenSim coordinate is clamped to
        ``[-50, 50]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it.

        Returns
        -------
        float
            Current left ankle flexion/dorsiflexion angle, in degrees.
        """
        return self.coordinate("ankle_angle_l").value_degrees

    @property
    def right_ankle_flexiondorsiflexion(self) -> float:
        """Return the current right ankle flexion/dorsiflexion angle, in degrees.

        Reads the OpenSim coordinate ``ankle_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_ankle_flexiondorsiflexion` or any
        other writer of this coordinate). This combines plantarflexion and
        dorsiflexion into a single degree of freedom; which sign is which
        is not firmly pinned down by this package (see
        :meth:`set_right_ankle_flexiondorsiflexion` for the available
        evidence). The underlying OpenSim coordinate is clamped to
        ``[-50, 50]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it.

        Returns
        -------
        float
            Current right ankle flexion/dorsiflexion angle, in degrees.
        """
        return self.coordinate("ankle_angle_r").value_degrees

    @property
    def left_subtalar_inversion(self) -> float:
        """Return the current left subtalar inversion/eversion angle, in degrees.

        Reads the OpenSim coordinate ``subtalar_angle_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_subtalar_inversion` or any other writer
        of this coordinate). Positive values mean the foot is inverted
        (sole turned toward the midline); negative values mean it is
        everted (sole turned outward). The underlying OpenSim coordinate
        is clamped to ``[-35, 35]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it. This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_subtalar_inversion`).

        Returns
        -------
        float
            Current left subtalar inversion/eversion angle, in degrees.
        """
        return self.coordinate("subtalar_angle_l").value_degrees

    @property
    def right_subtalar_inversion(self) -> float:
        """Return the current right subtalar inversion/eversion angle, in degrees.

        Reads the OpenSim coordinate ``subtalar_angle_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_subtalar_inversion` or any other
        writer of this coordinate). Positive values mean the foot is
        inverted (sole turned toward the midline); negative values mean it
        is everted (sole turned outward). The underlying OpenSim
        coordinate is clamped to ``[-35, 35]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it. This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance (see :meth:`set_right_subtalar_inversion`).

        Returns
        -------
        float
            Current right subtalar inversion/eversion angle, in degrees.
        """
        return self.coordinate("subtalar_angle_r").value_degrees

    @property
    def left_mtp_flexion(self) -> float:
        """Return the current left MTP (toe) flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``mtp_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_mtp_flexion` or any other writer of
        this coordinate). Positive values mean the toes are flexed (curled
        plantarward/downward); negative values mean they are extended
        (dorsiflexed, pointing up). The underlying OpenSim coordinate is
        clamped to ``[-45, 30]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it. This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_mtp_flexion`).

        Returns
        -------
        float
            Current left MTP flexion/extension angle, in degrees.
        """
        return self.coordinate("mtp_angle_l").value_degrees

    @property
    def right_mtp_flexion(self) -> float:
        """Return the current right MTP (toe) flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``mtp_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_mtp_flexion` or any other writer of
        this coordinate). Positive values mean the toes are flexed (curled
        plantarward/downward); negative values mean they are extended
        (dorsiflexed, pointing up). The underlying OpenSim coordinate is
        clamped to ``[-45, 30]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it. This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_right_mtp_flexion`).

        Returns
        -------
        float
            Current right MTP flexion/extension angle, in degrees.
        """
        return self.coordinate("mtp_angle_r").value_degrees

    @property
    def left_shoulder_flexion(self) -> float:
        """Return the current left shoulder flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``arm_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_flexion` or any other writer
        of this coordinate). Positive values mean the shoulder is flexed
        (arm forward/up, toward overhead); negative values mean it is
        extended (arm backward). The underlying OpenSim coordinate is
        clamped to ``[-90, 90]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current left shoulder flexion/extension angle, in degrees.
        """
        return self.coordinate("arm_flex_l").value_degrees

    @property
    def right_shoulder_flexion(self) -> float:
        """Return the current right shoulder flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``arm_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_flexion` or any other writer
        of this coordinate). Positive values mean the shoulder is flexed
        (arm forward/up, toward overhead); negative values mean it is
        extended (arm backward). The underlying OpenSim coordinate is
        clamped to ``[-90, 90]`` degrees, so this is always the value
        actually read back within that range, even if a caller requested
        something outside it.

        Returns
        -------
        float
            Current right shoulder flexion/extension angle, in degrees.
        """
        return self.coordinate("arm_flex_r").value_degrees

    @property
    def left_shoulder_adduction(self) -> float:
        """Return the current left shoulder adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``arm_add_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_adduction` or any other writer
        of this coordinate). Positive values mean the shoulder is adducted
        (arm toward/across the body); negative values mean it is abducted
        (arm away from the body, overhead). The underlying OpenSim
        coordinate is clamped to ``[-120, 90]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current left shoulder adduction/abduction angle, in degrees.
        """
        return self.coordinate("arm_add_l").value_degrees

    @property
    def right_shoulder_adduction(self) -> float:
        """Return the current right shoulder adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``arm_add_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_adduction` or any other
        writer of this coordinate). Positive values mean the shoulder is
        adducted (arm toward/across the body); negative values mean it is
        abducted (arm away from the body, overhead). The underlying
        OpenSim coordinate is clamped to ``[-120, 90]`` degrees, so this
        is always the value actually read back within that range, even if
        a caller requested something outside it.

        Returns
        -------
        float
            Current right shoulder adduction/abduction angle, in degrees.
        """
        return self.coordinate("arm_add_r").value_degrees

    @property
    def left_shoulder_rotation(self) -> float:
        """Return the current left shoulder (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``arm_rot_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_rotation` or any other writer
        of this coordinate). This is the upper arm's rotation about its
        own long axis; whether positive is internal or external rotation
        is not pinned down by this package's code, README, or tests (see
        :meth:`set_left_shoulder_rotation`). The underlying OpenSim
        coordinate is clamped to ``[-90, 90]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current left shoulder rotation angle, in degrees.
        """
        return self.coordinate("arm_rot_l").value_degrees

    @property
    def right_shoulder_rotation(self) -> float:
        """Return the current right shoulder (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``arm_rot_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_rotation` or any other writer
        of this coordinate). This is the upper arm's rotation about its
        own long axis; whether positive is internal or external rotation
        is not pinned down by this package's code, README, or tests (see
        :meth:`set_right_shoulder_rotation`). The underlying OpenSim
        coordinate is clamped to ``[-90, 90]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current right shoulder rotation angle, in degrees.
        """
        return self.coordinate("arm_rot_r").value_degrees

    @property
    def left_elbow_flexion(self) -> float:
        """Return the current left elbow flexion angle, in degrees.

        Reads the OpenSim coordinate ``elbow_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_elbow_flexion` or any other writer of
        this coordinate). Positive values mean the elbow is flexed
        (forearm toward the upper arm); the coordinate cannot go below 0
        degrees (full extension) in this model. The underlying OpenSim
        coordinate is clamped to ``[0, 150]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current left elbow flexion angle, in degrees.
        """
        return self.coordinate("elbow_flex_l").value_degrees

    @property
    def right_elbow_flexion(self) -> float:
        """Return the current right elbow flexion angle, in degrees.

        Reads the OpenSim coordinate ``elbow_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_elbow_flexion` or any other writer of
        this coordinate). Positive values mean the elbow is flexed
        (forearm toward the upper arm); the coordinate cannot go below 0
        degrees (full extension) in this model. The underlying OpenSim
        coordinate is clamped to ``[0, 150]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current right elbow flexion angle, in degrees.
        """
        return self.coordinate("elbow_flex_r").value_degrees

    @property
    def left_wrist_flexion(self) -> float:
        """Return the current left wrist flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``wrist_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_wrist_flexion` or any other writer of
        this coordinate). Positive values mean the wrist is flexed
        (palmar flexion); negative values mean it is extended
        (dorsiflexion). The underlying OpenSim coordinate is clamped to
        ``[-70, 70]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it. This coordinate ships locked in the base .osim file
        but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_wrist_flexion`).

        Returns
        -------
        float
            Current left wrist flexion/extension angle, in degrees.
        """
        return self.coordinate("wrist_flex_l").value_degrees

    @property
    def right_wrist_flexion(self) -> float:
        """Return the current right wrist flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``wrist_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_wrist_flexion` or any other writer of
        this coordinate). Positive values mean the wrist is flexed
        (palmar flexion); negative values mean it is extended
        (dorsiflexion). The underlying OpenSim coordinate is clamped to
        ``[-70, 70]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it. This coordinate ships locked in the base .osim file
        but is force-unlocked for every ``User`` instance (see
        :meth:`set_right_wrist_flexion`).

        Returns
        -------
        float
            Current right wrist flexion/extension angle, in degrees.
        """
        return self.coordinate("wrist_flex_r").value_degrees

    @property
    def left_wrist_deviation(self) -> float:
        """Return the current left wrist radial/ulnar deviation angle, in degrees.

        Reads the OpenSim coordinate ``wrist_dev_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_wrist_deviation` or any other writer of
        this coordinate). This deviates the hand sideways relative to the
        forearm; whether positive is radial or ulnar deviation is not
        pinned down by this package's code, README, or tests (see
        :meth:`set_left_wrist_deviation`). The underlying OpenSim
        coordinate is clamped to ``[-25, 35]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it. This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance.

        Returns
        -------
        float
            Current left wrist deviation angle, in degrees.
        """
        return self.coordinate("wrist_dev_l").value_degrees

    @property
    def right_wrist_deviation(self) -> float:
        """Return the current right wrist radial/ulnar deviation angle, in degrees.

        Reads the OpenSim coordinate ``wrist_dev_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_wrist_deviation` or any other writer
        of this coordinate). This deviates the hand sideways relative to
        the forearm; whether positive is radial or ulnar deviation is not
        pinned down by this package's code, README, or tests (see
        :meth:`set_right_wrist_deviation`). The underlying OpenSim
        coordinate is clamped to ``[-25, 35]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it. This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance.

        Returns
        -------
        float
            Current right wrist deviation angle, in degrees.
        """
        return self.coordinate("wrist_dev_r").value_degrees

    @property
    def left_forearm_pronation(self) -> float:
        """Return the current left forearm pronation/supination angle, in degrees.

        Reads the OpenSim coordinate ``pro_sup_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_forearm_pronation` or any other writer
        of this coordinate). Positive values mean the forearm is pronated
        (palm facing backward/downward); the coordinate cannot go below 0
        degrees (the fully supinated reference posture) in this model.
        The underlying OpenSim coordinate is clamped to approximately
        ``[0, 119.75]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it.

        Returns
        -------
        float
            Current left forearm pronation/supination angle, in degrees.
        """
        return self.coordinate("pro_sup_l").value_degrees

    @property
    def right_forearm_pronation(self) -> float:
        """Return the current right forearm pronation/supination angle, in degrees.

        Reads the OpenSim coordinate ``pro_sup_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_forearm_pronation` or any other writer
        of this coordinate). Positive values mean the forearm is pronated
        (palm facing backward/downward); the coordinate cannot go below 0
        degrees (the fully supinated reference posture) in this model.
        The underlying OpenSim coordinate is clamped to approximately
        ``[0, 119.75]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it.

        Returns
        -------
        float
            Current right forearm pronation/supination angle, in degrees.
        """
        return self.coordinate("pro_sup_r").value_degrees

    @property
    def lumbar_extension(self) -> float:
        """Return the current lumbar extension/flexion angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_extension`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_extension` or any other writer of
        this coordinate). There is no left/right pair: the trunk has a
        single lumbar joint in this model. Positive values mean the trunk
        is extended (leaning backward); negative values mean it is flexed
        (leaning forward). The underlying OpenSim coordinate is clamped to
        ``[-90, 90]`` degrees, so this is always the value actually read
        back within that range, even if a caller requested something
        outside it.

        Returns
        -------
        float
            Current lumbar extension/flexion angle, in degrees.
        """
        return self.coordinate("lumbar_extension").value_degrees

    @property
    def lumbar_bending(self) -> float:
        """Return the current lumbar lateral bending angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_bending`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_bending` or any other writer of this
        coordinate). There is no left/right pair: the trunk has a single
        lumbar joint in this model. Whether positive bends the trunk
        toward the left or the right is not pinned down by this package's
        code, README, or tests (see :meth:`set_lumbar_bending`). The
        underlying OpenSim coordinate is clamped to ``[-90, 90]`` degrees,
        so this is always the value actually read back within that range,
        even if a caller requested something outside it.

        Returns
        -------
        float
            Current lumbar lateral bending angle, in degrees.
        """
        return self.coordinate("lumbar_bending").value_degrees

    @property
    def lumbar_rotation(self) -> float:
        """Return the current lumbar (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_rotation`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_rotation` or any other writer of this
        coordinate). There is no left/right pair: the trunk has a single
        lumbar joint in this model. This is the trunk's rotation about its
        own long axis; whether positive rotates it toward the left or the
        right is not pinned down by this package's code, README, or tests
        (see :meth:`set_lumbar_rotation`). The underlying OpenSim
        coordinate is clamped to ``[-90, 90]`` degrees, so this is always
        the value actually read back within that range, even if a caller
        requested something outside it.

        Returns
        -------
        float
            Current lumbar rotation angle, in degrees.
        """
        return self.coordinate("lumbar_rotation").value_degrees
