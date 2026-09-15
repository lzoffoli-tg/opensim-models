from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "assets" / "ansur_ref.csv"


@dataclass(frozen=True)
class AnthropometricReference:
    """Anthropometric values resolved at one ANSUR percentile.

    Parameters
    ----------
    gender : str
        Normalized sex code, either ``"M"`` or ``"F"``.
    percentile : float
        Percentile used to calculate all numeric measurements.
    values : dict[str, float]
        Measurement names mapped to values in their documented source units.
    """

    gender: str
    percentile: float
    values: dict[str, float]

    @property
    def height_m(self) -> float:
        """Return stature in metres.

        Returns
        -------
        float
            ANSUR stature in metres.
        """
        return self.values["stature_m"]

    @property
    def height_cm(self) -> float:
        """Return stature in centimetres.

        Returns
        -------
        float
            ANSUR stature in centimetres.
        """
        return self.height_m * 100.0


def _read_raw_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))

    if not rows:
        raise ValueError(f"ANSUR dataset is empty: {path}")

    header = [name.strip() for name in rows[0] if name.strip()]
    if not header or header[0] != "Branch":
        raise ValueError("Unexpected ANSUR header: expected Branch as first column")

    expected = len(header) + 1
    data_rows = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != expected:
            raise ValueError(
                f"Invalid ANSUR row {line_number}: expected {expected} fields, got {len(row)}"
            )
        data_rows.append(row)

    return pd.DataFrame(data_rows, columns=["subject_id", *header])


def load_ansur(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load ANSUR II while repairing its implicit leading subject_id column.

    Linear measurements are kept in their source units (mostly millimetres).
    Only the percentile reference exposes stature in metres, matching OpenSim.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the ANSUR CSV file.

    Returns
    -------
    pandas.DataFrame
        Normalized ANSUR records with an explicit ``subject_id`` column.

    Raises
    ------
    ValueError
        If the file is empty, malformed, or contains unsupported values.
    """
    frame = _read_raw_csv(Path(path))
    frame["Gender"] = (
        frame["Gender"]
        .str.strip()
        .str.upper()
        .map({"MALE": "M", "FEMALE": "F", "M": "M", "F": "F"})
    )
    if frame["Gender"].isna().any():
        raise ValueError("ANSUR contains an unsupported gender value")

    numeric_columns = frame.columns.difference(
        ["Branch", "Component", "Gender", "BMI_class", "Height_class"]
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["stature_m"].isna().any():
        raise ValueError("ANSUR contains invalid stature_m values")
    return frame


def _height_percentile(statures_m: np.ndarray, height_cm: float) -> float:
    height_m = height_cm / 100.0
    minimum, maximum = float(np.min(statures_m)), float(np.max(statures_m))
    if not minimum <= height_m <= maximum:
        raise ValueError(
            f"height={height_cm:g} cm is outside ANSUR range "
            f"[{minimum * 100:.1f}, {maximum * 100:.1f}] cm"
        )
    # Empirical CDF: no subject selection and no extrapolation.
    return float(
        np.searchsorted(np.sort(statures_m), height_m, side="right")
        * 100
        / len(statures_m)
    )


def _percentile_values(frame: pd.DataFrame, percentile: float) -> dict[str, float]:
    values: dict[str, float] = {}
    excluded = {
        "subject_id",
        "Branch",
        "Component",
        "Gender",
        "BMI_class",
        "Height_class",
    }
    for column in frame.columns:
        if column in excluded:
            continue
        numeric = (
            pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        )
        if numeric.size == 0:
            continue
        values[column] = float(np.percentile(numeric, percentile, method="linear"))
    return values


def resolve_reference(
    gender: str,
    height: float | None = None,
    percentile: float = 50.0,
    dataset: str | Path = DEFAULT_DATASET,
) -> AnthropometricReference:
    """Resolve one anthropometric reference using a common ANSUR percentile.

    ``height`` is expressed in centimetres. When present, it determines the
    empirical stature percentile and takes precedence over ``percentile``.
    When absent, ``percentile`` is applied to every numeric ANSUR measurement,
    including stature. Quantiles are calculated directly with NumPy.

    Parameters
    ----------
    gender : str
        Sex code, either ``"M"`` or ``"F"``.
    height : float or None, optional
        Requested stature in centimetres. If provided, its empirical ANSUR
        percentile overrides ``percentile``.
    percentile : float, optional
        Common target percentile when ``height`` is not provided. Must be in
        the inclusive interval ``[0.1, 99.9]``.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR CSV file.

    Returns
    -------
    AnthropometricReference
        Resolved measurements and the effective percentile.

    Raises
    ------
    ValueError
        If the gender, height, percentile, or dataset values are invalid.
    """
    normalized_gender = str(gender).strip().upper()
    if normalized_gender not in {"M", "F"}:
        raise ValueError("gender must be 'M' or 'F'")
    if height is not None and (not np.isfinite(height) or height <= 0):
        raise ValueError("height must be a positive finite value in centimetres")
    if not np.isfinite(percentile) or not 0.1 <= percentile <= 99.9:
        raise ValueError("percentile must be between 0.1 and 99.9")

    frame = load_ansur(dataset)
    gender_frame = frame.loc[frame["Gender"] == normalized_gender]
    if gender_frame.empty:
        raise ValueError(f"ANSUR has no records for gender {normalized_gender!r}")

    target_percentile = percentile
    if height is not None:
        target_percentile = _height_percentile(
            gender_frame["stature_m"].to_numpy(dtype=float),
            float(height),
        )
        target_percentile = min(99.9, max(0.1, target_percentile))

    values = _percentile_values(gender_frame, target_percentile)
    return AnthropometricReference(
        normalized_gender,
        target_percentile,
        values,
    )
