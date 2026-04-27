"""DSS I/O for subbasin hyetographs.

Writes per-subbasin incremental-precipitation time-series records to a DSS
file using the standard HEC-HMS Specified Hyetograph pathname convention:

    /A-part: project tag /B-part: subbasin /C-part: PRECIP-INC /
     D-part: start date /E-part: time interval /F-part: scenario id /

The B-part keeps the subbasin name as it appears in the HMS basin model
(only the path separator '/' is escaped) so HMS can find each record by
configuring its Specified Hyetograph entries to point at the matching path.
DSS data type is PER-CUM (depth that fell during the interval ending at the
record's timestamp), matching HMS's expectation for incremental precipitation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydsstools.core import TimeSeriesContainer
from pydsstools.heclib.dss.HecDss import Open


def _safe_b_part(name: str) -> str:
    """Escape characters that conflict with DSS pathname syntax."""
    return str(name).strip().replace("/", "_")


def _interval_token(dt_minutes: int) -> str:
    """Map an output dt to its DSS E-part token (5MIN, 15MIN, 1HOUR, ...)."""
    if dt_minutes < 60:
        return f"{dt_minutes}MIN"
    if dt_minutes % 60 == 0:
        h = dt_minutes // 60
        return "1HOUR" if h == 1 else f"{h}HOUR"
    raise ValueError(f"Unsupported dt of {dt_minutes} min — use a multiple of 5/15/30/60.")


def write_subbasin_hyetographs_dss(
    hyeto_df: pd.DataFrame,
    dss_path: str | Path,
    project: str = "",
    scenario_tag: str = "",
    parameter: str = "PRECIP-INC",
    units: str = "INCHES",
    overwrite: bool = False,
) -> list[str]:
    """Write per-subbasin hyetographs to a DSS file.

    Args:
        hyeto_df: DataFrame with a DatetimeIndex and one column per subbasin
            holding incremental precip per timestep (e.g., output of
            ``temporal.build_subbasin_hyetographs``).
        dss_path: target ``.dss`` file (created if missing, appended otherwise).
        project: A-part of the DSS path. Defaults to empty so the resulting
            pathnames match the form HMS references when the precip gages
            are configured against a hyeto DSS file with a blank A-part.
        scenario_tag: F-part. Defaults to empty for the same reason as
            ``project``. When iterating with a fixed HMS Met Model, leave
            both blank and rely on ``overwrite=True`` to refresh the file
            each iteration; HMS will pick up the new data without needing
            its Met Model edited.
        parameter: C-part. ``PRECIP-INC`` is the HMS-recognized name for
            incremental precipitation.
        units: pathname units field (``INCHES`` for in, ``MM`` for mm, etc.).
        overwrite: when True, delete an existing DSS at ``dss_path`` before
            writing — guarantees that no orphan paths from a prior iteration
            remain in the file.

    Returns:
        List of pathnames written.
    """
    if not isinstance(hyeto_df.index, pd.DatetimeIndex):
        raise TypeError("hyeto_df.index must be a DatetimeIndex.")
    if len(hyeto_df) < 2:
        raise ValueError("hyeto_df must have at least two timesteps.")

    dss_path = Path(dss_path)
    if overwrite and dss_path.exists():
        dss_path.unlink()

    dt_min = int(round((hyeto_df.index[1] - hyeto_df.index[0]).total_seconds() / 60.0))
    e_part = _interval_token(dt_min)

    # DSS PER-CUM convention: each value's timestamp is the END of the period it
    # represents. Our hyetograph DataFrame has a leading row at t=0 holding the
    # "period ending at t=0" placeholder (always 0 for an unstarted storm); drop
    # that and shift startDateTime forward by one dt so the first stored value
    # represents the period [t=0, t=dt] ending at t=dt.
    write_df = hyeto_df.iloc[1:]
    first_end = write_df.index[0]
    d_part = first_end.strftime("%d%b%Y").upper()
    start_dt = first_end.strftime("%d%b%Y %H:%M:%S").upper()

    written: list[str] = []
    with Open(str(dss_path)) as dss:
        for col in write_df.columns:
            b_part = _safe_b_part(col)
            pathname = f"/{project}/{b_part}/{parameter}/{d_part}/{e_part}/{scenario_tag}/"
            tsc = TimeSeriesContainer()
            tsc.pathname = pathname
            tsc.startDateTime = start_dt
            tsc.numberValues = len(write_df)
            tsc.units = units
            tsc.type = "PER-CUM"
            tsc.interval = dt_min
            tsc.values = write_df[col].to_numpy(dtype=np.float64)
            dss.put_ts(tsc)
            written.append(pathname)
    return written


if __name__ == "__main__":
    from hms_runner import HMS_HYETO_DSS

    PROJ = Path(__file__).parent
    CSV = PROJ / "output" / "I57" / "100yr_basinC_235deg_20000sqmi_hyeto.csv"
    if not CSV.exists():
        raise SystemExit(
            f"Missing {CSV}. Run isohyet_maker.py first to generate the hyetograph CSV."
        )

    hy = pd.read_csv(CSV, index_col=0, parse_dates=True)
    dt_min = int(round((hy.index[1] - hy.index[0]).total_seconds() / 60.0))
    print(f"Loaded {hy.shape[0]} timesteps x {hy.shape[1]} subbasins, dt={dt_min} min")
    print(f"Subbasin column sample sums (in): "
          f"{hy.sum().head(3).round(3).to_dict()}")

    # Default-blank A and F parts produce pathnames in the form
    # //<subbasin>/PRECIP-INC/<date>/5MIN// — matching the HMS Met Model's
    # configured precip gage references for the auto-hyeto setup. We write
    # directly to the HMS-referenced hyeto.dss so HMS picks up the new precip
    # without any Met Model edits.
    dss_out = HMS_HYETO_DSS
    print(f"\nTarget DSS: {dss_out}")
    if not dss_out.parent.exists():
        raise SystemExit(f"HMS data folder does not exist: {dss_out.parent}")

    paths = write_subbasin_hyetographs_dss(hy, dss_out, overwrite=True)
    print(f"Wrote {len(paths)} records.")
    print("Sample paths:")
    for p in paths[:3]:
        print(f"  {p}")

    # Round-trip read-back to verify a single record. The first row of the
    # source DataFrame is a t=0 placeholder (zero); the writer drops it.
    print("\nRound-trip check on first subbasin:")
    with Open(str(dss_out)) as dss:
        ts = dss.read_ts(paths[0])
        ts_vals = np.array(ts.values, dtype=float)
        original = hy.iloc[1:, 0].to_numpy(dtype=float)
        max_abs_diff = float(np.max(np.abs(ts_vals - original)))
        print(f"  pathname:  {paths[0]}")
        print(f"  n values:  written={len(original)}, read back={len(ts_vals)}")
        print(f"  total in:  written={original.sum():.4f}, read back={ts_vals.sum():.4f}")
        print(f"  max |diff|: {max_abs_diff:.2e}")
