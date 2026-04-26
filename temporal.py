"""Temporal distribution loading and application to a 2D depth field.

The CSV (e.g., HydroCad rainfall distributions) provides cumulative fractions
of total storm depth at 0.1-hr resolution from t=0 to t=24, for several
named distributions: MSE 1-6, Type I, Type IA, Type II (6/12/24-hr), Type III
(6/12/24-hr). Headers are multi-line strings within quoted CSV cells like
``"Type II\\n24-hr\\n(depth)"`` so we normalize whitespace on load and match
by case-insensitive substring.

For HMS gridded input, we want incremental precip per timestep, not cumulative,
so ``apply_temporal_distribution`` interpolates the cumulative curve to the
output dt and takes np.diff to get per-timestep increments. The increments are
then broadcast against the 2D depth field to produce a 3D (time, y, x) array,
wrapped in an xarray Dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _normalize_header(h: str) -> str:
    return " ".join(str(h).replace("\r", "").split())


def load_temporal_distribution(
    csv_path: str | Path,
    name: str = "Type II 24-hr",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cumulative temporal distribution by name.

    The CSV's first column is time (hr); subsequent columns are cumulative
    fractions for various distributions. ``name`` is matched as a
    case-insensitive substring against normalized headers (newlines and
    repeated whitespace collapsed to single spaces).

    Distributions that span less than 24 hours (e.g., the 6-hr and 12-hr
    variants) carry NaN past their end time — those rows are dropped before
    return.
    """
    df = pd.read_csv(csv_path)
    df.columns = [_normalize_header(c) for c in df.columns]
    target = name.lower()
    matches = [c for c in df.columns if target in c.lower()]
    if not matches:
        raise KeyError(
            f"No column matching {name!r}. Available: {df.columns.tolist()}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"Multiple columns match {name!r}: {matches}. Be more specific."
        )
    col = matches[0]
    time_col = df.columns[0]
    sub = df[[time_col, col]].dropna()
    return sub[time_col].to_numpy(dtype=float), sub[col].to_numpy(dtype=float)


def _incremental_fractions(
    cum_time_hr: np.ndarray,
    cum_fraction: np.ndarray,
    output_dt_minutes: float,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Resample a cumulative curve to the output timestep and return increments.

    Returns (t_out_hr, time_index_placeholder, incr_fraction). The
    placeholder is left as a 1D ndarray so callers can build their own
    DatetimeIndex with their preferred start_time.
    """
    if cum_time_hr[0] != 0.0 or abs(cum_fraction[0]) > 1e-9:
        raise ValueError("Cumulative distribution must start at (0, 0).")
    end_hr = float(cum_time_hr[-1])
    dt_hr = output_dt_minutes / 60.0
    n_steps = int(round(end_hr / dt_hr)) + 1
    t_out_hr = np.linspace(0.0, end_hr, n_steps)
    cum_out = np.interp(t_out_hr, cum_time_hr, cum_fraction)
    incr = np.diff(cum_out, prepend=0.0)
    return t_out_hr, pd.DatetimeIndex([]), incr


def build_subbasin_hyetographs(
    subbasin_means: pd.Series,
    cum_time_hr: np.ndarray,
    cum_fraction: np.ndarray,
    output_dt_minutes: float = 5.0,
    start_time: pd.Timestamp | str = "2000-01-01 00:00:00",
) -> pd.DataFrame:
    """Build per-subbasin hyetographs from per-subbasin storm-total depths.

    Because the temporal distribution is uniform across the storm extent, the
    hyetograph for any subbasin equals (cumulative-curve increments) times the
    subbasin's storm-total mean depth. No per-timestep spatial averaging needed.

    Args:
        subbasin_means: pd.Series indexed by subbasin name, values are
            storm-total mean depths in inches (output of
            ``basin.compute_subbasin_means``).
        cum_time_hr, cum_fraction: cumulative temporal distribution
            (output of ``load_temporal_distribution``).
        output_dt_minutes: output timestep in minutes.
        start_time: anchor for the time index.

    Returns:
        DataFrame with one column per subbasin and a DatetimeIndex. Summing
        each column over time recovers that subbasin's storm-total depth.
    """
    _t_out_hr, _, incr = _incremental_fractions(
        cum_time_hr, cum_fraction, output_dt_minutes
    )
    times = pd.date_range(
        start=pd.Timestamp(start_time),
        periods=len(incr),
        freq=f"{output_dt_minutes:g}min",
    )
    data = np.outer(incr, subbasin_means.to_numpy())
    df = pd.DataFrame(data, index=times, columns=subbasin_means.index)
    df.index.name = "time"
    return df


def apply_temporal_distribution(
    depth_field_2d: np.ndarray,
    cum_time_hr: np.ndarray,
    cum_fraction: np.ndarray,
    output_dt_minutes: float = 5.0,
    start_time: pd.Timestamp | str = "2000-01-01 00:00:00",
    x_coord: np.ndarray | None = None,
    y_coord: np.ndarray | None = None,
) -> xr.Dataset:
    """Distribute a 2D storm-total depth field over time.

    Args:
        depth_field_2d: shape (ny, nx). Storm-total depth at each cell, in
            inches. Typically the output of ``hmr52.build_isohyet_field``.
        cum_time_hr: 1D array of cumulative-curve time stamps in hours.
            Must start at 0.0.
        cum_fraction: 1D array, cumulative fraction of total depth at each
            time, monotonically increasing from 0 to 1.
        output_dt_minutes: output timestep in minutes (default 5).
        start_time: anchor for the time coordinate.
        x_coord, y_coord: 1D coordinate arrays for the spatial dims (meters,
            EPSG:5070). Optional; if omitted, integer indices are used.

    Returns:
        xarray.Dataset with variable ``precip`` (time, y, x), incremental
        precip per timestep in the same units as ``depth_field_2d``. Summing
        ``precip`` over time at any cell recovers ``depth_field_2d`` at that
        cell.
    """
    if cum_time_hr[0] != 0.0 or abs(cum_fraction[0]) > 1e-9:
        raise ValueError("Cumulative distribution must start at (0, 0).")

    end_hr = float(cum_time_hr[-1])
    dt_hr = output_dt_minutes / 60.0
    n_steps = int(round(end_hr / dt_hr)) + 1
    t_out_hr = np.linspace(0.0, end_hr, n_steps)
    cum_out = np.interp(t_out_hr, cum_time_hr, cum_fraction)
    incr_fraction = np.diff(cum_out, prepend=0.0)

    precip = incr_fraction[:, None, None] * depth_field_2d[None, :, :]

    times = pd.date_range(
        start=pd.Timestamp(start_time),
        periods=n_steps,
        freq=f"{output_dt_minutes:g}min",
    )
    ny, nx = depth_field_2d.shape
    coords = {
        "time": times,
        "y": y_coord if y_coord is not None else np.arange(ny),
        "x": x_coord if x_coord is not None else np.arange(nx),
    }
    return xr.Dataset(
        data_vars={"precip": (("time", "y", "x"), precip)},
        coords=coords,
        attrs={
            "description": (
                "Incremental precip per timestep, derived from elliptical "
                "depth field x cumulative temporal distribution."
            ),
        },
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    CSV = (
        Path(__file__).parent
        / "data"
        / "HydroCad Rainfall Temporal Distribtions.csv"
    )
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")

    t, frac = load_temporal_distribution(CSV, "Type II 24-hr")
    print(
        f"Type II 24-hr loaded: {len(t)} points, "
        f"t in [{t.min():.1f}, {t.max():.1f}] hr, cum end = {frac[-1]:.4f}"
    )

    ny, nx = 50, 50
    yy, xx = np.indices((ny, nx))
    cy_idx, cx_idx = ny // 2, nx // 2
    depth_field = 10.0 * np.exp(
        -((xx - cx_idx) ** 2 + (yy - cy_idx) ** 2) / 200.0
    )

    ds = apply_temporal_distribution(
        depth_field, t, frac, output_dt_minutes=15.0
    )
    print(f"Dataset dims: {dict(ds.sizes)}")
    print(
        f"precip range: [{ds.precip.min().item():.5f}, "
        f"{ds.precip.max().item():.5f}] in"
    )
    center_total = float(ds.precip[:, cy_idx, cx_idx].sum().item())
    target = float(depth_field[cy_idx, cx_idx])
    print(
        f"sum at center over time = {center_total:.4f} in "
        f"(should match depth_field[c]={target:.4f}, diff={center_total - target:+.6f})"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    t_axis_hr = (np.arange(len(ds.time)) * 15) / 60.0
    ax.plot(t_axis_hr, ds.precip[:, cy_idx, cx_idx].values)
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Incremental precip at center (in / 15-min)")
    ax.set_title("SCS Type II 24-hr applied to centroid (15-min steps)")
    ax.grid(True, alpha=0.3)
    out = Path(__file__).parent / "test_temporal.png"
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
