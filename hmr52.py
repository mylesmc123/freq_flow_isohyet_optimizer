"""HMR52 spatial pattern: depth-area-duration curve loader, ellipse polygon
construction, and rasterization of the elliptical depth field.

The DAD curve is loaded from a CSV (see data/hmr52_dad_24hr.csv). Depths are
normalized so that the value at the smallest tabulated area = 1.0 — the curve
becomes a relative decay shape that gets multiplied by a per-scenario centroid
depth (sampled from Atlas 14) to produce absolute depths.

Ellipse math follows glo.py lines 84-108, vectorized: for area A and ratio r
(major/minor), b = sqrt(A / (pi * r)), a = r * b.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ELLIPSE_RATIO = 2.5
MI2_TO_M2 = 1609.34 ** 2


def load_dad_curve(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HMR52 DAD curve from CSV.

    CSV format (header required):
        area_sqmi,depth_24hr_in
        10,37.51
        ...

    Returns (areas_sqmi, normalized_ratios) where ratios[0] = 1.0.
    """
    df = pd.read_csv(csv_path)
    areas = df.iloc[:, 0].to_numpy(dtype=float)
    depths = df.iloc[:, 1].to_numpy(dtype=float)
    ratios = depths / depths[0]
    return areas, ratios


def dad_interp(
    area_sqmi: np.ndarray | float,
    dad_areas: np.ndarray,
    dad_ratios: np.ndarray,
) -> np.ndarray:
    """Log-area linear interpolation of the DAD curve. Clamped at endpoints."""
    a = np.atleast_1d(area_sqmi).astype(float)
    a = np.clip(a, dad_areas[0], dad_areas[-1])
    return np.interp(np.log(a), np.log(dad_areas), dad_ratios)


def build_ellipse_polygon(
    center_xy: tuple[float, float],
    area_sqmi: float,
    orientation_deg: float = 0.0,
    ratio: float = ELLIPSE_RATIO,
    n_points: int = 180,
) -> Polygon:
    """Construct an elliptical Polygon in projected meter coordinates.

    Major axis aligns with orientation_deg measured CCW from x-axis.
    """
    cx, cy = center_xy
    area_m2 = area_sqmi * MI2_TO_M2
    b = np.sqrt(area_m2 / (np.pi * ratio))
    a = ratio * b
    beta = np.deg2rad(orientation_deg)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    alpha = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x = cx + a * np.cos(alpha) * cos_b - b * np.sin(alpha) * sin_b
    y = cy + a * np.cos(alpha) * sin_b + b * np.sin(alpha) * cos_b
    return Polygon(zip(x, y))


def build_isohyet_field(
    center_xy: tuple[float, float],
    storm_area_sqmi: float,
    centroid_depth: float,
    dad_areas: np.ndarray,
    dad_ratios: np.ndarray,
    ref_grid_x: np.ndarray,
    ref_grid_y: np.ndarray,
    orientation_deg: float = 0.0,
    ratio: float = ELLIPSE_RATIO,
    clip_outside: bool = True,
) -> np.ndarray:
    """Rasterize the HMR52 elliptical depth pattern onto a reference grid.

    For each cell:
      1. Express position in ellipse-aligned (translated, rotated) coordinates.
      2. Compute normalized elliptical radius rho where rho=1 is the outer ellipse.
      3. The similar ellipse passing through the cell encloses area
         A_inner = rho^2 * storm_area_sqmi (similar ellipses scale area with rho^2).
      4. Depth = centroid_depth * dad_interp(A_inner).

    With ``clip_outside=True`` (default), cells outside the storm ellipse
    (rho > 1) are set to zero — appropriate for HEC-Vortex / HMS gridded
    inputs where phantom rainfall outside the storm should not be applied.
    With ``clip_outside=False``, those cells are clamped to the outer-contour
    depth, useful for visualization of the underlying continuous pattern.
    """
    cx, cy = center_xy
    area_m2 = storm_area_sqmi * MI2_TO_M2
    b = np.sqrt(area_m2 / (np.pi * ratio))
    a = ratio * b
    beta = np.deg2rad(orientation_deg)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    XX, YY = np.meshgrid(ref_grid_x, ref_grid_y)
    dx = XX - cx
    dy = YY - cy
    xp = dx * cos_b + dy * sin_b
    yp = -dx * sin_b + dy * cos_b
    rho2 = (xp / a) ** 2 + (yp / b) ** 2
    outside = rho2 > 1.0
    rho2_clipped = np.clip(rho2, 0.0, 1.0)

    enclosed_area_sqmi = rho2_clipped * storm_area_sqmi
    ratios = dad_interp(enclosed_area_sqmi, dad_areas, dad_ratios)
    depth = centroid_depth * ratios
    if clip_outside:
        depth[outside] = 0.0
    return depth


def build_nested_isohyet_polygons(
    center_xy: tuple[float, float],
    storm_area_sqmi: float,
    centroid_depth: float,
    dad_areas: np.ndarray,
    dad_ratios: np.ndarray,
    orientation_deg: float = 0.0,
    ratio: float = ELLIPSE_RATIO,
    n_points: int = 180,
) -> list[dict]:
    """Build nested elliptical isohyet *bands* from the DAD curve.

    For each tabulated DAD area at or below ``storm_area_sqmi``, plus
    ``storm_area_sqmi`` itself if not already in the table, build an annular
    band between consecutive contours so every ring is simultaneously visible
    when rendered. The innermost band is a solid ellipse (no inner hole);
    each outer band is ``current_ellipse.difference(previous_ellipse)``.

    Returns a list of dicts (smallest area first) with keys:
        ``area_sqmi``        -- outer-contour area
        ``inner_area_sqmi``  -- inner-contour area (None for innermost band)
        ``depth_in``         -- depth at the outer contour (lower bound of
                                depths within this band)
        ``polygon``          -- shapely Polygon (annular geometry)
    """
    areas = sorted({float(a) for a in dad_areas if a <= storm_area_sqmi})
    if not areas or areas[-1] < storm_area_sqmi:
        areas.append(float(storm_area_sqmi))

    features: list[dict] = []
    prev_poly = None
    prev_area: float | None = None
    for area in areas:
        depth = float(centroid_depth) * float(
            dad_interp(area, dad_areas, dad_ratios)
        )
        outer_poly = build_ellipse_polygon(
            center_xy, area, orientation_deg, ratio, n_points
        )
        if prev_poly is None:
            band_poly = outer_poly
        else:
            band_poly = outer_poly.difference(prev_poly)
        features.append(
            {
                "area_sqmi": area,
                "inner_area_sqmi": prev_area,
                "depth_in": depth,
                "polygon": band_poly,
            }
        )
        prev_poly = outer_poly
        prev_area = area
    return features


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DAD_CSV = Path(__file__).parent / "data" / "hmr52_dad_24hr.csv"
    if not DAD_CSV.exists():
        raise SystemExit(f"DAD CSV missing at {DAD_CSV}")

    areas, ratios = load_dad_curve(DAD_CSV)
    print(f"DAD curve loaded: areas={areas.tolist()}")
    print(f"                  ratios={ratios.round(3).tolist()}")

    poly = build_ellipse_polygon((0.0, 0.0), area_sqmi=1000.0, orientation_deg=30.0)
    actual = poly.area / MI2_TO_M2
    print(f"Ellipse polygon: target=1000 sq mi, actual={actual:.1f} sq mi")

    half_extent_m = 200_000
    cell_m = 1_000
    xs = np.arange(-half_extent_m, half_extent_m + cell_m, cell_m)
    ys = np.arange(-half_extent_m, half_extent_m + cell_m, cell_m)
    field = build_isohyet_field(
        center_xy=(0.0, 0.0),
        storm_area_sqmi=2150.0,
        centroid_depth=10.0,
        dad_areas=areas,
        dad_ratios=ratios,
        ref_grid_x=xs,
        ref_grid_y=ys,
        orientation_deg=30.0,
    )
    print(f"Depth field shape: {field.shape}, range [{field.min():.2f}, {field.max():.2f}] in")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        field,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        cmap="Blues",
    )
    plt.colorbar(im, ax=ax, label="Depth (in)")
    px, py = poly.exterior.xy
    ax.plot(px, py, "r-", linewidth=1.0, label="Outer ellipse (1000 sq mi)")
    poly_storm = build_ellipse_polygon((0.0, 0.0), 2150.0, 30.0)
    spx, spy = poly_storm.exterior.xy
    ax.plot(spx, spy, "k-", linewidth=1.0, label="Storm boundary (2150 sq mi)")
    ax.set_aspect("equal")
    ax.set_title("HMR52 depth field test — 2150 sq mi, 30 deg, centroid=10 in")
    ax.legend()
    out = Path(__file__).parent / "test_isohyet_field.png"
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
