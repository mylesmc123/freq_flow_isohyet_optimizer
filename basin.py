"""Basin geometry loading, Atlas 14 sampling, and centroid-grid generation.

Working CRS is USA Contiguous Albers Equal Area (EPSG:5070) so that area
calculations and ellipse construction happen in meters with minimal distortion.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure GDAL_DATA is set when launchers (VSCode debugpy, bare cmd) start Python
# without running the conda env activation scripts. Avoids "Cannot find
# gdalvrt.xsd" warnings from pyogrio/GDAL during shapefile and GeoJSON I/O.
if "GDAL_DATA" not in os.environ:
    _gdal_dir = Path(sys.prefix) / "Library" / "share" / "gdal"
    if _gdal_dir.exists():
        os.environ["GDAL_DATA"] = str(_gdal_dir)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point

WORKING_CRS = "EPSG:5070"
MI2_TO_M2 = 1609.34 ** 2

def load_basin(subbasin_shp: str | Path) -> gpd.GeoDataFrame:
    """Read a subbasin shapefile, dissolve to a single basin polygon, reproject to WORKING_CRS."""
    gdf = gpd.read_file(subbasin_shp).to_crs(WORKING_CRS)
    gdf["_diss"] = 1
    return gdf.dissolve(by="_diss")[["geometry"]]


def get_centroid_depth(
    centroid_xy: tuple[float, float],
    atlas14_raster: str | Path,
    units_scale: float = 0.001,
) -> float:
    """Sample the Atlas 14 raster at a single (x, y) location in WORKING_CRS.

    The Atlas 14 raster's native CRS may differ (typically NAD83, EPSG:4269);
    we reproject the single point rather than the raster.

    NOAA Atlas 14 ASCII grids store depth as integer thousandths of inches, so
    the default ``units_scale`` of 0.001 converts the raw value to inches.

    Returns NaN when the sampled cell is the raster's nodata value (e.g., a
    centroid candidate outside the Atlas 14 coverage area).
    """
    with rasterio.open(atlas14_raster) as src:
        if src.crs.to_string() != WORKING_CRS:
            xs, ys = warp_transform(
                WORKING_CRS, src.crs, [centroid_xy[0]], [centroid_xy[1]]
            )
            sample_xy = (xs[0], ys[0])
        else:
            sample_xy = centroid_xy
        for val in src.sample([sample_xy]):
            raw = float(val[0])
            if src.nodata is not None and raw == src.nodata:
                return float("nan")
            return raw * units_scale


def build_centroid_grid(
    basin_gdf: gpd.GeoDataFrame,
    spacing_mi: float,
    clip_to_basin: bool = True,
) -> gpd.GeoDataFrame:
    """Generate a regular grid of candidate storm-centroid points across the basin extent."""
    minx, miny, maxx, maxy = basin_gdf.total_bounds
    spacing_m = spacing_mi * 1609.34
    xs = np.arange(minx, maxx + spacing_m, spacing_m)
    ys = np.arange(miny, maxy + spacing_m, spacing_m)
    XX, YY = np.meshgrid(xs, ys)
    points = [Point(x, y) for x, y in zip(XX.ravel(), YY.ravel())]
    grid = gpd.GeoDataFrame(geometry=points, crs=WORKING_CRS)
    if clip_to_basin:
        basin_geom = basin_gdf.union_all()
        grid = grid[grid.intersects(basin_geom)].reset_index(drop=True)
    return grid


def load_subbasins(
    subbasin_shp: str | Path,
    name_col: str = "Name",
) -> gpd.GeoDataFrame:
    """Read a subbasin shapefile preserving individual polygons, reproject to WORKING_CRS."""
    gdf = gpd.read_file(subbasin_shp).to_crs(WORKING_CRS)
    if name_col not in gdf.columns:
        raise KeyError(
            f"Column {name_col!r} not in shapefile. Available: {list(gdf.columns)}"
        )
    return gdf[[name_col, "geometry"]].reset_index(drop=True)


def build_subbasin_centroids(
    subbasins_gdf: gpd.GeoDataFrame,
    name_col: str = "Name",
    use_representative_point: bool = True,
) -> gpd.GeoDataFrame:
    """One candidate storm centroid per subbasin, tagged with the subbasin name.

    With ``use_representative_point=True`` (default), the returned point is
    guaranteed to fall inside the subbasin polygon — important for irregular
    or narrow shapes where the geometric centroid can land outside. Set
    False to use the geometric centroid instead.
    """
    geom = (
        subbasins_gdf.geometry.representative_point()
        if use_representative_point
        else subbasins_gdf.geometry.centroid
    )
    return gpd.GeoDataFrame(
        {"centroid_id": subbasins_gdf[name_col].to_numpy()},
        geometry=geom.to_numpy(),
        crs=subbasins_gdf.crs,
    ).reset_index(drop=True)


def compute_subbasin_means(
    depth_field_2d: np.ndarray,
    ref_grid_x: np.ndarray,
    ref_grid_y: np.ndarray,
    subbasins_gdf: gpd.GeoDataFrame,
    name_col: str = "Name",
) -> pd.Series:
    """Area-weighted mean of the depth field within each subbasin polygon.

    Rasterizes subbasin polygons to the same grid as the depth field, then
    averages depth-field values over the cells belonging to each polygon.

    The depth field comes from ``hmr52.build_isohyet_field`` where rows of the
    array correspond to ``ref_grid_y`` (which is increasing with row index).
    Rasterio expects rows top-down (y decreasing), so we flip the depth array
    vertically to match the transform.

    Returns:
        pd.Series of basin-mean depth in inches, indexed by subbasin name.
        Subbasins that don't overlap the grid get NaN.
    """
    cell_x = float(ref_grid_x[1] - ref_grid_x[0])
    cell_y = float(ref_grid_y[1] - ref_grid_y[0])
    nrows, ncols = depth_field_2d.shape
    transform = from_origin(
        ref_grid_x[0] - cell_x / 2.0,
        ref_grid_y[-1] + cell_y / 2.0,
        cell_x,
        cell_y,
    )
    flipped = np.flipud(depth_field_2d)

    shapes = [
        (geom, idx + 1) for idx, geom in enumerate(subbasins_gdf.geometry)
    ]
    zones = rasterize(
        shapes,
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,
        dtype="int32",
    )

    means: dict[str, float] = {}
    for sb_id, (_, row) in enumerate(subbasins_gdf.iterrows(), start=1):
        mask = zones == sb_id
        if mask.any():
            means[row[name_col]] = float(flipped[mask].mean())
        else:
            means[row[name_col]] = float("nan")
    return pd.Series(means, name="basin_mean_depth_in")


if __name__ == "__main__":
    import sys

    SUBBASIN_SHP = Path(r"C:\py\freq_flow_isohyet_optimizer\input\I57\I-57_HMS_Subbasins.shp")
    ATLAS14_RASTER = Path(r"C:\py\freq_flow_isohyet_optimizer\input\I57\I57_050yr24ha.asc")

    if not SUBBASIN_SHP.exists():
        print(f"Edit SUBBASIN_SHP in {__file__} to point to a real shapefile.")
        sys.exit(0)

    basin = load_basin(SUBBASIN_SHP)
    area_sqmi = basin.geometry.area.sum() / MI2_TO_M2
    print(f"Basin loaded: {area_sqmi:.1f} sq mi")

    centroid = basin.geometry.iloc[0].centroid
    print(f"Basin centroid (EPSG:5070): ({centroid.x:.0f}, {centroid.y:.0f})")

    if ATLAS14_RASTER.exists():
        depth = get_centroid_depth((centroid.x, centroid.y), ATLAS14_RASTER)
        print(f"Atlas 14 sampled at basin centroid: {depth:.2f}")

    grid = build_centroid_grid(basin, spacing_mi=5.0)
    print(f"Centroid grid: {len(grid)} candidate points at 5-mi spacing")
