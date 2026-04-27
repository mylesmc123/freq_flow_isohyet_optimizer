'''
This file creates elliptical isohyetal storms by combining:
1. A basin outline dissolved from a subbasin shapefile.
2. An Atlas 14 raster grid (https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_grid.html) to set
   the storm magnitude (50-yr, 100-yr, 500-yr, 24-hr durations).
3. The HMR52 depth-area-duration relationship to set the storm's spatial decay shape,
   with a 2.5:1 elliptical major-to-minor axis ratio.

Magnitude/spatial split (the engineering rationale):
  Atlas 14 grid values are point/grid-cell precipitation frequencies. Stamping a
  uniform Atlas 14 depth across an 800 sq mi basin would imply every point in the
  basin simultaneously experiences its 100-yr point depth, which is physically
  unrealistic and inflates flows. The HMR52 spatial decay implements the depth-area
  reduction that fixes that — basin-average depth ends up *less* than the Atlas 14
  centroid value, which is the correct hydrologic behavior.

Centroid depth sourcing:
  For each iterated scenario, the storm centroid depth is sampled directly from the
  Atlas 14 raster at the scenario's (x, y) centroid location — Atlas 14 grid values
  are not aggregated or averaged. The HMR52 DAD curve sets the relative depth at
  every other point in the storm ellipse, scaled to the centroid value.

Outputs per scenario:
  - Polygon shapefile of the elliptical isohyets.
  - Starting depth raster.
  - xarray dataset with time dimension (after applying a temporal distribution),
    written as a CF-compliant netCDF.
  - Gridded DSS file via HEC-Vortex for use in HEC-HMS.
  - Subbasin-average hyetographs as a DSS file (HEC-HMS Specified Hyetograph met model).

HEC-HMS automation:
  - The hyetograph DSS is patched into an existing HEC-HMS Met Model.
  - HEC-HMS is invoked via subprocess; results DSS is parsed for the target location's
    flow series, exported as CSV.

Optimization:
  - A grid of candidate storm centroids is built across the basin extent.
  - For each centroid, a sweep of ellipse orientations and storm-area sizes is run.
  - Each scenario produces its own shapefile, netCDF, hyetograph DSS, and results.
  - The scenario producing peak flow at the target location is reported.

HMR 52 Notes (https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/files/179606790/179606808/1/1707407159327/HMR52.pdf)
 
Table 9 Major Storms - Central Plains: No. of Storms - 6,  Avg Orientation (deg) - 235, Range (deg) - 160 to 285 
Area (sqmi)	%
10	    1
100	    0.827
200	    0.756
300	    0.72
400	    0.695
500	    0.676
700	    0.648
900	    0.628
1000	0.611
1500	0.553
2000	0.507
3000	0.433
4000	0.4
5000	0.382
6000	0.366
7000	0.352
8000	0.34
9000	0.331
10000	0.323

'''


from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from basin import (
    MI2_TO_M2,
    WORKING_CRS,
    build_subbasin_centroids,
    compute_subbasin_means,
    get_centroid_depth,
    load_basin,
    load_subbasins,
)
from dss_io import write_subbasin_hyetographs_dss
from hmr52 import (
    build_isohyet_field,
    build_nested_isohyet_polygons,
    load_dad_curve,
)
from hms_runner import (
    HMS_HYETO_DSS,
    TARGET_ELEMENT,
    find_results_dss,
    hms_run_name,
    read_target_hydrograph,
    run_hms,
)
from results_viz import build_results_html
from temporal import (
    build_subbasin_hyetographs,
    load_temporal_distribution,
)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
DAD_CSV = DATA_DIR / "hmr52_dad_24hr.csv"
TEMPORAL_CSV = DATA_DIR / "HydroCad Rainfall Temporal Distribtions.csv"

# Project tag — outputs are written to OUTPUT_DIR/PROJECT_NAME/ so that runs
# for different basins/studies don't collide.
PROJECT_NAME = "I57"
SUBBASIN_SHP = INPUT_DIR / "I57" / "I-57_HMS_Subbasins.shp"

# Atlas 14 raster path per frequency. 50-yr currently excluded from sweeps.
ATLAS14_RASTERS: dict[int, Path] = {
    100: INPUT_DIR / "I57" / "I57_100yr24ha.asc",
    500: INPUT_DIR / "I57" / "I57_500yr24ha.asc",
}
FREQUENCIES_YR: list[int] = [100, 500]

# Hyetograph start time — must overlap the HMS run's Control Specifications
# time window. Set to match the gridded-precip control spec already used in
# the HMS project so the same control window can drive Specified Hyetograph
# runs without edits.
STORM_START_TIME = "1970-01-01 00:00:00"

# Locked storm-area size — see docstring for rationale.
STORM_AREA_SQMI = 20000.0


def _safe_id(s: str) -> str:
    """Sanitize a string for use as a filename component."""
    return re.sub(r"\W+", "_", str(s).strip()).strip("_")


def scenario_basename(
    frequency_yr: int,
    centroid_id: str,
    orientation_deg: float,
    storm_area_sqmi: float,
) -> str:
    """Deterministic basename for scenario output files.

    Encodes frequency, centroid grid id, orientation, and storm-area size so
    that paired files (storm GeoJSON, hyetograph CSV, target hydrograph CSV)
    share a common prefix and can be matched by listing.
    """
    return (
        f"{frequency_yr:03d}yr_{centroid_id}"
        f"_{int(round(orientation_deg)):03d}deg"
        f"_{int(round(storm_area_sqmi)):05d}sqmi"
    )


def build_reference_grid(
    basin_gdf: gpd.GeoDataFrame,
    storm_polygon: Polygon,
    cell_size_m: float = 1000.0,
    margin_m: float = 5_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference grid (in WORKING_CRS meters) covering basin + storm with margin."""
    bx_min, by_min, bx_max, by_max = basin_gdf.total_bounds
    sx_min, sy_min, sx_max, sy_max = storm_polygon.bounds
    minx = min(bx_min, sx_min) - margin_m
    miny = min(by_min, sy_min) - margin_m
    maxx = max(bx_max, sx_max) + margin_m
    maxy = max(by_max, sy_max) + margin_m
    xs = np.arange(minx, maxx + cell_size_m, cell_size_m)
    ys = np.arange(miny, maxy + cell_size_m, cell_size_m)
    return xs, ys


def run_scenario(
    subbasin_shp: Path,
    atlas14_raster: Path,
    centroid_xy: tuple[float, float] | None,
    orientation_deg: float,
    storm_area_sqmi: float,
    frequency_yr: int,
    centroid_id: str,
    output_dir: Path,
    dad_csv: Path = DAD_CSV,
    temporal_csv: Path = TEMPORAL_CSV,
    distribution_name: str = "Type II 24-hr",
    output_dt_minutes: float = 5.0,
    cell_size_m: float = 1000.0,
    name_col: str = "Name",
    start_time: str = STORM_START_TIME,
) -> dict:
    """Run a single isohyet scenario through the in-memory pipeline.

    Pipeline: load basin/subbasins -> sample Atlas 14 at centroid -> build
    HMR52 elliptical depth field on a reference grid -> compute subbasin
    storm-total means -> apply temporal distribution to produce per-subbasin
    hyetograph DataFrame.

    Side effects: writes ``{basename}_storm.geojson`` (storm ellipse polygon
    in EPSG:4326 with scenario metadata) and ``{basename}_hyeto.csv``
    (per-subbasin incremental precip per timestep) to ``output_dir``.

    Returns a diagnostics dict including the in-memory hyetograph DataFrame
    so the caller can inspect/use it without re-reading the CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    basin = load_basin(subbasin_shp)
    subbasins = load_subbasins(subbasin_shp, name_col=name_col)

    if centroid_xy is None:
        c = basin.geometry.union_all().centroid
        centroid_xy = (c.x, c.y)

    centroid_depth = get_centroid_depth(centroid_xy, atlas14_raster)
    if not np.isfinite(centroid_depth):
        raise ValueError(f"Atlas 14 returned NaN at centroid {centroid_xy}.")

    dad_areas, dad_ratios = load_dad_curve(dad_csv)
    nested_isohyets = build_nested_isohyet_polygons(
        center_xy=centroid_xy,
        storm_area_sqmi=storm_area_sqmi,
        centroid_depth=centroid_depth,
        dad_areas=dad_areas,
        dad_ratios=dad_ratios,
        orientation_deg=orientation_deg,
    )
    storm_polygon = nested_isohyets[-1]["polygon"]

    xs, ys = build_reference_grid(
        basin, storm_polygon, cell_size_m=cell_size_m
    )

    depth_field = build_isohyet_field(
        center_xy=centroid_xy,
        storm_area_sqmi=storm_area_sqmi,
        centroid_depth=centroid_depth,
        dad_areas=dad_areas,
        dad_ratios=dad_ratios,
        ref_grid_x=xs,
        ref_grid_y=ys,
        orientation_deg=orientation_deg,
        clip_outside=True,
    )

    sb_means = compute_subbasin_means(
        depth_field, xs, ys, subbasins, name_col=name_col
    )

    t_cum, frac_cum = load_temporal_distribution(
        temporal_csv, distribution_name
    )
    hyeto_df = build_subbasin_hyetographs(
        sb_means,
        t_cum,
        frac_cum,
        output_dt_minutes=output_dt_minutes,
        start_time=start_time,
    )

    basename = scenario_basename(
        frequency_yr, centroid_id, orientation_deg, storm_area_sqmi
    )
    geojson_path = output_dir / f"{basename}_storm.geojson"
    centroid_geojson_path = output_dir / f"{basename}_storm_centroid.geojson"
    hyeto_csv_path = output_dir / f"{basename}_hyeto.csv"

    n_rings = len(nested_isohyets)
    storm_gdf = gpd.GeoDataFrame(
        {
            "area_sqmi": [f["area_sqmi"] for f in nested_isohyets],
            "inner_area_sqmi": [f["inner_area_sqmi"] for f in nested_isohyets],
            "depth_in": [f["depth_in"] for f in nested_isohyets],
            "frequency_yr": [frequency_yr] * n_rings,
            "centroid_id": [centroid_id] * n_rings,
            "orientation_deg": [orientation_deg] * n_rings,
            "storm_area_sqmi": [storm_area_sqmi] * n_rings,
            "centroid_x_m": [centroid_xy[0]] * n_rings,
            "centroid_y_m": [centroid_xy[1]] * n_rings,
            "centroid_depth_in": [centroid_depth] * n_rings,
        },
        geometry=[f["polygon"] for f in nested_isohyets],
        crs=WORKING_CRS,
    ).to_crs("EPSG:4326")
    storm_gdf.to_file(geojson_path, driver="GeoJSON")
    # pyogrio writes the CRS as the GeoJSON-spec name "urn:ogc:def:crs:OGC:1.3:CRS84"
    # (functionally identical to EPSG:4326 but some tools don't recognize the
    # alias). Rewrite the crs member to the explicit EPSG identifier.
    fc = json.loads(geojson_path.read_text())
    fc["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
    }
    geojson_path.write_text(json.dumps(fc))

    # Single-Point GeoJSON marking the storm centroid for visual verification
    # against the Atlas 14 raster in QGIS. If the marker lands where you read
    # the centroid_depth_in value below, the sampling is correct.
    centroid_gdf = gpd.GeoDataFrame(
        {
            "frequency_yr": [frequency_yr],
            "centroid_id": [centroid_id],
            "orientation_deg": [orientation_deg],
            "storm_area_sqmi": [storm_area_sqmi],
            "centroid_x_m": [centroid_xy[0]],
            "centroid_y_m": [centroid_xy[1]],
            "centroid_depth_in": [centroid_depth],
        },
        geometry=[Point(centroid_xy)],
        crs=WORKING_CRS,
    ).to_crs("EPSG:4326")
    centroid_gdf.to_file(centroid_geojson_path, driver="GeoJSON")
    fc_c = json.loads(centroid_geojson_path.read_text())
    fc_c["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
    }
    centroid_geojson_path.write_text(json.dumps(fc_c))

    hyeto_df.to_csv(hyeto_csv_path)

    return {
        "basename": basename,
        "centroid_xy": centroid_xy,
        "centroid_depth_in": centroid_depth,
        "basin_area_sqmi": float(basin.geometry.area.sum() / MI2_TO_M2),
        "subbasin_means_in": sb_means,
        "hyeto_df": hyeto_df,
        "geojson_path": geojson_path,
        "centroid_geojson_path": centroid_geojson_path,
        "hyeto_csv_path": hyeto_csv_path,
    }


def run_full_scenario(
    centroid_id: str,
    centroid_xy: tuple[float, float],
    orientation_deg: float,
    frequency_yr: int,
    output_dir: Path,
    subbasin_shp: Path = SUBBASIN_SHP,
    storm_area_sqmi: float = STORM_AREA_SQMI,
    name_col: str = "name",
) -> dict:
    """Full pipeline for one scenario: hyetograph -> HMS -> target hydrograph.

    Steps:
      1. ``run_scenario`` — build the storm field, subbasin means, and the
         per-subbasin incremental-precip DataFrame; also writes the storm
         GeoJSON + per-subbasin hyetograph CSV.
      2. ``write_subbasin_hyetographs_dss`` overwrites ``HMS_HYETO_DSS`` so the
         HMS Met Model picks up the new precip without any edits.
      3. ``run_hms`` invokes the HMS CLI in batch mode to compute the named run.
      4. ``read_target_hydrograph`` pulls the outlet flow series from the HMS
         results DSS (multi-block stitched, missing-value sentinels filtered).
      5. The target hydrograph is saved as ``<basename>_target.csv``.

    Returns the scenario diagnostics dict (output of ``run_scenario``) augmented
    with ``target_csv_path``, ``peak_cfs``, ``peak_time``, and ``hms_returncode``.
    """
    if frequency_yr not in ATLAS14_RASTERS:
        raise ValueError(
            f"No Atlas 14 raster registered for {frequency_yr}-yr; "
            f"available: {list(ATLAS14_RASTERS)}"
        )

    safe_id = _safe_id(centroid_id)

    result = run_scenario(
        subbasin_shp=subbasin_shp,
        atlas14_raster=ATLAS14_RASTERS[frequency_yr],
        centroid_xy=centroid_xy,
        orientation_deg=orientation_deg,
        storm_area_sqmi=storm_area_sqmi,
        frequency_yr=frequency_yr,
        centroid_id=safe_id,
        output_dir=output_dir,
        name_col=name_col,
    )

    write_subbasin_hyetographs_dss(
        result["hyeto_df"], HMS_HYETO_DSS, overwrite=True
    )

    run_name = hms_run_name(frequency_yr)
    print(f"  Running HMS ({run_name}) ...", flush=True)
    hms_result = run_hms(run_name)
    if hms_result.returncode != 0:
        print(f"  HMS exit code: {hms_result.returncode}")
        if hms_result.stderr:
            print("  HMS stderr (tail):")
            for line in hms_result.stderr.splitlines()[-6:]:
                print(f"    {line}")

    target_df = read_target_hydrograph(
        find_results_dss(run_name), TARGET_ELEMENT, run_name=run_name
    )
    target_csv = output_dir / f"{result['basename']}_target.csv"
    target_df[["flow_cfs"]].to_csv(target_csv)

    peak_cfs = float(target_df["flow_cfs"].max())
    peak_time = target_df["flow_cfs"].idxmax()

    return {
        **result,
        "target_csv_path": target_csv,
        "target_df": target_df[["flow_cfs"]],
        "peak_cfs": peak_cfs,
        "peak_time": peak_time,
        "hms_returncode": hms_result.returncode,
        "centroid_id_raw": centroid_id,
        "frequency_yr": frequency_yr,
        "orientation_deg": orientation_deg,
    }


if __name__ == "__main__":
    if not SUBBASIN_SHP.exists():
        raise SystemExit(f"Subbasin shapefile not found: {SUBBASIN_SHP}")
    for fy in (100, 500):
        if not ATLAS14_RASTERS[fy].exists():
            raise SystemExit(f"Atlas 14 raster missing for {fy}-yr: {ATLAS14_RASTERS[fy]}")

    out_dir = OUTPUT_DIR / PROJECT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- TEST SWEEP: 2 centroids x 2 orientations x 2 frequencies = 8 runs ----
    subs = load_subbasins(SUBBASIN_SHP, name_col="name")
    centroids_gdf = build_subbasin_centroids(subs, name_col="name")
    test_centroids = centroids_gdf.iloc[:2]   # first two subbasins by file order
    test_orientations = (220, 260)
    test_frequencies = (100, 500)

    rows: list[dict] = []
    n_total = len(test_centroids) * len(test_orientations) * len(test_frequencies)
    n = 0
    for _, row in test_centroids.iterrows():
        cid = str(row["centroid_id"])
        cxy = (row.geometry.x, row.geometry.y)
        for orient in test_orientations:
            for freq in test_frequencies:
                n += 1
                print(f"\n[{n}/{n_total}] {freq}-yr  centroid={cid!r}  orient={orient}deg")
                result = run_full_scenario(
                    centroid_id=cid,
                    centroid_xy=cxy,
                    orientation_deg=float(orient),
                    frequency_yr=freq,
                    output_dir=out_dir,
                )
                print(
                    f"  Peak: {result['peak_cfs']:>12,.0f} cfs"
                    f"   at {result['peak_time']}"
                    f"   centroid_depth={result['centroid_depth_in']:.2f} in"
                )
                metadata = {
                    "iteration_name": result["basename"],
                    "frequency_yr": freq,
                    "centroid_id": cid,
                    "orientation_deg": orient,
                    "centroid_depth_in": result["centroid_depth_in"],
                    "peak_cfs": result["peak_cfs"],
                    "peak_time": str(result["peak_time"]),
                    "storm_geojson_path": str(result["geojson_path"]),
                    "hyeto_csv_path": str(result["hyeto_csv_path"]),
                    "target_csv_path": str(result["target_csv_path"]),
                    "hms_returncode": result["hms_returncode"],
                }
                # Wide-format hydrograph: one column per timestamp ("Q@YYYY-MM-DD HH:MM").
                # Append at the end of the row so metadata stays readable.
                target_series = result["target_df"]["flow_cfs"]
                hyd_cols = {
                    f"Q@{ts.strftime('%Y-%m-%d %H:%M')}": float(v)
                    for ts, v in target_series.items()
                }
                rows.append({**metadata, **hyd_cols})

    manifest = pd.DataFrame(rows)
    manifest_csv = out_dir / "_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    print("\n=== Sweep complete ===")
    # Print only the metadata columns to keep the console summary readable.
    summary_cols = [
        "iteration_name", "frequency_yr", "centroid_id", "orientation_deg",
        "centroid_depth_in", "peak_cfs", "peak_time", "hms_returncode",
    ]
    print(manifest[summary_cols].to_string(index=False))
    print(f"\nManifest (wide, with hydrograph cols): {manifest_csv}")
    winner = manifest.loc[manifest["peak_cfs"].idxmax()]
    print(
        f"\nHighest peak: {winner['peak_cfs']:,.0f} cfs"
        f"  ({winner['frequency_yr']}-yr, {winner['centroid_id']!r}, {winner['orientation_deg']}°)"
    )

    # Build the HTML results visualization next to the manifest.
    html_path = build_results_html(
        manifest_csv=manifest_csv,
        out_dir=out_dir,
        subbasin_shp=SUBBASIN_SHP,
        name_col="name",
        title=f"Iteration Results — {PROJECT_NAME}",
        target_element=TARGET_ELEMENT,
    )
    print(f"Results page: {html_path}")

    # Mirror the rendered HTML + storm thumbnails into presentation/ so the
    # results page is reachable from the deployed methodology site (Render
    # only serves files inside the publish directory).
    import shutil
    deploy_dir = PROJECT_DIR / "presentation"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_path, deploy_dir / html_path.name)
    src_viz = out_dir / "_viz"
    dst_viz = deploy_dir / "_viz"
    if src_viz.exists():
        if dst_viz.exists():
            shutil.rmtree(dst_viz)
        shutil.copytree(src_viz, dst_viz)
    print(f"Deployed copy:  {deploy_dir / html_path.name}")

