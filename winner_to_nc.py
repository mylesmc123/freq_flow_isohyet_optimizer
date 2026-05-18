"""Generate gridded netCDF precipitation files from the winning elliptical storm.

The optimizer's sweep produces a single winning (centroid, orientation, storm_area)
combination — the configuration that maximizes peak flow at the target element
under the chosen ranking frequency. That geometry is assumed to be representative
across return periods (the storm pattern is fixed; only the Atlas 14 centroid
depth scales). This script lifts the winner's geometry from ``_manifest.csv``
and emits one CF-convention-compliant netCDF per (frequency, temporal-distribution)
combination, suitable for ingestion by HEC-Vortex / HMS Specified Hyetograph.

Per-cell depths are computed by:
    1. Sampling the frequency's Atlas 14 raster at the winner's centroid for
       the storm-total centroid depth.
    2. Applying the HMR52 DAD ratio curve to the elliptical isohyet field at
       the winner's orientation + area.
    3. Multiplying that storm-total depth field by the cumulative fraction
       from the chosen temporal distribution at each timestep.

The output grid matches the Atlas 14 raster's native cell footprint (lat/lon,
EPSG:4326) so the netCDF aligns with the source rainfall product.

Structure mirrors ``Atlas14_Grid_Custom_Distro_I57.py`` from the
Atlas14_temporal_distribution workspace.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from basin import WORKING_CRS, get_centroid_depth, load_basin
from hmr52 import (
    build_isohyet_field,
    build_nested_isohyet_polygons,
    load_dad_curve,
)
from isohyet_maker import (
    ATLAS14_RASTERS,
    DAD_CSV,
    INPUT_DIR,
    OUTPUT_DIR,
    PROJECT_NAME,
    SUBBASIN_SHP,
    TEMPORAL_CSV,
    build_reference_grid,
)


# ---- Config -----------------------------------------------------------------

# Frequencies to emit. The 200-yr raster exists for I57 even though it isn't in
# the optimizer's FREQUENCIES_YR sweep set, so it's pulled in here explicitly.
FREQUENCIES_YR: list[int] = [2, 5, 10, 25, 50, 100, 200, 500]

# Atlas 14 raster lookup with the 200-yr entry added.
ATLAS14_RASTERS_NC: dict[int, Path] = {
    **ATLAS14_RASTERS,
    200: INPUT_DIR / "I57" / "I57_200yr24ha.asc",
}

# Cell size for the in-memory depth field before reprojecting onto the Atlas 14
# native grid. 500 m is finer than Atlas 14's ~800 m footprint so the bilinear
# reproject doesn't lose detail at the isohyet boundary.
CELL_SIZE_M: float = 500.0

# Temporal distributions to emit (one netCDF per selected alias per frequency).
# Aliases must appear in TEMPORAL_HEADER_ALIASES below.
SELECTED_TEMPORAL_DISTRIBUTION_ALIASES: list[str] = ["MSE5_24hr"]

# Mapping from temporal-CSV header (post-normalization) -> output-filename alias.
TEMPORAL_HEADER_ALIASES: dict[str, str] = {
    "MSE1 24-hr (depth)": "MSE1_24hr",
    "MSE2 24-hr (depth)": "MSE2_24hr",
    "MSE3 24-hr (depth)": "MSE3_24hr",
    "MSE4 24-hr (depth)": "MSE4_24hr",
    "MSE5 24-hr (depth)": "MSE5_24hr",
    "MSE6 24-hr (depth)": "MSE6_24hr",
    "Type I 24-hr (depth)": "TypeI_24hr",
    "Type IA 24-hr (depth)": "TypeIA_24hr",
    "Type II 6-hr (depth)": "TypeII_6hr",
    "Type II 12-hr (depth)": "TypeII_12hr",
    "Type II 24-hr (depth)": "TypeII_24hr",
    "Type III 6-hr (depth)": "TypeIII_6hr",
    "Type III 12-hr (depth)": "TypeIII_12hr",
    "Type III 24-hr (depth)": "TypeIII_24hr",
}

TEMPORAL_HOURS_COLUMN = "Time (hours)"

# All scenario hyetographs and the winner-comparison page anchor on this date.
STORM_START = datetime.datetime(1970, 1, 1, 0, 0, 0)


# ---- Helpers ----------------------------------------------------------------

def _normalize_csv_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse whitespace + strip in column names; required to match aliases."""
    df = df.copy()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def _load_temporal_table(csv_path: Path, header: str) -> pd.DataFrame:
    """Load (hours, cumulative-fraction) pairs for one named distribution."""
    df = _normalize_csv_headers(pd.read_csv(csv_path))
    if TEMPORAL_HOURS_COLUMN not in df.columns:
        raise KeyError(f"Missing hours column: {TEMPORAL_HOURS_COLUMN!r}")
    if header not in df.columns:
        raise KeyError(f"Missing distribution column: {header!r}")
    sub = df[[TEMPORAL_HOURS_COLUMN, header]].rename(
        columns={TEMPORAL_HOURS_COLUMN: "hours", header: "value"}
    )
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"]).sort_values("hours").reset_index(drop=True)
    return sub


def _resolve_winner(manifest_csv: Path) -> dict:
    """Pull the highest-peak row + its companion storm-geojson properties."""
    manifest = pd.read_csv(manifest_csv)
    if manifest.empty:
        raise SystemExit(f"Manifest is empty: {manifest_csv}")
    winner_row = manifest.loc[manifest["peak_cfs"].idxmax()]

    geojson_path = Path(winner_row["storm_geojson_path"])
    if not geojson_path.exists():
        raise FileNotFoundError(f"Winner storm geojson missing: {geojson_path}")
    fc = json.loads(geojson_path.read_text())
    # Every ring in the storm GeoJSON carries the same scenario metadata
    # (centroid_x_m, centroid_y_m, orientation_deg, storm_area_sqmi), so read
    # the first feature's properties.
    props = fc["features"][0]["properties"]

    return {
        "iteration_name": str(winner_row["iteration_name"]),
        "centroid_id": str(winner_row["centroid_id"]),
        "ranking_frequency_yr": int(winner_row["frequency_yr"]),
        "ranking_peak_cfs": float(winner_row["peak_cfs"]),
        "centroid_xy_m": (float(props["centroid_x_m"]), float(props["centroid_y_m"])),
        "orientation_deg": float(props["orientation_deg"]),
        "storm_area_sqmi": float(props["storm_area_sqmi"]),
    }


def _build_depth_field_da(
    centroid_xy_m: tuple[float, float],
    orientation_deg: float,
    storm_area_sqmi: float,
    centroid_depth_in: float,
    dad_areas: np.ndarray,
    dad_ratios: np.ndarray,
    basin_gdf,
    cell_size_m: float,
) -> xr.DataArray:
    """Rasterize the HMR52 isohyet for one frequency onto WORKING_CRS grid.

    Returns a CRS-tagged DataArray in EPSG:5070 meters; caller is responsible
    for reprojecting to the desired output grid.
    """
    nested = build_nested_isohyet_polygons(
        center_xy=centroid_xy_m,
        storm_area_sqmi=storm_area_sqmi,
        centroid_depth=centroid_depth_in,
        dad_areas=dad_areas,
        dad_ratios=dad_ratios,
        orientation_deg=orientation_deg,
    )
    storm_polygon = nested[-1]["polygon"]
    xs, ys = build_reference_grid(basin_gdf, storm_polygon, cell_size_m=cell_size_m)
    depth = build_isohyet_field(
        center_xy=centroid_xy_m,
        storm_area_sqmi=storm_area_sqmi,
        centroid_depth=centroid_depth_in,
        dad_areas=dad_areas,
        dad_ratios=dad_ratios,
        ref_grid_x=xs,
        ref_grid_y=ys,
        orientation_deg=orientation_deg,
        clip_outside=True,
    )
    # build_isohyet_field uses np.meshgrid in default 'xy' indexing, so
    # depth[0, :] is the south (low-y) row. rioxarray expects north-up
    # arrays (row 0 = max y) — flip and reverse y coords accordingly.
    depth_north_up = depth[::-1, :]
    ys_north_up = ys[::-1]
    da = xr.DataArray(
        depth_north_up,
        dims=("y", "x"),
        coords={"x": xs, "y": ys_north_up},
    )
    return da.rio.write_crs(WORKING_CRS)


def _sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


# ---- Main -------------------------------------------------------------------

def main() -> None:
    project_out = OUTPUT_DIR / PROJECT_NAME
    manifest_csv = project_out / "_manifest.csv"
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest not found: {manifest_csv} — run isohyet_maker first.")

    nc_out_dir = project_out / "nc"
    nc_out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve winner once (geometry is shared across all freqs).
    winner = _resolve_winner(manifest_csv)
    print("=== Winning storm ===")
    print(f"  iteration         : {winner['iteration_name']}")
    print(f"  ranking peak      : {winner['ranking_peak_cfs']:,.0f} cfs ({winner['ranking_frequency_yr']}-yr)")
    print(f"  centroid (EPSG:5070): {winner['centroid_xy_m']}")
    print(f"  orientation       : {winner['orientation_deg']}°")
    print(f"  storm area        : {winner['storm_area_sqmi']:.0f} sq mi")
    print()

    # Validate inputs upfront so we don't get partway through and bail.
    missing_atlas: list[int] = [
        fy for fy in FREQUENCIES_YR if not ATLAS14_RASTERS_NC.get(fy, Path()).exists()
    ]
    if missing_atlas:
        raise SystemExit(f"Missing Atlas 14 rasters for: {missing_atlas}")
    missing_aliases = sorted(
        set(SELECTED_TEMPORAL_DISTRIBUTION_ALIASES) - set(TEMPORAL_HEADER_ALIASES.values())
    )
    if missing_aliases:
        raise SystemExit(f"Unknown temporal distribution aliases: {missing_aliases}")

    # Reverse-lookup header for each selected alias so we read the CSV once
    # per distro and can use the alias as the file-name suffix.
    alias_to_header = {alias: h for h, alias in TEMPORAL_HEADER_ALIASES.items()}
    selected_tables: dict[str, pd.DataFrame] = {
        alias: _load_temporal_table(TEMPORAL_CSV, alias_to_header[alias])
        for alias in SELECTED_TEMPORAL_DISTRIBUTION_ALIASES
    }

    dad_areas, dad_ratios = load_dad_curve(DAD_CSV)
    basin_gdf = load_basin(SUBBASIN_SHP)

    for freq in FREQUENCIES_YR:
        atlas_path = ATLAS14_RASTERS_NC[freq]
        print(f"--- {freq}-yr ---")
        print(f"  Atlas 14 raster: {atlas_path.name}")

        centroid_depth = get_centroid_depth(winner["centroid_xy_m"], atlas_path)
        if not np.isfinite(centroid_depth):
            raise ValueError(
                f"Atlas 14 returned NaN for {freq}-yr at centroid "
                f"{winner['centroid_xy_m']}."
            )
        print(f"  Centroid depth : {centroid_depth:.3f} in")

        # Depth field on WORKING_CRS @ CELL_SIZE_M (over-resolved).
        depth_da_5070 = _build_depth_field_da(
            centroid_xy_m=winner["centroid_xy_m"],
            orientation_deg=winner["orientation_deg"],
            storm_area_sqmi=winner["storm_area_sqmi"],
            centroid_depth_in=centroid_depth,
            dad_areas=dad_areas,
            dad_ratios=dad_ratios,
            basin_gdf=basin_gdf,
            cell_size_m=CELL_SIZE_M,
        )

        # Reproject onto the Atlas 14 raster's native grid so output cells
        # align pixel-for-pixel with the source rainfall product.
        atlas_da = rioxarray.open_rasterio(atlas_path, masked=True).squeeze(
            "band", drop=True
        )
        depth_da_4326 = depth_da_5070.rio.reproject_match(atlas_da)
        # Cells outside the storm ellipse landed as NaN (or 0) post-reproject;
        # treat them as zero rainfall so HMS/Vortex doesn't see missing data.
        depth_da_4326 = depth_da_4326.fillna(0.0)

        for alias, df_table in selected_tables.items():
            # Build time-stacked cumulative depth field.
            chunks: list[xr.DataArray] = []
            for _, row in df_table.iterrows():
                ts = STORM_START + datetime.timedelta(hours=float(row["hours"]))
                da_t = depth_da_4326 * float(row["value"])
                da_t = da_t.assign_coords(time=ts).expand_dims("time")
                chunks.append(da_t)
            cumulative = xr.concat(chunks, dim="time").rename("PrecipCumulative")
            ds = cumulative.to_dataset(name="PrecipCumulative")

            # Drop any band coordinate that survived reproject_match.
            if "band" in ds.coords:
                ds = ds.drop_vars("band")

            # Incremental precip per timestep. PrecipCumulative[0] = 0 for all
            # MSE/Type distributions (they start at fraction 0 at t=0), so
            # filling the first row with 0 matches the physical interpretation.
            ds["PrecipInc"] = (
                ds["PrecipCumulative"]
                .diff(dim="time", label="upper")
                .reindex(time=ds.time, fill_value=0.0)
            )

            # CF-conventions metadata. reproject_match left coords as x/y,
            # rename to longitude/latitude per the example's convention.
            ds = ds.rename({"x": "longitude", "y": "latitude"})
            ds["latitude"].attrs.update(
                units="degrees_north", standard_name="latitude",
                long_name="latitude", axis="Y",
            )
            ds["longitude"].attrs.update(
                units="degrees_east", standard_name="longitude",
                long_name="longitude", axis="X",
            )
            ds["time"].attrs.update(
                standard_name="time", long_name="time", axis="T",
            )
            ds["PrecipCumulative"].attrs.update(
                units="inches", long_name="Cumulative Precipitation",
            )
            ds["PrecipInc"].attrs.update(
                units="inches", long_name="Incremental Precipitation",
            )

            # Temporal distribution trace alongside the gridded fields.
            ds_td = xr.Dataset(
                data_vars={
                    "TemporalDistribution": ("time", df_table["value"].to_numpy())
                },
                coords={
                    "time": ds.time,
                    "hours": ("time", df_table["hours"].to_numpy()),
                },
            )
            ds = xr.merge([ds, ds_td])
            ds["TemporalDistribution"].attrs.update(
                units="fraction",
                long_name="Temporal Distribution Cumulative Fraction",
                temporalDuration=alias,
                source=(
                    f"{alias_to_header[alias]} Temporal Distribution Table "
                    f"(alias: {alias})"
                ),
            )

            # Provenance for traceability: who built this and why.
            ds.attrs.update(
                title=f"Atlas 14 {freq}-yr 24-hr {alias} — winner storm {winner['iteration_name']}",
                source="freq_flow_isohyet_optimizer / winner_to_nc.py",
                winner_iteration=winner["iteration_name"],
                winner_centroid_id=winner["centroid_id"],
                winner_orientation_deg=winner["orientation_deg"],
                winner_storm_area_sqmi=winner["storm_area_sqmi"],
                winner_ranking_frequency_yr=winner["ranking_frequency_yr"],
                frequency_yr=freq,
                temporal_distribution=alias,
                atlas14_centroid_depth_in=float(centroid_depth),
                Conventions="CF-1.8",
            )

            alias_safe = _sanitize(alias)
            out_path = nc_out_dir / f"Atlas14_{PROJECT_NAME}_{freq:03d}yr_{alias_safe}.nc"
            # float32 + zlib level 4 keeps precision well below precip
            # measurement noise (~0.01 in) and shrinks the sparse
            # outside-ellipse zero regions ~10x.
            encoding = {
                "PrecipCumulative": {"dtype": "float32", "zlib": True, "complevel": 4},
                "PrecipInc": {"dtype": "float32", "zlib": True, "complevel": 4},
            }
            print(f"  -> {out_path.name}")
            ds.to_netcdf(out_path, encoding=encoding)

    print(f"\nDone. netCDFs written to {nc_out_dir}")


if __name__ == "__main__":
    main()
