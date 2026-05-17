# Frequency-Flow Isohyet Optimizer

Builds elliptical isohyetal storms (Atlas 14 magnitude + HMR52 spatial decay),
sweeps storm centroids/orientations/frequencies through an existing HEC-HMS
project, and reports the scenario producing peak flow at a target element.

See [presentation/index.html](presentation/index.html) for the full methodology
write-up.

## One-time setup

1. **Install the conda env** (Windows, Miniforge/Mambaforge recommended):
   ```powershell
   conda env create -f environment.yml
   conda activate hmr
   ```
2. **Install HEC-HMS 4.12** at the default path
   `C:\Program Files\HEC\HEC-HMS\4.12\`. If yours differs, update `HMS_CMD` in
   [hms_runner.py:28](hms_runner.py#L28).

## Make a new run

A "run" = one sweep of {centroids} x {orientations} x {frequencies} for one
study basin. Each sweep is keyed by `PROJECT_NAME` so its outputs land in
`output/<PROJECT_NAME>/` and don't collide with prior sweeps.

### 1. Stage the input data

Under `input/<PROJECT_NAME>/`, drop:

- **Subbasin shapefile** — one polygon per HEC-HMS subbasin. The attribute
  column holding the subbasin name must match the name used in the HMS basin
  model exactly (HMS-side hyetograph paths are matched on it).
- **NOAA Atlas 14 ASCII grids** for each frequency you want to sweep — one
  `.asc` (+ `.prj`) per frequency, 24-hr duration. Download from
  https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_grid.html. Values are assumed to be
  thousandths of an inch (Atlas 14 default); `basin.get_centroid_depth` scales
  by `0.001`.

### 2. Prepare the HEC-HMS project

The sweep does NOT build an HMS project — it patches one you already have.

- Copy your HMS project to a working location (e.g. under `C:\Temp\...`) so
  the sweep can overwrite `data/hyeto.dss` on every iteration without touching
  your master copy.
- The project must contain:
  - A **Specified Hyetograph** Met Model whose gages point at
    `data/hyeto.dss` with blank A-part and F-part, B-part = subbasin name,
    C-part = `PRECIP-INC`. (Matches what
    [dss_io.py](dss_io.py) writes.)
  - **Simulation runs** named `050yr_24hr_Hyeto`, `100yr_24hr_Hyeto`,
    `500yr_24hr_Hyeto` (one per frequency; see
    [hms_runner.py:320-327](hms_runner.py#L320-L327)) — Hyeto-driven runs.
  - A **target basin element** whose name matches `TARGET_ELEMENT` in
    [hms_runner.py:331](hms_runner.py#L331).
  - (Optional, for the winner-vs-baseline comparison page) baseline runs
    named `050yr_24hr`, `100yr_24hr`, `500yr_24hr` driven by gridded Atlas 14.

### 3. Wire up the config

Edit [hms_runner.py:29-32](hms_runner.py#L29-L32):
- `HMS_PROJECT_DIR` — path to your working HMS project.
- `HMS_PROJECT_NAME` — base name of the `.hms` file in that directory.
- `TARGET_ELEMENT` — HMS basin element name where peak flow is evaluated.

Edit [isohyet_maker.py:124-133](isohyet_maker.py#L124-L133):
- `PROJECT_NAME` — output-folder tag for this sweep.
- `SUBBASIN_SHP` — path to the shapefile from step 1.
- `ATLAS14_RASTERS` — map of `frequency_yr -> .asc path`.
- `FREQUENCIES_YR` — frequencies to actually sweep.
- `TEMPORAL_CSV` — table of temporal distributions from HydroCAD
- `TEMPORAL_DISTRIBUTION` - column to use from TEMPORAL_CSV
-  `DAD_CSV` - the Depth Area Duration Table used to create the elliptical storm

Optional tuning in [isohyet_maker.py:485](isohyet_maker.py#L485):
- `test_orientations` — ellipse orientations in degrees (default
  `(160, 180, 200, 220, 240, 260, 280)`).
- `test_centroids` — defaults to one representative point per subbasin via
  `build_subbasin_centroids`; swap for `build_centroid_grid(basin, spacing_mi=...)`
  for a regular grid.
- `STORM_AREA_SQMI` — locked storm-area size at
  [isohyet_maker.py:142](isohyet_maker.py#L142).
- `name_col` passed to `load_subbasins` — column name in your shapefile.

### 4. Run the sweep

```powershell
conda activate hmr
python isohyet_maker.py
```

What it does:
- Snapshots the HMS project to `<HMS_PROJECT_DIR>_pristine\` on first run
  (used to recover from HMS-corruption-on-failure; delete that folder
  manually if you've made HMS edits you want re-captured).
- For each `(centroid, orientation, frequency)`:
  1. Build elliptical storm + per-subbasin hyetograph.
  2. Overwrite `data/hyeto.dss`.
  3. Invoke HMS CLI in batch mode (~30-90 s/run for JVM + compute).
  4. Read flow at `TARGET_ELEMENT` from results DSS, write `<basename>_target.csv`.
  5. Append a row to `output/<PROJECT_NAME>/_manifest.csv`.

The manifest is written incrementally — an interrupted sweep resumes from
where it left off when re-run (rows already present are skipped).

### 5. Starting over / clearing caches

Two cache artifacts persist across runs. The per-run HMS files (`<run>.dss`,
`results/RUN_<run>.results`, `<run>.out`) are auto-cleared every iteration by
[hms_runner._clear_hms_run_cache](hms_runner.py#L175) and need no action.

To force a fresh sweep — e.g. after editing inputs, the DAD/temporal CSVs, or
shapefile attributes — delete the **resume manifest** before re-running:

```powershell
Remove-Item output\<PROJECT_NAME>\_manifest.csv
```

Without this, scenarios already listed in the manifest will be skipped and
the page outputs will be rebuilt from stale rows. To start completely fresh,
delete the whole `output\<PROJECT_NAME>\` folder.

To force a fresh **pristine HMS-project snapshot** (e.g. after you've edited
the HMS project itself — basin model, control specs, met model wiring),
delete the pristine backup folder so the next sweep re-captures it:

```powershell
Remove-Item -Recurse -Force "<HMS_PROJECT_DIR>_pristine"
```

The `_lastfailure` folder next to it is just forensic state from the most
recent HMS crash; safe to delete any time.

### 6. Outputs

In `output/<PROJECT_NAME>/`:
- `_manifest.csv` — one row per scenario, with metadata + a wide column per
  hydrograph timestep (`Q@YYYY-MM-DD HH:MM`).
- `<basename>_storm.geojson` — nested isohyet polygons (EPSG:4326).
- `<basename>_storm_centroid.geojson` — single-point centroid for QGIS
  verification against the Atlas 14 raster.
- `<basename>_hyeto.csv` — per-subbasin incremental precip per timestep.
- `<basename>_target.csv` — target-element flow series.
- `_iteration_results.html` — interactive results page (all hydrographs +
  storm thumbnails, grouped by frequency).
- `_comparison.html` — winner Hyeto scenario vs. gridded Atlas 14 baseline.

The two HTML pages are also mirrored into [presentation/](presentation/) so
they're reachable from the deployed methodology site (see
[render.yaml](render.yaml)).

## Project layout

| File | Role |
| --- | --- |
| [isohyet_maker.py](isohyet_maker.py) | Entry point; defines the sweep and per-scenario pipeline. |
| [basin.py](basin.py) | Shapefile loading, Atlas 14 raster sampling, subbasin centroid grid. |
| [hmr52.py](hmr52.py) | HMR52 elliptical depth-area-duration field. |
| [temporal.py](temporal.py) | Apply temporal distribution (default `Type II 24-hr`) to subbasin means. |
| [dss_io.py](dss_io.py) | Write per-subbasin hyetographs to DSS (HMS Specified Hyetograph format). |
| [hms_runner.py](hms_runner.py) | HEC-HMS batch invocation + results-DSS parsing + project snapshot/restore. |
| [results_viz.py](results_viz.py) | Build `_iteration_results.html`. |
| [compare_results.py](compare_results.py) | Build `_comparison.html` (winner vs. gridded Atlas 14 baseline). |
| [data/hmr52_dad_24hr.csv](data/hmr52_dad_24hr.csv) | HMR52 depth-area-duration curve. |
| [data/HydroCad Rainfall Temporal Distribtions.csv](data/HydroCad%20Rainfall%20Temporal%20Distribtions.csv) | Temporal distributions (incl. Type II 24-hr). |
