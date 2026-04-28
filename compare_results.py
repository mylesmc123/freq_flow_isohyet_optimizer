"""Compare per-frequency winners (Hyeto-driven) against the Atlas-14-everywhere
gridded-precip baseline runs that already exist in the HMS project.

Per frequency the page shows:
  - One hydrograph chart with two lines: baseline (gridded Atlas 14 applied
    everywhere) vs. winning Hyeto scenario (Atlas 14 magnitude at the storm
    centroid + HMR52 spatial decay).
  - Two side-by-side Leaflet maps over an OSM basemap with the basin outlines
    overlaid: the Atlas 14 raster (left) vs. the winning elliptical storm
    pattern (right).

The baseline runs are NOT recomputed here — we only read their existing
results DSS files (``<project_dir>/<freq>yr_24hr.dss``).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from basin import WORKING_CRS, load_subbasins
from hms_runner import HMS_PROJECT_DIR, TARGET_ELEMENT, read_target_hydrograph


def _baseline_run_name(frequency_yr: int) -> str:
    """Name HMS used for the gridded-Atlas-14 baseline run, e.g. ``050yr_24hr``."""
    return f"{frequency_yr:03d}yr_24hr"


def _cmap_stops(
    cmap_name: str, vmin: float, vmax: float, n_steps: int = 50
) -> list[dict]:
    """Discretize a matplotlib colormap into ``n_steps + 1`` hex stops for JS use."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    stops = []
    for i in range(n_steps + 1):
        v = vmin + (vmax - vmin) * i / n_steps
        r, g, b, _ = cmap(norm(v))
        stops.append({
            "value": round(v, 3),
            "color": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
        })
    return stops


def _render_atlas14_png(
    asc_path: Path,
    png_path: Path,
    json_path: Path,
    cmap_name: str = "viridis",
) -> None:
    """Render a NOAA Atlas 14 ESRI ASCII grid as a viridis PNG with bounds metadata.

    The PNG is written with vmin/vmax tightened to the data range so the
    color spread covers the whole basin variability. ``json_path`` records
    the lat/lon bounds (for Leaflet ``imageOverlay``) plus the colormap.
    """
    with rasterio.open(asc_path) as src:
        raw = src.read(1).astype(float)
        nd = src.nodata
        inches = raw * 0.001
        if nd is not None:
            mask = raw == nd
        else:
            mask = np.zeros_like(raw, dtype=bool)
        inches[mask] = np.nan
        bounds = src.bounds

    valid = inches[~np.isnan(inches)]
    vmin, vmax = float(valid.min()), float(valid.max())
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.where(np.isnan(inches), vmin, inches)))
    rgba[..., 3] = np.where(np.isnan(inches), 0.0, 0.85)
    img = (rgba * 255).astype(np.uint8)
    plt.imsave(png_path, img)

    n_steps = 50
    stops = []
    for i in range(n_steps + 1):
        v = vmin + (vmax - vmin) * i / n_steps
        r, g, b, _ = cmap(norm(v))
        stops.append({
            "value": round(v, 3),
            "color": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
        })
    json_path.write_text(json.dumps({
        "bounds_latlon": [bounds.bottom, bounds.left, bounds.top, bounds.right],
        "vmin": vmin, "vmax": vmax,
        # Aliases for the methodology-page schema, which reads ``data_min_in``
        # and ``data_max_in`` for the "X-Y in across the basin" caption.
        "data_min_in": vmin, "data_max_in": vmax,
        "cmap": cmap_name,
        "stops": stops,
    }, indent=2))


def build_comparison_html(
    manifest_csv: Path,
    out_dir: Path,
    input_dir: Path,
    subbasin_shp: Path,
    name_col: str = "name",
    title: str = "Hyetograph Winner vs. Gridded Atlas 14 Baseline",
) -> Path:
    """Build the comparison HTML in ``out_dir``.

    Atlas 14 PNGs and metadata files are written into
    ``out_dir/example/data/`` next to the HTML so the page works locally via
    relative ``example/data/...`` URLs. The caller is responsible for
    mirroring the HTML and the asset folder to a deploy directory if needed.

    The baseline DSS files must already exist at
    ``<HMS_PROJECT_DIR>/<freq>yr_24hr.dss``.
    """
    manifest = pd.read_csv(manifest_csv)
    if "iteration_name" not in manifest.columns:
        raise ValueError("manifest is missing 'iteration_name'")
    hyd_cols = [c for c in manifest.columns if c.startswith("Q@")]
    if not hyd_cols:
        raise ValueError("manifest has no hydrograph columns (Q@<timestamp>)")

    winner_times = pd.to_datetime([c.removeprefix("Q@") for c in hyd_cols])

    # Subbasins -> GeoJSON (EPSG:4326) for inline embedding in the HTML.
    subs_4326 = load_subbasins(subbasin_shp, name_col=name_col).to_crs("EPSG:4326")
    subbasins_geojson = json.loads(subs_4326.to_json())

    asset_dir = out_dir / "example" / "data"
    asset_dir.mkdir(parents=True, exist_ok=True)

    frequencies = sorted(manifest["frequency_yr"].unique())
    sections: list[dict] = []

    for freq in frequencies:
        sub = manifest[manifest["frequency_yr"] == freq]
        if sub.empty:
            continue
        winner_idx = sub["peak_cfs"].idxmax()
        winner = sub.loc[winner_idx]

        winner_flow = pd.Series(
            [float(winner[c]) if pd.notna(winner[c]) else np.nan for c in hyd_cols],
            index=winner_times,
        )

        baseline_run = _baseline_run_name(int(freq))
        baseline_dss = HMS_PROJECT_DIR / f"{baseline_run}.dss"
        if not baseline_dss.exists():
            print(f"  [warn] missing baseline DSS for {freq}-yr at {baseline_dss}; skipping")
            continue
        baseline_df = read_target_hydrograph(
            baseline_dss, TARGET_ELEMENT, run_name=baseline_run
        )
        baseline_flow = baseline_df["flow_cfs"]

        # Combined time axis across both series.
        combined_idx = winner_flow.index.union(baseline_flow.index).sort_values()
        winner_aligned = winner_flow.reindex(combined_idx)
        baseline_aligned = baseline_flow.reindex(combined_idx)

        # Atlas 14 PNG + metadata (write to presentation asset dir).
        atlas_png = asset_dir / f"atlas14_{freq:03d}yr_24hr.png"
        atlas_json = asset_dir / f"atlas14_{freq:03d}yr_24hr.json"
        if not atlas_png.exists() or not atlas_json.exists():
            asc = input_dir / f"I57_{freq:03d}yr24ha.asc"
            if not asc.exists():
                print(f"  [warn] Atlas 14 raster missing for {freq}-yr at {asc}; skipping map")
                continue
            _render_atlas14_png(asc, atlas_png, atlas_json)
        atlas_meta = json.loads(atlas_json.read_text())

        # Inline the winner storm GeoJSON so the page is self-contained.
        storm_path = Path(winner["storm_geojson_path"])
        storm_geojson = json.loads(storm_path.read_text())

        # Storm gets its own colormap (magma) tuned to its own depth range so
        # the storm map is visually distinct from the Atlas 14 raster (viridis)
        # and uses the full dynamic range of its color scale.
        storm_depths = [
            float(f["properties"]["depth_in"])
            for f in storm_geojson.get("features", [])
            if f.get("properties", {}).get("depth_in") is not None
        ]
        if storm_depths:
            storm_vmin, storm_vmax = 0.0, max(storm_depths)
        else:
            storm_vmin, storm_vmax = 0.0, 10.0
        storm_meta = {
            "vmin": storm_vmin,
            "vmax": storm_vmax,
            "cmap": "magma",
            "stops": _cmap_stops("magma", storm_vmin, storm_vmax),
        }

        baseline_peak_cfs = float(baseline_aligned.dropna().max())
        baseline_peak_time = (
            baseline_aligned.dropna().idxmax().isoformat()
            if not baseline_aligned.dropna().empty else ""
        )

        sections.append({
            "frequency_yr": int(freq),
            "anchor": f"freq-{int(freq)}",
            "winner_iteration_name": str(winner["iteration_name"]),
            "winner_centroid_id": str(winner["centroid_id"]),
            "winner_orientation_deg": float(winner["orientation_deg"]),
            "winner_peak_cfs": float(winner["peak_cfs"]),
            "winner_peak_time": str(winner["peak_time"]),
            "baseline_peak_cfs": baseline_peak_cfs,
            "baseline_peak_time": baseline_peak_time,
            "time_labels": [t.isoformat() for t in combined_idx],
            "baseline_flow_cfs": [None if pd.isna(v) else float(v)
                                  for v in baseline_aligned.tolist()],
            "winner_flow_cfs": [None if pd.isna(v) else float(v)
                                for v in winner_aligned.tolist()],
            "atlas14_png": f"example/data/{atlas_png.name}",
            "atlas14_meta": atlas_meta,
            "storm_geojson": storm_geojson,
            "storm_meta": storm_meta,
        })

    payload = {
        "subbasins_geojson": subbasins_geojson,
        "sections": sections,
    }

    html = _render_html(title, json.dumps(payload), sections)
    out_html = out_dir / "_comparison_results.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html


def _render_html(title: str, payload_json: str, sections: list[dict]) -> str:
    toc_html = " &middot; ".join(
        f'<a href="#{s["anchor"]}">{s["frequency_yr"]}-yr</a>'
        for s in sections
    )
    sections_html = "\n".join(
        f"""
<section id="{s['anchor']}">
  <h2>{s['frequency_yr']}-yr</h2>
  <div class="banner" id="banner-{s['frequency_yr']}"></div>
  <h3>Outlet hydrographs</h3>
  <div class="chart-wrap"><canvas id="chart-{s['frequency_yr']}"></canvas></div>
  <h3>Storm spatial patterns</h3>
  <div class="map-row">
    <figure>
      <figcaption><strong>Atlas 14 grid</strong> &mdash; baseline (Atlas-14-everywhere)</figcaption>
      <div id="map-{s['frequency_yr']}-baseline" class="mini-map"></div>
    </figure>
    <figure>
      <figcaption><strong>Elliptical storm</strong> &mdash; winning Hyeto scenario ({s['winner_iteration_name']})</figcaption>
      <div id="map-{s['frequency_yr']}-storm" class="mini-map"></div>
    </figure>
  </div>
</section>
"""
        for s in sections
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{ --accent:#144870; --rule:#d4dae2; --bg:#fafbfc; --text:#1f2937; }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin:0; padding:0; color:var(--text); background:var(--bg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    line-height:1.5;
  }}
  main {{ max-width:1200px; margin:0 auto; padding:28px 24px 80px; }}
  h1 {{
    margin:0 0 6px; font-size:26px; color:var(--accent);
    border-bottom:2px solid var(--accent); padding-bottom:6px;
  }}
  .meta {{ color:#5b6574; font-size:14px; margin:0 0 24px; }}
  h2 {{
    margin:36px 0 10px; font-size:19px; color:var(--accent);
    border-left:4px solid var(--accent); padding-left:10px;
  }}
  h3 {{ margin:22px 0 6px; font-size:15px; color:#2b3543; }}
  nav.toc {{
    margin:8px 0 24px; padding:10px 14px;
    background:#eef3f8; border:1px solid #cdd7e3; border-radius:5px;
    font-size:14px;
  }}
  nav.toc strong {{ color:var(--accent); margin-right:8px; }}
  nav.toc a {{ color:var(--accent); text-decoration:none; }}
  nav.toc a:hover {{ text-decoration:underline; }}
  .page-nav {{
    margin:12px 0; padding:8px 14px;
    background:#eef3f8; border:1px solid #cdd7e3; border-radius:5px;
    font-size:14px; text-align:center;
  }}
  .page-nav a {{ color:var(--accent); text-decoration:none; font-weight:500; }}
  .page-nav a:hover {{ text-decoration:underline; }}
  .page-nav .current {{ color:#6b7280; font-weight:500; }}
  .page-nav .sep {{ color:#9aa6b4; margin:0 6px; }}
  section {{ scroll-margin-top: 12px; }}
  .banner {{
    background:#eff6ff; border:1px solid #93c5fd; border-radius:5px;
    padding:10px 14px; margin:8px 0 18px; font-size:14px;
  }}
  .banner strong {{ color:var(--accent); }}
  .chart-wrap {{ position:relative; height:380px; margin:10px 0 24px; }}
  .map-row {{
    display:grid; grid-template-columns:1fr 1fr; gap:16px;
    margin:10px 0 28px;
  }}
  .map-row figure {{ margin:0; }}
  .map-row figcaption {{
    font-size:13px; color:#2b3543; margin-bottom:6px;
  }}
  .mini-map {{
    height:380px; border:1px solid #b3bcc7; border-radius:4px; background:#fff;
  }}
  .legend {{
    background:white; padding:8px 12px; line-height:1.4;
    border:1px solid #999; border-radius:4px;
    box-shadow:0 1px 4px rgba(0,0,0,0.18); font-size:12px;
  }}
  .legend .row {{ display:flex; align-items:center; margin:2px 0; }}
  .legend .swatch {{
    display:inline-block; width:22px; height:14px;
    margin-right:6px; border:1px solid #6b6b6b;
  }}
  .legend .title {{ font-weight:600; margin-bottom:4px; font-size:12px; }}
  @media (max-width: 720px) {{
    .map-row {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>
<main>

<nav class="page-nav">
  <a href="./index.html">Methodology</a>
  <span class="sep">&middot;</span>
  <a href="./_iteration_results.html">Iteration Results</a>
  <span class="sep">&middot;</span>
  <span class="current">Winner vs. Baseline</span>
</nav>

<h1>{title}</h1>
<div class="meta">For each frequency, the Hyeto-method winner (Atlas&nbsp;14 magnitude + HMR&nbsp;52 spatial decay) is plotted alongside the Atlas-14-everywhere baseline, with side-by-side maps of the two storm spatial patterns.</div>

<nav class="toc"><strong>Jump to:</strong> {toc_html}</nav>

{sections_html}

<script>
const PAYLOAD = {payload_json};
// interactive:false so the subbasin layer doesn't capture mouse events when
// drawn on top of the storm rings — tooltips on the storm bands stay reachable.
const SUBBASIN_STYLE = {{ color:"#dc2626", weight:1.4, fillOpacity:0, interactive:false }};

function depthToColor(depth, cmap) {{
  if (depth == null || isNaN(depth)) return "#cccccc";
  const t = Math.max(0, Math.min(1, (depth - cmap.vmin) / (cmap.vmax - cmap.vmin)));
  const idx = Math.round(t * (cmap.stops.length - 1));
  return cmap.stops[idx].color;
}}

function buildLegend(cmap, title, ticks) {{
  const el = document.createElement("div");
  el.className = "legend";
  let html = `<div class="title">${{title}}</div>`;
  for (const t of ticks) {{
    html += `<div class="row"><span class="swatch" style="background:${{depthToColor(t, cmap)}}"></span><span>${{t.toFixed(1)}} in</span></div>`;
  }}
  el.innerHTML = html;
  return el;
}}

(function () {{
  for (const sec of PAYLOAD.sections) {{
    // ---- Hydrograph chart ----
    const winnerLabel = `Winner — ${{sec.winner_iteration_name}} (peak ${{sec.winner_peak_cfs.toLocaleString(undefined, {{maximumFractionDigits:0}})}} cfs)`;
    const baselineLabel = `Baseline (Atlas-14-everywhere) — peak ${{sec.baseline_peak_cfs.toLocaleString(undefined, {{maximumFractionDigits:0}})}} cfs`;
    new Chart(document.getElementById(`chart-${{sec.frequency_yr}}`), {{
      type: "line",
      data: {{
        labels: sec.time_labels,
        datasets: [
          {{
            label: baselineLabel,
            data: sec.baseline_flow_cfs,
            borderColor: "#3949ab",
            backgroundColor: "transparent",
            borderWidth: 2.0, pointRadius: 0, tension: 0.2,
          }},
          {{
            label: winnerLabel,
            data: sec.winner_flow_cfs,
            borderColor: "#d81b60",
            backgroundColor: "transparent",
            borderWidth: 2.0, pointRadius: 0, tension: 0.2,
          }},
        ],
      }},
      options: {{
        responsive:true, maintainAspectRatio:false,
        interaction: {{ mode:"nearest", intersect:false }},
        plugins: {{
          title: {{ display:true, text: `${{sec.frequency_yr}}-yr outfall flow vs. time` }},
          legend: {{ position:"bottom", labels:{{ boxWidth:14, font:{{size:11}} }} }},
          tooltip: {{ mode:"nearest", intersect:false }},
        }},
        scales: {{
          x: {{ title:{{display:true,text:"Time"}}, ticks:{{maxTicksLimit:14}} }},
          y: {{ title:{{display:true,text:"Flow (cfs)"}}, beginAtZero:true }},
        }},
      }},
    }});

    // ---- Banner ----
    const banner = document.getElementById(`banner-${{sec.frequency_yr}}`);
    const wp = sec.winner_peak_cfs, bp = sec.baseline_peak_cfs;
    const delta = wp - bp;
    const pct = bp ? (delta / bp * 100) : 0;
    const sign = delta >= 0 ? "+" : "";
    banner.innerHTML =
      `<strong>Baseline:</strong> ${{bp.toLocaleString(undefined,{{maximumFractionDigits:0}})}} cfs &nbsp;&middot;&nbsp; ` +
      `<strong>Winner:</strong> ${{wp.toLocaleString(undefined,{{maximumFractionDigits:0}})}} cfs ` +
      `(${{sign}}${{delta.toLocaleString(undefined,{{maximumFractionDigits:0}})}} cfs / ${{sign}}${{pct.toFixed(1)}}%) ` +
      `&mdash; centroid ${{sec.winner_centroid_id}}, ${{sec.winner_orientation_deg}}&deg;`;

    const am = sec.atlas14_meta;
    const [s1,w1,n1,e1] = am.bounds_latlon;
    const atlasBounds = [[s1,w1],[n1,e1]];

    // ---- Map: Atlas 14 baseline ----
    // Set the view BEFORE adding any vector layers; otherwise Leaflet's
    // _clipPoints crashes when GeoJSON layers try to render against an
    // unset pixel-bounds map.
    const map1 = L.map(`map-${{sec.frequency_yr}}-baseline`, {{ scrollWheelZoom:false }})
      .fitBounds(atlasBounds);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom:18,
      attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }}).addTo(map1);
    L.imageOverlay(sec.atlas14_png, atlasBounds, {{ opacity:0.85 }}).addTo(map1);
    const subsLayer1 = L.geoJSON(PAYLOAD.subbasins_geojson, {{ style: SUBBASIN_STYLE }}).addTo(map1);
    map1.fitBounds(subsLayer1.getBounds(), {{ padding:[16,16] }});
    const legend1 = L.control({{ position:"bottomright" }});
    const ticks = [am.vmin, am.vmin + (am.vmax-am.vmin)*0.25, am.vmin + (am.vmax-am.vmin)*0.5, am.vmin + (am.vmax-am.vmin)*0.75, am.vmax];
    legend1.onAdd = () => buildLegend(am, `Atlas 14, ${{sec.frequency_yr}}-yr 24-hr (in)`, ticks);
    legend1.addTo(map1);

    // ---- Map: winning elliptical storm ----
    const map2 = L.map(`map-${{sec.frequency_yr}}-storm`, {{ scrollWheelZoom:false }})
      .fitBounds(atlasBounds);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom:18,
      attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }}).addTo(map2);
    // Sort largest-area first so smaller bands paint on top.
    const stormSorted = {{
      type:"FeatureCollection",
      features: sec.storm_geojson.features.slice().sort(
        (a,b) => b.properties.area_sqmi - a.properties.area_sqmi
      ),
    }};
    const sm = sec.storm_meta;
    L.geoJSON(stormSorted, {{
      style: (f) => ({{
        color:"#1f2937", weight:0.6,
        fillColor: depthToColor(f.properties.depth_in, sm),
        fillOpacity:0.85,
      }}),
      onEachFeature: (f, layer) => {{
        const p = f.properties;
        layer.bindTooltip(`${{p.depth_in.toFixed(2)}} in &middot; band ${{p.inner_area_sqmi ?? 0}}-${{p.area_sqmi}} sq mi`, {{ sticky:true }});
      }},
    }}).addTo(map2);
    const subsLayer2 = L.geoJSON(PAYLOAD.subbasins_geojson, {{ style: SUBBASIN_STYLE }}).addTo(map2);
    map2.fitBounds(subsLayer2.getBounds(), {{ padding:[16,16] }});
    const stormTicks = [
      sm.vmin,
      sm.vmin + (sm.vmax - sm.vmin) * 0.25,
      sm.vmin + (sm.vmax - sm.vmin) * 0.50,
      sm.vmin + (sm.vmax - sm.vmin) * 0.75,
      sm.vmax,
    ];
    const legend2 = L.control({{ position:"bottomright" }});
    legend2.onAdd = () => buildLegend(sm, `Storm depth (magma), ${{sec.frequency_yr}}-yr 24-hr (in)`, stormTicks);
    legend2.addTo(map2);
  }}
}})();
</script>

<nav class="page-nav">
  <a href="./index.html">Methodology</a>
  <span class="sep">&middot;</span>
  <a href="./_iteration_results.html">Iteration Results</a>
  <span class="sep">&middot;</span>
  <span class="current">Winner vs. Baseline</span>
</nav>

</main>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    PROJ = Path(__file__).parent
    if len(sys.argv) > 1:
        manifest = Path(sys.argv[1])
    else:
        manifest = PROJ / "output" / "I57" / "_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")
    out = build_comparison_html(
        manifest_csv=manifest,
        out_dir=manifest.parent,
        input_dir=PROJ / "input" / "I57",
        subbasin_shp=PROJ / "input" / "I57" / "I-57_HMS_Subbasins.shp",
    )
    print(f"Wrote {out}")
