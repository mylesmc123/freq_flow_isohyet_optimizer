"""Build a single-page HTML visualization of an iteration sweep.

Inputs:
    - Manifest CSV (output of isohyet_maker.run_full_scenario sweeps) with
      metadata + per-timestep ``Q@<timestamp>`` flow columns + a
      ``storm_geojson_path`` reference for each scenario.

Outputs in <out_dir>:
    - ``_viz/storms/<idx>_<iteration_name>.png`` -- one PNG per scenario,
      basin outlines (red) over the elliptical storm pattern (viridis fill)
      with a centroid dot and a colored border that matches the scenario's
      hydrograph line on the chart above.
    - ``_iteration_results.html`` -- self-contained page (Chart.js via CDN)
      showing all hydrographs overlaid, plus a grid of storm thumbnails each
      labeled with iteration name + peak flow.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive; this exact placement worked previously

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from basin import WORKING_CRS, load_subbasins


def build_results_html(
    manifest_csv: Path,
    out_dir: Path,
    subbasin_shp: Path,
    name_col: str = "name",
    title: str = "Iteration Results",
    target_element: str | None = None,
) -> Path:
    """Render an HTML results page next to the manifest CSV.

    Scenarios are grouped by ``frequency_yr``; each frequency gets its own
    section (anchor + chart + storm grid + winner banner) and a TOC at the
    top links to each.
    """
    manifest = pd.read_csv(manifest_csv)
    if "iteration_name" not in manifest.columns:
        raise ValueError("manifest is missing the 'iteration_name' column")

    hyd_cols = [c for c in manifest.columns if c.startswith("Q@")]
    if not hyd_cols:
        raise ValueError("manifest has no hydrograph columns (Q@<timestamp>)")
    time_labels = [c.removeprefix("Q@") for c in hyd_cols]

    viz_dir = out_dir / "_viz"
    storms_dir = viz_dir / "storms"
    storms_dir.mkdir(parents=True, exist_ok=True)

    subs = load_subbasins(subbasin_shp, name_col=name_col)

    # Common map extent across all storms so thumbnails are visually comparable
    # across frequencies.
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for _, row in manifest.iterrows():
        gj = Path(row["storm_geojson_path"])
        if not gj.exists():
            continue
        bx_min, by_min, bx_max, by_max = (
            gpd.read_file(gj).to_crs(WORKING_CRS).total_bounds
        )
        minx, miny = min(minx, bx_min), min(miny, by_min)
        maxx, maxy = max(maxx, bx_max), max(maxy, by_max)
    pad = 5_000.0
    extent_x = (minx - pad, maxx + pad)
    extent_y = (miny - pad, maxy + pad)

    palette = plt.get_cmap("tab10")
    viridis = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0.0, vmax=10.0)

    # Group scenarios by frequency; colors restart per group so each section's
    # palette is internally distinct without 19+ near-duplicate hues.
    frequencies = sorted(manifest["frequency_yr"].unique())
    sections: list[dict] = []

    for freq in frequencies:
        sub = manifest[manifest["frequency_yr"] == freq].reset_index(drop=True)
        scenarios: list[dict] = []
        for j, row in sub.iterrows():
            color = mcolors.to_hex(palette(j % 10))
            gj_path = Path(row["storm_geojson_path"])
            gdf = gpd.read_file(gj_path).to_crs(WORKING_CRS)
            gdf = gdf.sort_values("area_sqmi", ascending=False)
            cx = float(gdf["centroid_x_m"].iloc[0])
            cy = float(gdf["centroid_y_m"].iloc[0])

            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            subs.boundary.plot(ax=ax, color="#dc2626", linewidth=0.55)
            for _, ring in gdf.iterrows():
                geom = ring.geometry
                if geom is None or geom.is_empty:
                    continue
                rcolor = viridis(norm(float(ring["depth_in"])))
                geoms = (
                    list(geom.geoms) if hasattr(geom, "geoms") else [geom]
                )
                for g in geoms:
                    if g.geom_type != "Polygon":
                        continue
                    x, y = g.exterior.xy
                    ax.fill(x, y, facecolor=rcolor, edgecolor="none", alpha=0.88)
                    for hole in g.interiors:
                        hx, hy = hole.xy
                        ax.fill(hx, hy, facecolor="white", edgecolor="none")
            ax.plot(cx, cy, "ko", markersize=4)
            ax.set_xlim(extent_x); ax.set_ylim(extent_y)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"{row['iteration_name']}\nPeak: {row['peak_cfs']:,.0f} cfs",
                fontsize=8, pad=2,
            )
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.5)
            fig.tight_layout(pad=0.3)
            thumb_path = storms_dir / f"{int(row['frequency_yr']):03d}yr_{j:03d}_{row['iteration_name']}.png"
            fig.savefig(thumb_path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            scenarios.append({
                "iteration_name": row["iteration_name"],
                "color": color,
                "peak_cfs": float(row["peak_cfs"]),
                "peak_time": str(row["peak_time"]),
                "centroid_id": str(row["centroid_id"]),
                "orientation_deg": float(row["orientation_deg"]),
                "thumbnail": thumb_path.relative_to(out_dir).as_posix(),
                "flow_cfs": [
                    float(row[c]) if pd.notna(row[c]) else None
                    for c in hyd_cols
                ],
            })

        winner = max(scenarios, key=lambda s: s["peak_cfs"])
        sections.append({
            "frequency_yr": int(freq),
            "anchor": f"freq-{int(freq)}",
            "chart_id": f"hydroChart-{int(freq)}",
            "scenarios": scenarios,
            "winner": winner,
        })

    payload = {"time_labels": time_labels, "sections": sections}
    payload_json = json.dumps(payload)

    html = _render_html(title, payload_json, sections, target_element)
    out_html = out_dir / "_iteration_results.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html


def _render_html(
    title: str,
    payload_json: str,
    sections: list[dict],
    target_element: str | None = None,
) -> str:
    """Render the results page. Single template, payload embedded inline."""
    n_total = sum(len(s["scenarios"]) for s in sections)
    target_clause = (
        f" &mdash; the outlet hydrograph is located at {target_element}."
        if target_element
        else " &mdash; outlet hydrographs and elliptical storm patterns."
    )
    toc_html = " &middot; ".join(
        f'<a href="#{s["anchor"]}">{s["frequency_yr"]}-yr ({len(s["scenarios"])})</a>'
        for s in sections
    )
    sections_html = "\n".join(
        f"""
<section id="{s['anchor']}">
  <h2>{s['frequency_yr']}-yr scenarios <span class="count">({len(s['scenarios'])})</span></h2>
  <div class="winner-banner" id="winner-{s['frequency_yr']}"></div>
  <h3>Outlet hydrographs</h3>
  <div class="chart-wrap"><canvas id="{s['chart_id']}"></canvas></div>
  <h3>Elliptical storm patterns</h3>
  <div class="storm-grid" id="grid-{s['frequency_yr']}"></div>
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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --accent: #144870;
    --rule: #d4dae2;
    --bg: #fafbfc;
    --text: #1f2937;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
                 Helvetica, Arial, sans-serif;
    color: var(--text); background: var(--bg); line-height: 1.5;
  }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 28px 24px 80px; }}
  h1 {{
    margin: 0 0 6px; font-size: 26px; color: var(--accent);
    border-bottom: 2px solid var(--accent); padding-bottom: 6px;
  }}
  .meta {{ color: #5b6574; font-size: 14px; margin: 0 0 26px; }}
  h2 {{
    margin: 36px 0 10px; font-size: 19px; color: var(--accent);
    border-left: 4px solid var(--accent); padding-left: 10px;
  }}
  .chart-wrap {{ position: relative; height: 420px; margin: 12px 0 28px; }}
  .storm-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
    margin: 12px 0;
  }}
  .storm-card {{
    border: 1px solid var(--rule); border-radius: 6px; padding: 10px;
    background: white; display: flex; flex-direction: column;
  }}
  .storm-card .swatch-row {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    font-size: 13px; color: #2b3543;
  }}
  .storm-card .swatch {{
    display: inline-block; width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #6b6b6b;
  }}
  .storm-card img {{
    width: 100%; height: auto; display: block; border-radius: 3px;
  }}
  .storm-card .peak {{
    margin-top: 6px; font-weight: 600; color: var(--accent);
  }}
  .storm-card .meta {{
    font-size: 12px; color: #4b5563; margin-top: 2px;
  }}
  .storm-card.winner {{
    border: 2px solid #fbbf24;
    background: #fef9c3;
    box-shadow: 0 0 0 1px #fbbf24;
    position: relative;
  }}
  .storm-card.winner::before {{
    content: "\2605 Highest peak";
    position: absolute;
    top: -10px; right: 10px;
    background: #fbbf24; color: #1f2937;
    padding: 2px 9px; border-radius: 11px;
    font-size: 11px; font-weight: 600;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
  }}
  .winner-banner {{
    background: #fef9c3; border: 1px solid #fbbf24; border-radius: 5px;
    padding: 10px 14px; margin: 12px 0 20px; font-size: 14px;
  }}
  h3 {{ margin: 22px 0 6px; font-size: 15px; color: #2b3543; }}
  nav.toc {{
    margin: 8px 0 24px; padding: 10px 14px;
    background: #eef3f8; border: 1px solid #cdd7e3; border-radius: 5px;
    font-size: 14px;
  }}
  nav.toc strong {{ color: var(--accent); margin-right: 8px; }}
  nav.toc a {{ color: var(--accent); text-decoration: none; }}
  nav.toc a:hover {{ text-decoration: underline; }}
  section {{ scroll-margin-top: 12px; }}
  .count {{ color: #6b7280; font-weight: 400; font-size: 0.85em; }}
</style>
</head>
<body>
<main>

<h1>{title}</h1>
<div class="meta">{n_total} scenario{'' if n_total == 1 else 's'} swept across {len(sections)} frequenc{'y' if len(sections) == 1 else 'ies'}{target_clause}</div>

<nav class="toc"><strong>Jump to:</strong> {toc_html}</nav>

{sections_html}

<script>
const PAYLOAD = {payload_json};

(function () {{
  for (const sec of PAYLOAD.sections) {{
    // Hydrograph chart for this frequency
    const ctx = document.getElementById(`hydroChart-${{sec.frequency_yr}}`);
    new Chart(ctx, {{
      type: "line",
      data: {{
        labels: PAYLOAD.time_labels,
        datasets: sec.scenarios.map((s) => ({{
          label: `${{s.iteration_name}}  (peak ${{s.peak_cfs.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} cfs)`,
          data: s.flow_cfs,
          borderColor: s.color,
          backgroundColor: "transparent",
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.2,
        }})),
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        interaction: {{ mode: "nearest", intersect: false }},
        plugins: {{
          title: {{ display: true, text: `${{sec.frequency_yr}}-yr outfall flow vs. time` }},
          legend: {{ position: "bottom", labels: {{ boxWidth: 14, font: {{ size: 11 }} }} }},
          tooltip: {{ mode: "nearest", intersect: false }},
        }},
        scales: {{
          x: {{ title: {{ display: true, text: "Time" }}, ticks: {{ maxTicksLimit: 16 }} }},
          y: {{ title: {{ display: true, text: "Flow (cfs)" }}, beginAtZero: true }},
        }},
      }},
    }});

    // Winner banner for this frequency
    let winner = sec.scenarios[0];
    for (const s of sec.scenarios) {{
      if (s.peak_cfs > winner.peak_cfs) winner = s;
    }}
    document.getElementById(`winner-${{sec.frequency_yr}}`).innerHTML =
      `<strong>Highest peak:</strong> ` +
      `${{winner.peak_cfs.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} cfs ` +
      `&mdash; ${{winner.iteration_name}} ` +
      `(${{winner.centroid_id}}, ${{winner.orientation_deg}}&deg;) ` +
      `at ${{winner.peak_time}}.`;

    // Storm grid for this frequency — winner gets a highlighted border
    const grid = document.getElementById(`grid-${{sec.frequency_yr}}`);
    for (const s of sec.scenarios) {{
      const card = document.createElement("div");
      const isWinner = s.iteration_name === winner.iteration_name;
      card.className = isWinner ? "storm-card winner" : "storm-card";
      card.innerHTML = `
        <div class="swatch-row">
          <span class="swatch" style="background:${{s.color}}"></span>
          <span>${{s.iteration_name}}</span>
        </div>
        <img src="${{s.thumbnail}}" alt="${{s.iteration_name}}">
        <div class="peak">Peak: ${{s.peak_cfs.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} cfs</div>
        <div class="meta">${{s.centroid_id}} &middot; ${{s.orientation_deg}}&deg;</div>
        <div class="meta">at ${{s.peak_time}}</div>
      `;
      grid.appendChild(card);
    }}
  }}
}})();
</script>

</main>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    from hms_runner import TARGET_ELEMENT

    if len(sys.argv) > 1:
        manifest = Path(sys.argv[1])
    else:
        manifest = Path(__file__).parent / "output" / "I57" / "_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")
    out = build_results_html(
        manifest_csv=manifest,
        out_dir=manifest.parent,
        subbasin_shp=Path(__file__).parent / "input" / "I57" / "I-57_HMS_Subbasins.shp",
        target_element=TARGET_ELEMENT,
    )
    print(f"Wrote {out}")
