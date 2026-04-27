"""HEC-HMS batch automation: invoke a named simulation run via the HMS CLI's
Jython script entry point, then read the resulting flow at a target element
out of the run's results DSS file.

Iteration loop pattern:

    1. write_subbasin_hyetographs_dss(hyeto_df, HMS_HYETO_DSS, overwrite=True)
    2. run_hms(...)            -- HMS picks up the new precip via its existing
                                  Met Model gage references; no Met Model edits
                                  needed because the DSS pathnames are the same
                                  every iteration.
    3. read_target_hydrograph(results_dss, run_name, element_name) -> DataFrame
    4. df.to_csv(<scenario_basename>_target.csv)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pydsstools.heclib.dss.HecDss import Open

# --- Project-specific defaults; overridable per call ------------------------
HMS_CMD = Path(r"C:\Program Files\HEC\HEC-HMS\4.12\HEC-HMS.cmd")
HMS_PROJECT_DIR = Path(
    r"C:\Temp\I_57_Gage_Analysis_HMS4.12_AutoHyeto\I-57 Gage Analysis"
)
HMS_PROJECT_NAME = "I_57_Gage_Analysis"
HMS_HYETO_DSS = HMS_PROJECT_DIR / "data" / "hyeto.dss"

# Sibling directories for project snapshots used by snapshot/restore. The
# pristine backup is captured once at sweep start and reused as the source
# of truth for restoring a clean model. The last-failure copy preserves
# whatever HMS left behind on a failed compute, for forensics.
HMS_PRISTINE_BACKUP = HMS_PROJECT_DIR.with_name(HMS_PROJECT_DIR.name + "_pristine")
HMS_FAILURE_BACKUP = HMS_PROJECT_DIR.with_name(HMS_PROJECT_DIR.name + "_lastfailure")


# Subdirectory names skipped during snapshot/restore. ``terrain/`` holds
# tens of thousands of GeoTIFF tiles that HMS doesn't modify and that
# Windows often refuses to delete (file-indexer / AV contention), so we
# leave it in place on both ends.
HMS_SNAPSHOT_SKIP_DIRS: set[str] = {"terrain"}


def _safe_remove(path: Path) -> None:
    """Best-effort deletion of a file or directory; swallows PermissionError."""
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except PermissionError:
        pass


def snapshot_hms_project(
    dest: Path,
    project_dir: Path = HMS_PROJECT_DIR,
) -> None:
    """Snapshot the HMS project to ``dest``, excluding ``terrain/``.

    Captures every top-level file and every subdirectory other than the
    ones in ``HMS_SNAPSHOT_SKIP_DIRS`` (currently just ``terrain``). HMS
    corruption can affect both top-level config files (``.hms``, ``.basin``,
    ...) and files inside ``data/`` and ``results/``, so we need full coverage
    minus the static terrain rasters.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # Clear stale items in dest, but leave any pre-existing terrain alone.
    for item in dest.iterdir():
        if item.is_dir() and item.name in HMS_SNAPSHOT_SKIP_DIRS:
            continue
        _safe_remove(item)
    # Copy from project_dir, skipping terrain.
    for item in project_dir.iterdir():
        target = dest / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir() and item.name not in HMS_SNAPSHOT_SKIP_DIRS:
            shutil.copytree(item, target)


def restore_hms_project(
    src: Path,
    project_dir: Path = HMS_PROJECT_DIR,
) -> None:
    """Restore the HMS project from a snapshot at ``src``.

    Replaces every top-level file and every non-skipped subdirectory in
    ``project_dir`` with the snapshot's contents. The ``terrain/`` subdir
    in ``project_dir`` is left untouched (the snapshot is expected not to
    contain it; if it does, it's still skipped here). Callers that need
    ``data/hyeto.dss`` to reflect the current scenario must re-write it
    after restore.
    """
    if not src.exists():
        raise FileNotFoundError(f"HMS project snapshot not found at {src}")
    project_dir.mkdir(parents=True, exist_ok=True)
    # Clear project_dir except terrain.
    for item in project_dir.iterdir():
        if item.is_dir() and item.name in HMS_SNAPSHOT_SKIP_DIRS:
            continue
        _safe_remove(item)
    # Copy from snapshot, skipping terrain.
    for item in src.iterdir():
        target = project_dir / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir() and item.name not in HMS_SNAPSHOT_SKIP_DIRS:
            shutil.copytree(item, target)


def ensure_pristine_backup(
    project_dir: Path = HMS_PROJECT_DIR,
    pristine: Path = HMS_PRISTINE_BACKUP,
) -> bool:
    """Create a pristine backup of the HMS project if one doesn't already exist.

    Returns True if a new backup was created, False if an existing one was
    reused. To force a fresh capture (e.g. after manual HMS edits), delete
    ``HMS_PRISTINE_BACKUP`` before calling.
    """
    if pristine.exists():
        return False
    snapshot_hms_project(pristine, project_dir)
    return True


def _build_jython_script(
    project_name: str, project_dir: Path, run_name: str
) -> str:
    """Build the Jython source HMS will execute. Forward slashes work fine in Java.

    Notes:
      - We *don't* call ``Exit(...)`` so the JVM stays alive until the script
        actually finishes. Calling ``Exit()`` raises a SystemExit that can
        terminate the process before any compute work has flushed to disk.
      - The ``Compute`` call is wrapped in try/except: when HMS hits routing
        instability it raises a Jython ``ValueError`` here, which would
        otherwise propagate out and kill the JVM mid-shutdown. That ungraceful
        exit has been observed to leave the project's ``.basin`` file in a
        partial state (basin elements disappearing). Catching here lets HMS's
        shutdown finish cleanly so the on-disk model is preserved.
      - ``print`` lines go to subprocess stdout — useful for verifying the
        script executed end-to-end.
    """
    pdir = str(project_dir).replace("\\", "/")
    return (
        "from hms.model.JythonHms import *\n"
        "import time\n"
        f'print("HMS auto: opening project {project_name}")\n'
        f'OpenProject("{project_name}", "{pdir}")\n'
        f'print("HMS auto: computing {run_name}")\n'
        "try:\n"
        f'    Compute("{run_name}")\n'
        '    print("HMS auto: compute returned cleanly")\n'
        "except Exception as _e:\n"
        '    print("HMS auto: compute raised: " + str(_e))\n'
        # Hold the JVM open 8 s after Compute so any deferred DSS writes /
        # shutdown hooks can flush before the subprocess exits. The HMS GUI
        # gets this for free because the user closes the app some time after
        # compute; in batch mode without this delay the FLOW records sometimes
        # never land on disk for runs with many Muskingum routing warnings.
        'print("HMS auto: settling 8 s for DSS flush")\n'
        "time.sleep(8)\n"
        'print("HMS auto: settle complete")\n'
    )


def _clear_hms_run_cache(run_name: str, project_dir: Path) -> None:
    """Delete cached run artifacts so HMS is forced to actually recompute.

    HMS skips Compute when its <run_name>.results cache file is newer than
    its tracked inputs, which causes silent no-op iterations when we keep
    rewriting hyeto.dss between runs.
    """
    candidates = [
        project_dir / f"{run_name}.dss",
        project_dir / "results" / f"RUN_{run_name}.results",
        project_dir / f"{run_name}.out",
    ]
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except PermissionError:
            # File locked (HMS GUI open, DSSVue, etc.) — surface a clear error.
            raise RuntimeError(
                f"Cannot delete {p}; close HMS or any process holding it."
            )


def run_hms(
    run_name: str,
    project_name: str = HMS_PROJECT_NAME,
    project_dir: Path = HMS_PROJECT_DIR,
    hms_cmd: Path = HMS_CMD,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess:
    """Run an HMS simulation in batch mode (no GUI). Always forces recompute."""
    _clear_hms_run_cache(run_name, project_dir)
    script_text = _build_jython_script(project_name, project_dir, run_name)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jy", delete=False, encoding="utf-8"
    ) as f:
        f.write(script_text)
        script_path = f.name
    try:
        # HEC-HMS.cmd invokes ``jre\bin\java`` with a relative path, so it must
        # run from the HMS install directory or the JVM won't be found.
        return subprocess.run(
            [str(hms_cmd), "-s", script_path],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(hms_cmd.parent),
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def find_results_dss(
    run_name: str, project_dir: Path = HMS_PROJECT_DIR
) -> Path:
    """Locate the DSS file HMS writes for a named run.

    HMS 4.x writes results to <project_dir>/results/<run_name>.dss
    """
    return project_dir / f"{run_name}.dss"


def read_target_hydrograph(
    results_dss: Path,
    element_name: str,
    parameter: str = "FLOW",
    run_name: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Read a flow time series from an HMS results DSS file.

    Matches DSS pathnames part-by-part:
        /A=*  /B=element_name  /C=parameter  /D=*  /E=interval?  /F~=run_name?/
    The D-part (date range) is intentionally wildcarded so we don't need to
    know HMS's simulation window. The A-part is also unconstrained.

    Args:
        element_name: B-part — exact match against the HMS element name.
        parameter: C-part — exact match (case-insensitive). Default ``FLOW``.
        run_name: when provided, must appear as a substring of the F-part
            (HMS writes ``RUN:<run_name>`` there).
        interval: when provided, exact match against the E-part (e.g.
            ``"1Hour"``). Useful when the same element has multiple-resolution
            records.
    """
    with Open(str(results_dss)) as dss:
        # getPathnameDict groups paths by record type (TS, RTS, ITS, ...).
        # Flatten across types so we can match regardless.
        path_dict = dss.getPathnameDict()
        all_paths: list[str] = [p for plist in path_dict.values() for p in plist]
        candidates: list[str] = []
        for p in all_paths:
            # DSS pathname is /A/B/C/D/E/F/ which splits to 8 elements with
            # empty sentinels at index 0 and -1. We must NOT strip first,
            # because empty parts (e.g. blank A-part) collapse and shift the
            # indexing.
            parts = p.split("/")
            if len(parts) < 8:
                continue
            b, c, e, f = parts[2], parts[3], parts[5], parts[6]
            if b.lower() != element_name.lower():
                continue
            if c.lower() != parameter.lower():
                continue
            if interval is not None and e.lower() != interval.lower():
                continue
            if run_name is not None and run_name.lower() not in f.lower():
                continue
            candidates.append(p)
        if not candidates:
            sample = "\n  ".join(all_paths[:10])
            raise KeyError(
                f"No DSS path matches B={element_name!r} C={parameter!r}"
                f"{' E=' + interval if interval else ''}"
                f"{' F~=' + run_name if run_name else ''}.\n"
                f"First 10 paths in file:\n  {sample}"
            )
        # Each remaining candidate is a separate monthly block of the same
        # logical series (only the D-part differs). Read them all and stitch.
        chunks: list[pd.DataFrame] = []
        for path in candidates:
            ts = dss.read_ts(path)
            times = pd.to_datetime([
                f"{t.year:04d}-{t.month:02d}-{t.day:02d} "
                f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                for t in ts.pytimes
            ])
            chunks.append(pd.DataFrame(
                {"flow_cfs": np.array(ts.values, dtype=float)},
                index=pd.DatetimeIndex(times, name="time"),
            ))

    df = (
        pd.concat(chunks)
        .sort_index()
        .loc[lambda d: ~d.index.duplicated(keep="first")]
    )
    # HMS writes a missing-value sentinel (~ -3.4e38) for timestamps where it
    # had no input. Drop those rather than carrying garbage into the analysis.
    df = df[df["flow_cfs"] > -1e30]
    df["_path"] = candidates[0]
    return df


def hms_run_name(frequency_yr: int) -> str:
    """Run-name template used in the HMS project: ``<freq>yr_24hr_Hyeto``.

    Zero-padded to 3 digits — matches HMS's stored run name for sub-100-yr
    frequencies (e.g., ``050yr_24hr_Hyeto`` for 50-yr) and is unchanged for
    3-digit frequencies (100, 500).
    """
    return f"{frequency_yr:03d}yr_24hr_Hyeto"


# Watershed outlet element where peak flow is evaluated. HMS basin element name.
TARGET_ELEMENT = "Black River at Black Rock"


if __name__ == "__main__":
    RUN_NAME = hms_run_name(100)

    print(f"HMS CLI:        {HMS_CMD}")
    print(f"Project:        {HMS_PROJECT_NAME} @ {HMS_PROJECT_DIR}")
    print(f"Run name:       {RUN_NAME}")
    print(f"Target element: {TARGET_ELEMENT}")
    print()

    if not HMS_CMD.exists():
        raise SystemExit(f"HMS CLI not found at {HMS_CMD}")
    if not (HMS_PROJECT_DIR / f"{HMS_PROJECT_NAME}.hms").exists():
        raise SystemExit(
            f"HMS project file not found at {HMS_PROJECT_DIR / (HMS_PROJECT_NAME + '.hms')}"
        )

    print("Launching HMS (this may take 30-90 s for JVM startup + compute) ...")
    result = run_hms(RUN_NAME)
    print(f"HMS exit code: {result.returncode}")
    if result.stdout:
        print("--- HMS stdout (tail) ---")
        print("\n".join(result.stdout.splitlines()[-15:]))
    if result.stderr:
        print("--- HMS stderr (tail) ---")
        print("\n".join(result.stderr.splitlines()[-15:]))

    results_dss = find_results_dss(RUN_NAME)
    print(f"\nResults DSS: {results_dss}")
    if not results_dss.exists():
        raise SystemExit(f"Results DSS not found at {results_dss}")

    df = read_target_hydrograph(results_dss, TARGET_ELEMENT, run_name=RUN_NAME)
    print(f"\nTarget hydrograph: {len(df)} timesteps")
    print(f"  Path:          {df['_path'].iloc[0]}")
    print(f"  Time range:    {df.index[0]} -> {df.index[-1]}")
    print(f"  Peak:          {df['flow_cfs'].max():.0f} cfs at {df['flow_cfs'].idxmax()}")
