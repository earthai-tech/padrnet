"""01_africa_data_audit.py
========================
Audit F:\\_DATA\\FloodData for all data layers relevant to the three
Africa study sub-regions used in the Mathematical Geosciences paper.

Outputs
-------
tables/data_audit_africa.csv   - layer × attribute availability matrix
results/data_audit_africa.json - machine-readable summary with file sizes
                                  and row counts for key tables

Run
---
    python scripts/01_africa_data_audit.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# ---------- project imports --------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA_ROOT, RAW_DIR, INTERIM_DIR,
    EMDAT_FILE, GFD_META, GFD_SUMMARY, ERA5_DIR, TOPO_DIR, SOIL_DIR,
    HARMONISED_EVENTS,
    AFRICA_REGIONS, ALL_AFRICA_ISO,
    TABLES_DIR, RESULTS_DIR,
    print_banner, print_rule, timestamp,
)


# =============================================================================
# helpers
# =============================================================================

def _size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / 1e6, 2)
    except FileNotFoundError:
        return 0.0


def _dir_file_count(d: Path, suffix: str = "") -> int:
    if not d.is_dir():
        return 0
    if suffix:
        return sum(1 for _ in d.rglob(f"*{suffix}"))
    return sum(1 for _ in d.rglob("*") if _.is_file())


def audit_emdat(path: Path) -> dict:
    """Check EM-DAT file and count Africa flood events."""
    result: dict = {"path": str(path), "exists": path.exists(), "size_mb": _size_mb(path)}
    if not path.exists():
        result["africa_flood_rows"] = 0
        result["note"] = "FILE MISSING"
        return result
    try:
        df = pd.read_excel(path, sheet_name=0)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        # flexible column name matching
        iso_col = next((c for c in df.columns if "iso" in c or "country_code" in c), None)
        type_col = next((c for c in df.columns if "disaster_type" in c or "type" in c), None)
        if iso_col and type_col:
            mask = df[iso_col].isin(ALL_AFRICA_ISO) & (
                df[type_col].str.lower().str.contains("flood", na=False)
            )
            result["africa_flood_rows"] = int(mask.sum())
            result["total_rows"] = len(df)
            result["columns"] = list(df.columns[:10])   # first 10 for inspection
        else:
            result["africa_flood_rows"] = "unknown - cols not found"
            result["columns"] = list(df.columns)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def audit_gfd(meta_path: Path, summary_path: Path) -> dict:
    """Check Global Flood Database files."""
    result: dict = {
        "meta_exists": meta_path.exists(),
        "summary_exists": summary_path.exists(),
    }
    if meta_path.exists():
        result["meta_size_mb"] = _size_mb(meta_path)
        with open(meta_path) as fh:
            meta = json.load(fh)
        result["meta_events"] = len(meta) if isinstance(meta, list) else "dict"
    if summary_path.exists():
        result["summary_size_mb"] = _size_mb(summary_path)
        df = pd.read_csv(summary_path)
        # check for Africa region rows
        africa_keys = set(AFRICA_REGIONS.keys())
        region_col = next((c for c in df.columns if "region" in c.lower()), None)
        if region_col:
            africa_rows = df[df[region_col].isin(africa_keys)]
            result["africa_region_rows"] = len(africa_rows)
        result["total_rows"] = len(df)
        result["columns"] = list(df.columns)
    return result


def audit_harmonised_events(path: Path) -> dict:
    """Check the pre-built harmonised flood-event table."""
    result: dict = {"path": str(path), "exists": path.exists(), "size_mb": _size_mb(path)}
    if not path.exists():
        result["note"] = "FILE MISSING - run build_flood_event_table.py first"
        return result
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    result["total_rows"] = len(df)
    result["columns"] = list(df.columns)
    # Africa subset
    region_col = next((c for c in df.columns if "region" in c), None)
    if region_col:
        africa_rows = df[df[region_col].isin(set(AFRICA_REGIONS.keys()))]
        result["africa_rows"] = len(africa_rows)
        result["africa_year_range"] = (
            [int(africa_rows["year"].min()), int(africa_rows["year"].max())]
            if "year" in africa_rows.columns and len(africa_rows) > 0
            else "n/a"
        )
        result["africa_region_counts"] = (
            africa_rows[region_col].value_counts().to_dict()
            if len(africa_rows) > 0 else {}
        )
    return result


def audit_era5(era5_dir: Path) -> dict:
    """Summarise available ERA5 files."""
    result: dict = {"path": str(era5_dir), "exists": era5_dir.exists()}
    if not era5_dir.is_dir():
        result["note"] = "DIRECTORY MISSING"
        return result
    nc_files = list(era5_dir.rglob("*.nc"))
    zip_files = list(era5_dir.rglob("*.zip"))
    result["nc_files"] = len(nc_files)
    result["zip_files"] = len(zip_files)
    result["total_size_mb"] = round(sum(_size_mb(f) for f in nc_files + zip_files), 1)
    result["sample_files"] = [f.name for f in (nc_files + zip_files)[:5]]
    return result


def audit_topography(topo_dir: Path) -> dict:
    result: dict = {"path": str(topo_dir), "exists": topo_dir.is_dir()}
    if not topo_dir.is_dir():
        return result
    tif_files = list(topo_dir.rglob("*.tif")) + list(topo_dir.rglob("*.tiff"))
    result["tif_files"] = len(tif_files)
    result["total_size_mb"] = round(sum(_size_mb(f) for f in tif_files), 1)
    result["sample_files"] = [f.name for f in tif_files[:5]]
    return result


def audit_soil(soil_dir: Path) -> dict:
    result: dict = {"path": str(soil_dir), "exists": soil_dir.is_dir()}
    if not soil_dir.is_dir():
        return result
    files = list(soil_dir.rglob("*.*"))
    result["total_files"] = len(files)
    result["total_size_mb"] = round(sum(_size_mb(f) for f in files), 1)
    return result


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print_banner("01 -- Africa Data Audit")
    print(f"Timestamp : {timestamp()}")
    print(f"DATA_ROOT : {DATA_ROOT}  (exists={DATA_ROOT.exists()})\n")

    audit: dict = {
        "timestamp": timestamp(),
        "data_root": str(DATA_ROOT),
        "africa_regions": list(AFRICA_REGIONS.keys()),
        "layers": {},
    }

    # ---- EM-DAT -------------------------------------------------------
    print("Auditing EM-DAT ... ", end="", flush=True)
    audit["layers"]["emdat"] = audit_emdat(EMDAT_FILE)
    print("done")

    # ---- GFD ----------------------------------------------------------
    print("Auditing Global Flood Database ... ", end="", flush=True)
    audit["layers"]["gfd"] = audit_gfd(GFD_META, GFD_SUMMARY)
    print("done")

    # ---- Harmonised events --------------------------------------------
    print("Auditing harmonised event table ... ", end="", flush=True)
    audit["layers"]["harmonised_events"] = audit_harmonised_events(HARMONISED_EVENTS)
    print("done")

    # ---- ERA5 ---------------------------------------------------------
    print("Auditing ERA5 reanalysis ... ", end="", flush=True)
    audit["layers"]["era5"] = audit_era5(ERA5_DIR)
    print("done")

    # ---- Topography ---------------------------------------------------
    print("Auditing topography ... ", end="", flush=True)
    audit["layers"]["topography"] = audit_topography(TOPO_DIR)
    print("done")

    # ---- Soil moisture ------------------------------------------------
    print("Auditing soil moisture ... ", end="", flush=True)
    audit["layers"]["soil_moisture"] = audit_soil(SOIL_DIR)
    print("done")

    # ---- Summary table ------------------------------------------------
    rows = []
    for layer, info in audit["layers"].items():
        rows.append({
            "layer": layer,
            "exists": info.get("exists", info.get("meta_exists", "?")),
            "size_mb": info.get("size_mb", info.get("total_size_mb", 0)),
            "africa_rows": info.get("africa_rows", info.get("africa_flood_rows", "n/a")),
            "note": info.get("note", ""),
        })
    df_summary = pd.DataFrame(rows)

    out_csv = TABLES_DIR / "data_audit_africa.csv"
    out_json = RESULTS_DIR / "data_audit_africa.json"

    df_summary.to_csv(out_csv, index=False)
    with open(out_json, "w") as fh:
        json.dump(audit, fh, indent=2, default=str)

    print_rule()
    print(df_summary.to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_json}")

    # ---- Readiness summary --------------------------------------------
    print_rule()
    print("DATA READINESS FOR PADR-Net TRAINING")
    print_rule()
    for region_key, info in AFRICA_REGIONS.items():
        print(f"\n  {info['label']}")
        print(f"    ISO codes   : {sorted(info['iso'])}")
        print(f"    BBox        : lat [{info['bbox'][0]}, {info['bbox'][2]}], "
              f"lon [{info['bbox'][1]}, {info['bbox'][3]}]")

    emdat_ok = audit["layers"]["emdat"].get("exists", False)
    gfd_ok   = audit["layers"]["gfd"].get("meta_exists", False)
    era5_ok  = (audit["layers"]["era5"].get("nc_files", 0) > 0 or
                audit["layers"]["era5"].get("zip_files", 0) > 0)
    harm_ok  = audit["layers"]["harmonised_events"].get("exists", False)

    print(f"\n  EM-DAT disaster inventory : {'[OK] available' if emdat_ok else '[!!] MISSING'}")
    print(f"  Global Flood Database     : {'[OK] available' if gfd_ok else '[!!] MISSING'}")
    print(f"  ERA5 reanalysis           : {'[OK] available' if era5_ok else '[!!] MISSING - see download_flood_data.py'}")
    print(f"  Harmonised event table    : {'[OK] available' if harm_ok else '[!!] MISSING - run build_flood_event_table.py first'}")

    if not harm_ok:
        print(
            "\n  NOTE: Run the following to build the harmonised event table:\n"
            "     python scripts/build_flood_event_table.py\n"
            "  (located in the project root scripts/ directory)"
        )


if __name__ == "__main__":
    main()
