"""common.py – Shared paths, Africa region definitions, and utility helpers.

All PADR-Net scripts import from here.

By default DATA_ROOT resolves to the ``data/`` folder inside this repository,
which is sufficient to run scripts 04–06 (training + figures) without any
external downloads.

To re-run scripts 01–03 (raw data ingestion) override DATA_ROOT:

    export DATA_ROOT=/path/to/your/FloodData        # Linux / macOS
    $env:DATA_ROOT = "D:\\FloodData"                 # Windows PowerShell
"""

from __future__ import annotations

import os
from pathlib import Path
import json
import datetime as dt

# ── Root paths ────────────────────────────────────────────────────────────────
# REPO_ROOT is the top-level padrnet/ folder.
REPO_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT: env-var override → else the repo's own data/ directory.
_env_root = os.environ.get("DATA_ROOT")
DATA_ROOT = Path(_env_root) if _env_root else REPO_ROOT / "data"

# Convenience handles
RESULTS_DIR   = REPO_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"
SCENARIOS_DIR = RESULTS_DIR / "scenarios"

# Create output dirs on import
for _d in (FIGURES_DIR, TABLES_DIR, RESULTS_DIR, SCENARIOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data locations (used by scripts 01–03 only) ───────────────────────────
RAW_DIR     = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"

EMDAT_FILE  = RAW_DIR / "disaster_inventory" / "public_emdat_2026-05-11.xlsx"
GFD_META    = RAW_DIR / "satellite_flood_extent" / "global_flood_database" / "gfd_event_metadata.json"
GFD_SUMMARY = RAW_DIR / "satellite_flood_extent" / "global_flood_database" / "gfd_region_summary.csv"
ERA5_DIR    = RAW_DIR / "reanalysis" / "era5"
TOPO_DIR    = RAW_DIR / "topography"
SOIL_DIR    = RAW_DIR / "soil_moisture"

HARMONISED_EVENTS  = INTERIM_DIR / "harmonized_flood_events.csv"
STUDY_REGIONS_JSON = DATA_ROOT / "metadata" / "study_regions.json"

# Pre-computed processed tables — scripts 04–06 read these directly
ERA5_COVARIATES_CSV = DATA_ROOT / "processed" / "era5_covariates.csv"
DATA_AUDIT_CSV      = DATA_ROOT / "processed" / "data_audit_africa.csv"
AFRICA_EVENTS_CSV   = DATA_ROOT / "africa_event_table" / "africa_flood_events.csv"

# ── Africa study regions ──────────────────────────────────────────────────────
AFRICA_REGIONS = {
    "west_africa_niger_benue": {
        "label": "West Africa: Niger / Benue basin",
        "bbox": [4.0, -12.0, 15.0, 15.0],   # [lat_min, lon_min, lat_max, lon_max]
        "iso": {"BEN", "BFA", "CMR", "MLI", "NER", "NGA", "TCD"},
        "mechanisms": ["Sahel rainfall", "urban flood exposure", "data-sparse"],
    },
    "east_africa_nile_headwaters": {
        "label": "East Africa: Nile headwaters / Sudan-Ethiopia corridor",
        "bbox": [-4.0, 28.0, 16.0, 40.0],
        "iso": {"BDI", "ETH", "KEN", "RWA", "SDN", "SSD", "TZA", "UGA"},
        "mechanisms": ["seasonal rainfall", "large-basin routing", "high exposure"],
    },
    "southern_africa_limpopo_zambezi": {
        "label": "Southern Africa: Limpopo / Zambezi",
        "bbox": [-27.0, 20.0, -8.0, 37.0],
        "iso": {"AGO", "BWA", "MWI", "MOZ", "NAM", "ZAF", "ZMB", "ZWE"},
        "mechanisms": ["tropical cyclones", "reservoir influence", "drought-flood"],
    },
}

ALL_AFRICA_ISO: set[str] = set().union(
    *(r["iso"] for r in AFRICA_REGIONS.values())
)

# ── Validation setup ──────────────────────────────────────────────────────────
RETURN_PERIOD_TRAIN_MAX = 50
RETURN_PERIOD_TEST_MIN  = 100

TRAIN_YEARS = list(range(2000, 2018))
VAL_YEARS   = [2018, 2019]
TEST_YEARS  = list(range(2020, 2025))

METRIC_NAMES = ["NSE", "CSI", "RMSE", "MAE", "delta_mass_pct"]


def load_study_regions() -> dict:
    with open(STUDY_REGIONS_JSON) as fh:
        return json.load(fh)


def africa_region_names() -> list[str]:
    return list(AFRICA_REGIONS.keys())


def timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def print_banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def print_rule(width: int = 60) -> None:
    print("-" * width)


if __name__ == "__main__":
    print_banner("common.py self-test")
    print(f"REPO_ROOT       : {REPO_ROOT}")
    print(f"DATA_ROOT       : {DATA_ROOT}  (exists={DATA_ROOT.exists()})")
    print(f"ERA5_COVARIATES : {ERA5_COVARIATES_CSV.name}  (exists={ERA5_COVARIATES_CSV.exists()})")
    print(f"AFRICA_EVENTS   : {AFRICA_EVENTS_CSV.name}  (exists={AFRICA_EVENTS_CSV.exists()})")
    print(f"\nAfrica regions  : {africa_region_names()}")
