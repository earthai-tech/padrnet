"""02_build_africa_event_table.py
================================
Build an Africa-only flood event table for the PADR-Net paper by
subsetting the harmonised global event table (already includes EM-DAT
fields) and adding:
  - Estimated severity tier (low / moderate / extreme) as a proxy for
    return-period class
  - Train / val / test split label (used by 04_padrnet_training.py)

The harmonised table columns of interest:
  region, start_year, emdat_total_deaths, emdat_total_affected,
  era5_window_total_precip_mm, duration_days, event_inventory_source

Outputs
-------
tables/africa_flood_events.csv    - clean event table (one row per event)
tables/africa_region_summary.csv  - aggregate statistics per sub-region
results/africa_event_meta.json    - metadata + year coverage

Run
---
    python scripts/02_build_africa_event_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    HARMONISED_EVENTS,
    AFRICA_REGIONS,
    TABLES_DIR, RESULTS_DIR,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    print_banner, timestamp,
)


# =============================================================================
# helpers
# =============================================================================

def _assign_split(year) -> str:
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "unknown"
    if y in TRAIN_YEARS:
        return "train"
    if y in VAL_YEARS:
        return "val"
    if y in TEST_YEARS:
        return "test"
    return "out_of_range"


def _assign_severity(score: float, q50: float, q90: float) -> str:
    """Classify event severity using a composite impact score."""
    if score >= q90:
        return "extreme"
    if score >= q50:
        return "moderate"
    return "low"


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print_banner("02 -- Build Africa Event Table")
    print(f"Timestamp : {timestamp()}\n")

    # 1. Load harmonised event table ----------------------------------------
    print("Loading harmonised event table ...")
    if not HARMONISED_EVENTS.exists():
        raise FileNotFoundError(
            f"Harmonised event table not found at {HARMONISED_EVENTS}.\n"
            "Run:  python scripts/build_flood_event_table.py  (project root)"
        )
    df = pd.read_csv(HARMONISED_EVENTS)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  Harmonised table total rows : {len(df)}")
    print(f"  Columns available           : {len(df.columns)}")

    # 2. Filter to Africa regions -------------------------------------------
    africa = df[df["region"].isin(set(AFRICA_REGIONS.keys()))].copy()
    print(f"  Africa rows                 : {len(africa)}")
    print(f"  Africa region distribution  :")
    print(africa["region"].value_counts().to_string())

    # 3. Standardise key columns -------------------------------------------
    # year
    year_col = "start_year" if "start_year" in africa.columns else "year"
    africa["year"] = pd.to_numeric(africa[year_col], errors="coerce")

    # deaths and affected (from embedded EM-DAT fields)
    deaths_col = next(
        (c for c in africa.columns if "deaths" in c or "total_deaths" in c), None
    )
    aff_col = next(
        (c for c in africa.columns if "total_affected" in c or "affected" in c), None
    )
    precip_col = next(
        (c for c in africa.columns if "total_precip" in c or "precip_mm" in c), None
    )

    africa["deaths"]         = pd.to_numeric(africa[deaths_col], errors="coerce") if deaths_col else np.nan
    africa["total_affected"] = pd.to_numeric(africa[aff_col],    errors="coerce") if aff_col    else np.nan
    africa["precip_mm"]      = pd.to_numeric(africa[precip_col], errors="coerce") if precip_col else np.nan

    print(f"\n  Deaths column          : {deaths_col}")
    print(f"  Total affected column  : {aff_col}")
    print(f"  Precipitation column   : {precip_col}")

    # 4. Composite severity score  ------------------------------------------
    # Normalised: deaths + 0.001 * affected + 0.01 * precip (if available)
    combined_score = (
        africa["deaths"].fillna(0)
        + 0.001 * africa["total_affected"].fillna(0)
        + 0.01  * africa["precip_mm"].fillna(0)
    )
    q50 = float(combined_score.quantile(0.50))
    q90 = float(combined_score.quantile(0.90))
    africa["severity_score"] = combined_score
    africa["severity_tier"]  = combined_score.apply(
        lambda s: _assign_severity(s, q50, q90)
    )

    print(f"\n  Severity quantiles: q50={q50:.1f}, q90={q90:.1f}")
    print(f"  Severity distribution :")
    print(africa["severity_tier"].value_counts().to_string())

    # 5. Train / val / test split -------------------------------------------
    africa["split"] = africa["year"].apply(_assign_split)
    print(f"\n  Split distribution :")
    print(africa["split"].value_counts().to_string())

    # 6. Select clean output columns ----------------------------------------
    keep_cols = [
        "event_region_id", "region", "region_label", "year",
        "start_date", "end_date", "duration_days",
        "deaths", "total_affected", "precip_mm",
        "severity_score", "severity_tier", "split",
        "event_inventory_source",
    ]
    # add ERA5 summary columns if present
    for extra in ["era5_summary_status", "era5_window_mean_hourly_precip_mm",
                  "era5_window_max_t2m_c"]:
        if extra in africa.columns:
            keep_cols.append(extra)

    keep_cols = [c for c in keep_cols if c in africa.columns]
    africa_out = africa[keep_cols].copy()

    # 7. Per-region summary -------------------------------------------------
    region_summary = (
        africa.groupby("region")
        .agg(
            n_events=("region", "count"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            n_extreme=("severity_tier",   lambda x: (x == "extreme").sum()),
            n_moderate=("severity_tier",  lambda x: (x == "moderate").sum()),
            n_test=("split",              lambda x: (x == "test").sum()),
            total_deaths=("deaths",       "sum"),
            total_affected=("total_affected", "sum"),
            median_precip_mm=("precip_mm", "median"),
        )
        .reset_index()
    )
    region_summary["region_label"] = region_summary["region"].map(
        {k: v["label"] for k, v in AFRICA_REGIONS.items()}
    )

    # 8. Save outputs -------------------------------------------------------
    out_events  = TABLES_DIR / "africa_flood_events.csv"
    out_summary = TABLES_DIR / "africa_region_summary.csv"
    out_meta    = RESULTS_DIR / "africa_event_meta.json"

    africa_out.to_csv(out_events, index=False)
    region_summary.to_csv(out_summary, index=False)

    meta = {
        "timestamp": timestamp(),
        "source_harmonised": str(HARMONISED_EVENTS),
        "africa_regions": list(AFRICA_REGIONS.keys()),
        "total_africa_events": len(africa),
        "year_range": [int(africa["year"].min()), int(africa["year"].max())],
        "severity_quantiles": {"q50": q50, "q90": q90},
        "split_counts": africa["split"].value_counts().to_dict(),
        "severity_counts": africa["severity_tier"].value_counts().to_dict(),
        "note": (
            "Severity tier is a proxy for return period based on combined "
            "impact score (deaths + 0.001*affected + 0.01*precip_mm). "
            "extreme ~ >=100-yr class; moderate ~ 50-100 yr; low ~ <50 yr."
        ),
    }
    with open(out_meta, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print("REGION SUMMARY")
    print(f"{'='*60}")
    print(region_summary.to_string(index=False))

    print(f"\nSaved -> {out_events}")
    print(f"Saved -> {out_summary}")
    print(f"Saved -> {out_meta}")

    # Highlight test-split extreme events (paper evaluation targets)
    extreme_test = africa[
        (africa["severity_tier"] == "extreme") & (africa["split"] == "test")
    ]
    print(f"\nPAPER EVALUATION TARGETS")
    print(f"  Extreme events in TEST split : {len(extreme_test)}")
    if len(extreme_test) > 0:
        print(extreme_test[["region", "year", "deaths", "total_affected",
                             "severity_score"]].to_string(index=False))
    print(f"\nDone. {len(africa)} Africa flood events ready for PADR-Net training.\n")


if __name__ == "__main__":
    main()
