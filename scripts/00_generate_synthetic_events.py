"""00_generate_synthetic_events.py
==================================
Generate a synthetic Africa flood event table that mirrors the schema
expected by scripts 04–09.

The generated data captures realistic statistical properties of ERA5-based
flood event catalogues (Sahel, East Africa, Southern Africa), derived from
published EM-DAT aggregate statistics (2000–2024).  All fields are
synthesised from a seeded RNG so results are fully reproducible.

Outputs
-------
results/tables/africa_flood_events.csv   (N ≈ 420 events, 2000–2024)
results/tables/africa_region_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    AFRICA_REGIONS,
    TABLES_DIR,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    print_banner, timestamp,
)

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Region-level climatology parameters
# ---------------------------------------------------------------------------

REGION_PARAMS = {
    "west_africa_niger_benue": {
        "n_events_per_year": (6, 3),     # (mean, std)  Poisson-ish
        "precip_total_mm": (280, 120),   # (mu, sigma) log-normal
        "duration_days": (4.5, 2.2),
        "deaths_mu_log": (3.2, 1.4),     # log-normal deaths
        "affected_mu_log": (9.5, 2.0),
        "soil_moist_base": 0.24,
        "runoff_scale": 0.55,
        "evap_scale": 3.2,
        "t2m_c_base": 30.0,
        "wind_speed_base": 3.5,
    },
    "east_africa_nile_headwaters": {
        "n_events_per_year": (7, 3),
        "precip_total_mm": (310, 140),
        "duration_days": (5.5, 2.8),
        "deaths_mu_log": (3.5, 1.5),
        "affected_mu_log": (10.0, 2.2),
        "soil_moist_base": 0.28,
        "runoff_scale": 0.65,
        "evap_scale": 3.8,
        "t2m_c_base": 26.0,
        "wind_speed_base": 3.0,
    },
    "southern_africa_limpopo_zambezi": {
        "n_events_per_year": (5, 2),
        "precip_total_mm": (260, 110),
        "duration_days": (4.0, 2.0),
        "deaths_mu_log": (2.9, 1.3),
        "affected_mu_log": (9.0, 1.9),
        "soil_moist_base": 0.20,
        "runoff_scale": 0.45,
        "evap_scale": 2.9,
        "t2m_c_base": 24.0,
        "wind_speed_base": 2.8,
    },
}


def _assign_split(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    return "out_of_range"


def generate_events() -> pd.DataFrame:
    years = list(range(2000, 2025))
    rows = []
    event_id = 0

    for region, rp in REGION_PARAMS.items():
        mu_n, std_n = rp["n_events_per_year"]
        for yr in years:
            # Number of events this year (Poisson-like)
            n_ev = max(1, int(RNG.normal(mu_n, std_n)))

            for _ in range(n_ev):
                event_id += 1

                # --- Precipitation -----------------------------------------------
                precip_total = float(np.clip(
                    RNG.lognormal(
                        np.log(rp["precip_total_mm"][0]),
                        rp["precip_total_mm"][1] / rp["precip_total_mm"][0]),
                    20.0, 2000.0))
                dur_days = float(np.clip(
                    RNG.normal(rp["duration_days"][0], rp["duration_days"][1]),
                    1.0, 14.0))
                dur_h = dur_days * 24.0
                precip_7d_max = float(np.clip(
                    precip_total * RNG.uniform(0.12, 0.30),
                    2.0, precip_total))
                precip_intensity = precip_total / max(dur_h, 1.0)
                precip_onset_h = float(np.clip(
                    RNG.normal(168 * 0.22, 168 * 0.10), 6.0, 140.0))
                precip_mean_hourly = precip_total / 168.0

                # --- ERA5 antecedent state ----------------------------------------
                sm_noise = RNG.normal(0, 0.04)
                soil_moist = float(np.clip(
                    rp["soil_moist_base"] + sm_noise, 0.05, 0.65))
                runoff = float(np.clip(
                    rp["runoff_scale"] * (precip_total / 300.0)
                    * RNG.lognormal(0, 0.35), 0.01, 5.0))
                evap = float(np.clip(
                    rp["evap_scale"] * RNG.lognormal(0, 0.20), 0.3, 15.0))
                t2m_c = float(np.clip(
                    rp["t2m_c_base"] + RNG.normal(0, 3.0), 10.0, 45.0))
                wind_speed = float(np.clip(
                    rp["wind_speed_base"] + RNG.normal(0, 0.8), 0.5, 12.0))

                # --- Impact (composite severity score) ---------------------------
                deaths_raw = float(np.clip(
                    RNG.lognormal(
                        rp["deaths_mu_log"][0], rp["deaths_mu_log"][1]),
                    0.0, 5000.0))
                affected_raw = float(np.clip(
                    RNG.lognormal(
                        rp["affected_mu_log"][0], rp["affected_mu_log"][1]),
                    10.0, 5_000_000.0))
                # Composite score: weighted sum of log-scaled impact indicators
                severity_score = float(
                    1.5 * np.log1p(deaths_raw)
                    + 1.0 * np.log1p(affected_raw / 1000.0)
                    + 0.5 * np.log1p(precip_total / 50.0))

                rows.append({
                    "event_id":                         event_id,
                    "region":                           region,
                    "year":                             yr,
                    "split":                            _assign_split(yr),
                    "duration_days":                    dur_days,
                    # ERA5 precipitation features
                    "era5_precip_7d_total":             precip_total,
                    "era5_precip_7d_max":               precip_7d_max,
                    "era5_precip_onset_hour":           precip_onset_h,
                    "era5_precip_intensity":            precip_intensity,
                    "era5_window_mean_hourly_precip_mm": precip_mean_hourly,
                    # ERA5 antecedent state
                    "era5_soil_moist_mean":             soil_moist,
                    "era5_runoff_total":                runoff,
                    "era5_evap_total":                  evap,
                    "era5_t2m_mean_c":                  t2m_c,
                    "era5_wind_speed_mean":              wind_speed,
                    # Impact
                    "emdat_total_deaths":               deaths_raw,
                    "emdat_total_affected":             affected_raw,
                    "severity_score":                   severity_score,
                })

    df = pd.DataFrame(rows)
    # Ensure severity_score is normalised to [0, 100] for interpretability
    s = df["severity_score"].values
    df["severity_score"] = 100.0 * (s - s.min()) / (s.max() - s.min() + 1e-10)
    return df


def main() -> None:
    print_banner("00 -- Generate Synthetic Event Table")
    print(f"Timestamp : {timestamp()}\n", flush=True)

    df = generate_events()

    # --- summary by split -------------------------------------------------------
    for sp, grp in df.groupby("split"):
        print(f"  {sp:12s}: {len(grp):4d} events  "
              f"sev_mean={grp['severity_score'].mean():.2f}  "
              f"sev_p90={grp['severity_score'].quantile(0.90):.2f}", flush=True)

    # --- save events table -------------------------------------------------------
    out_events = TABLES_DIR / "africa_flood_events.csv"
    df.to_csv(out_events, index=False)
    print(f"\nSaved {len(df)} events -> {out_events}", flush=True)

    # --- region summary ---------------------------------------------------------
    region_rows = []
    for region, grp in df.groupby("region"):
        label = AFRICA_REGIONS[region]["label"]
        region_rows.append({
            "region":        region,
            "label":         label,
            "n_events":      len(grp),
            "n_train":       (grp["split"] == "train").sum(),
            "n_val":         (grp["split"] == "val").sum(),
            "n_test":        (grp["split"] == "test").sum(),
            "sev_mean":      grp["severity_score"].mean(),
            "sev_p90":       grp["severity_score"].quantile(0.90),
            "precip_mean":   grp["era5_precip_7d_total"].mean(),
        })

    reg_df = pd.DataFrame(region_rows)
    out_reg = TABLES_DIR / "africa_region_summary.csv"
    reg_df.to_csv(out_reg, index=False)
    print(f"Saved region summary -> {out_reg}", flush=True)

    print("\nDone.\n", flush=True)


if __name__ == "__main__":
    main()
