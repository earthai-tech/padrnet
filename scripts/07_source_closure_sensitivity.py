"""07_source_closure_sensitivity.py
====================================
Source-closure sensitivity analysis for PADR-Net.

Trains the full M6 model once (default SWE parameters), then evaluates
how metrics change when the SWE reference depth is recomputed under
perturbed parameter scenarios:

  S0  baseline          (FRICTION_CF=0.05, P_SCALE=1e-3, no lateral inflow)
  S1  low infiltration  (P_SCALE × 0.5)
  S2  high infiltration (P_SCALE × 1.5)
  S3  low friction      (FRICTION_CF × 0.8)
  S4  high friction     (FRICTION_CF × 1.2)
  S5  lateral inflow    (add Q_lat = 0.10 * mean(P) at every step)

Metrics reported: NSE_depth, CSI (top-25 pct), delta_mass_pct.

Outputs
-------
results/tables/source_closure_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABLES_DIR, print_banner, print_rule, timestamp

# --------------------------------------------------------------------------
# Re-use all helpers from 04 by import
# --------------------------------------------------------------------------

import importlib.util, types

def _load_script(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here  = Path(__file__).resolve().parent
_train = _load_script("_train04", _here / "04_padrnet_training.py")

# --------------------------------------------------------------------------
# Bring constants and helpers into local namespace
# --------------------------------------------------------------------------

HP           = _train.HP
FRICTION_CF  = _train.FRICTION_CF
P_SCALE      = _train.P_SCALE
DT           = _train.DT
FEATURE_GROUPS = _train.FEATURE_GROUPS
ABLATION_MODELS = _train.ABLATION_MODELS

ReservoirCore         = _train.ReservoirCore
reconstruct_precip_ts = _train.reconstruct_precip_ts
swe_depth_ts          = _train.swe_depth_ts
h_swe_params          = _train.h_swe_params
generate_e_h_features = _train.generate_e_h_features
extract_tab           = _train.extract_tab
get_tabular_cols      = _train.get_tabular_cols
build_padrnet         = _train.build_padrnet
load_data             = _train.load_data
split_data            = _train.split_data

nse            = _train.nse
csi_tss        = _train.csi_tss
delta_mass_pct = _train.delta_mass_pct


# --------------------------------------------------------------------------
# Perturbed SWE depth generator
# --------------------------------------------------------------------------

def swe_depth_ts_lateral(
    P: np.ndarray,
    h0: float = 0.05,
    c_f: float = FRICTION_CF,
    p_scale: float = P_SCALE,
    q_lat_frac: float = 0.10,     # fraction of mean P added as lateral inflow
) -> np.ndarray:
    """SWE with additive lateral inflow term Q_lat = q_lat_frac * mean(P)."""
    n     = len(P)
    h     = np.zeros(n)
    h[0]  = h0
    decay = np.exp(-c_f * DT)
    src   = (1.0 - decay) / (c_f * DT + 1e-10)
    q_lat = q_lat_frac * float(np.mean(P))
    for t_i in range(1, n):
        h[t_i] = decay * h[t_i - 1] + (p_scale * P[t_i] + q_lat) * DT * src
    return np.maximum(h, 0.0)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

SCENARIOS = {
    "S0_baseline":   {"c_f_mult": 1.0, "ps_mult": 1.0, "lateral": False},
    "S1_low_infilt": {"c_f_mult": 1.0, "ps_mult": 0.5, "lateral": False},
    "S2_high_infilt":{"c_f_mult": 1.0, "ps_mult": 1.5, "lateral": False},
    "S3_low_fric":   {"c_f_mult": 0.8, "ps_mult": 1.0, "lateral": False},
    "S4_high_fric":  {"c_f_mult": 1.2, "ps_mult": 1.0, "lateral": False},
    "S5_lateral":    {"c_f_mult": 1.0, "ps_mult": 1.0, "lateral": True},
}

SCENARIO_LABELS = {
    "S0_baseline":    r"Baseline ($c_f$=0.05, $\alpha_p$=1.0)",
    "S1_low_infilt":  r"Low infiltration ($\alpha_p \times 0.5$)",
    "S2_high_infilt": r"High infiltration ($\alpha_p \times 1.5$)",
    "S3_low_fric":    r"Low friction ($c_f \times 0.8$)",
    "S4_high_fric":   r"High friction ($c_f \times 1.2$)",
    "S5_lateral":     r"Lateral inflow ($Q_\mathrm{lat}=0.10\,\bar{P}$)",
}


def get_reference_maxh(
    df: pd.DataFrame,
    c_f_mult: float,
    ps_mult: float,
    lateral: bool,
    seed_start: int = 456,
) -> np.ndarray:
    """Compute perturbed max-depth reference for each event in df."""
    rng_e    = np.random.default_rng(seed_start)
    maxh_ref = []
    for _, ev in df.iterrows():
        P = reconstruct_precip_ts(
            ev, n_hours=HP["ts_length"],
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        c_f_base, p_sc_base = h_swe_params(ev)
        c_f = c_f_base * c_f_mult
        p_sc = p_sc_base * ps_mult
        if lateral:
            h = swe_depth_ts_lateral(P, c_f=c_f, p_scale=p_sc)
        else:
            h = swe_depth_ts(P, c_f=c_f, p_scale=p_sc)
        maxh_ref.append(float(np.max(h)))
    return np.array(maxh_ref)


def evaluate_on_scenario(
    model: dict,
    df_te: pd.DataFrame,
    c_f_mult: float,
    ps_mult: float,
    lateral: bool,
) -> dict:
    res         = model["res"]
    depth_ridge = model["depth_ridge"]
    depth_H_scaler = model["depth_H_scaler"]
    feature_groups = ["R", "M", "E", "H"]
    H_cols = FEATURE_GROUPS["H"]

    # reservoir summaries (same inputs regardless of perturbation)
    R_sum_te = []
    rng_e = np.random.default_rng(456)
    for _, ev in df_te.iterrows():
        P = reconstruct_precip_ts(
            ev, n_hours=HP["ts_length"],
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        S = res.drive(P)
        R_sum_te.append(res.summary(S))
    R_sum_te = np.vstack(R_sum_te)

    if depth_H_scaler is not None:
        H_te    = extract_tab(df_te, H_cols)
        H_te_sc = depth_H_scaler.transform(H_te)
        X_depth = np.hstack([R_sum_te, H_te_sc])
    else:
        X_depth = R_sum_te

    log_maxh_hat = depth_ridge.predict(X_depth)

    # perturbed reference
    maxh_ref = get_reference_maxh(df_te, c_f_mult, ps_mult, lateral)
    log_maxh_ref = np.log1p(maxh_ref)

    nse_d = nse(log_maxh_ref, log_maxh_hat)
    csi_d, _ = csi_tss(log_maxh_ref, log_maxh_hat, pct=75.0)
    dm_d  = delta_mass_pct(log_maxh_ref, log_maxh_hat)

    return {
        "NSE_depth":       round(nse_d, 4),
        "CSI":             round(csi_d, 4),
        "delta_mass_pct":  round(dm_d,  4),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    print_banner("07 -- Source-closure Sensitivity Analysis")
    print(f"Timestamp : {timestamp()}\n", flush=True)

    # --- load data --------------------------------------------------------------
    df = load_data()
    rng_feat = np.random.default_rng(42)
    eh_df = generate_e_h_features(df, rng_feat)
    df = pd.concat([df, eh_df], axis=1)
    train, val, test = split_data(df)

    # --- train full model once (M6) --------------------------------------------
    print_rule()
    print("Training M6 (full model) on default SWE parameters ...", flush=True)
    model = build_padrnet(
        train, lambda_phys=HP["lambda_opt"],
        feature_groups=["R", "M", "E", "H"], seed=42)
    print(f"  actual spectral radius: {model['spectral_radius']:.4f}", flush=True)
    print(f"  augmented ridge alpha:  {model['alpha_aug']:.6f}", flush=True)

    # --- evaluate all scenarios ------------------------------------------------
    print_rule()
    print("Evaluating sensitivity scenarios ...\n", flush=True)
    rows = []
    for sc_id, sc_params in SCENARIOS.items():
        m = evaluate_on_scenario(
            model, test,
            c_f_mult=sc_params["c_f_mult"],
            ps_mult=sc_params["ps_mult"],
            lateral=sc_params["lateral"],
        )
        label = SCENARIO_LABELS[sc_id]
        row = {"scenario_id": sc_id, "label": label, **m}
        rows.append(row)
        print(f"  {sc_id:20s}  NSE_depth={m['NSE_depth']:6.3f}  "
              f"CSI={m['CSI']:5.3f}  dMass={m['delta_mass_pct']:6.2f}%", flush=True)

    # --- save -------------------------------------------------------------------
    df_out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "source_closure_sensitivity.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}", flush=True)

    # --- delta table (relative to S0 baseline) ----------------------------------
    baseline = rows[0]
    print("\nDelta from baseline:")
    for r in rows[1:]:
        d_nse = r["NSE_depth"] - baseline["NSE_depth"]
        d_csi = r["CSI"]       - baseline["CSI"]
        d_dm  = r["delta_mass_pct"] - baseline["delta_mass_pct"]
        print(f"  {r['scenario_id']:20s}  dNSE={d_nse:+.3f}  "
              f"dCSI={d_csi:+.3f}  dMass%={d_dm:+.2f}", flush=True)

    print("\nDone.\n", flush=True)


if __name__ == "__main__":
    main()
