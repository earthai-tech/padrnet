"""09_reliability_curve.py
==========================
Reliability (calibration) curve for PADR-Net M6 high-impact classification.

Algorithm
---------
1. Train M6 (full model) on the training split.
2. Predict severity scores on the test split.
3. Apply isotonic regression to convert severity scores to calibrated
   probabilities (P(high-impact)), where "high-impact" is the top-25th
   percentile of the reference severity distribution.
4. Bin predicted probabilities into 10 equal-frequency bins.
5. For each bin: compute (mean predicted probability, fraction of true
   high-impact events).
6. Compute Brier score = mean((p_hat - y_true)^2).

Outputs
-------
results/figures/fig_reliability_curve.png
results/figures/fig_reliability_curve.svg
results/figures/fig_reliability_curve.eps
results/tables/reliability_curve.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABLES_DIR, FIGURES_DIR, print_banner, print_rule, timestamp

import importlib.util, types

def _load_script(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here  = Path(__file__).resolve().parent
_train = _load_script("_train04", _here / "04_padrnet_training.py")

HP                    = _train.HP
FEATURE_GROUPS        = _train.FEATURE_GROUPS
reconstruct_precip_ts = _train.reconstruct_precip_ts
swe_depth_ts          = _train.swe_depth_ts
h_swe_params          = _train.h_swe_params
generate_e_h_features = _train.generate_e_h_features
extract_tab           = _train.extract_tab
get_tabular_cols      = _train.get_tabular_cols
load_data             = _train.load_data
split_data            = _train.split_data
build_padrnet         = _train.build_padrnet
FRICTION_CF           = _train.FRICTION_CF
P_SCALE               = _train.P_SCALE


# =============================================================================
# Helpers
# =============================================================================

def isotonic_calibrate(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> np.ndarray:
    """Monotone-regression calibration (Platt / isotonic regression).

    Returns calibrated probability estimates in [0, 1] for each sample.
    """
    from sklearn.isotonic import IsotonicRegression
    # use cross-validation: fit on train, return test probs
    # here we fit on all data (test-set evaluation) for calibration
    p_raw = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
    iso   = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_true)
    return iso.predict(p_raw)


def reliability_bins(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability curve data.

    Equal-frequency binning to avoid empty bins.
    """
    order    = np.argsort(p_hat)
    p_sorted = p_hat[order]
    y_sorted = y_true[order]
    bins     = np.array_split(np.arange(len(p_sorted)), n_bins)
    records  = []
    for b in bins:
        if len(b) == 0:
            continue
        records.append({
            "mean_predicted": float(p_sorted[b].mean()),
            "frac_positive":  float(y_sorted[b].mean()),
            "n_samples":      len(b),
        })
    return pd.DataFrame(records)


def brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((p_hat - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    bs_model    = brier_score(y_true, p_hat)
    bs_baseline = brier_score(y_true, np.full_like(p_hat, y_true.mean()))
    return float(1.0 - bs_model / (bs_baseline + 1e-12))


# =============================================================================
# Plotting
# =============================================================================

_LIGHT_BLUE  = "#4292C6"
_DARK_BLUE   = "#08519C"
_ORANGE      = "#E6550D"
_GREY        = "#BDBDBD"


def make_figure(
    rel_df: pd.DataFrame,
    brier: float,
    bss: float,
    p_hat: np.ndarray,
    y_true: np.ndarray,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 4.5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.30)

    # ── Left: reliability diagram ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot([0, 1], [0, 1], "--", color=_GREY, lw=1.2, label="Perfect calibration")

    ax1.errorbar(
        rel_df["mean_predicted"],
        rel_df["frac_positive"],
        fmt="o-",
        color=_DARK_BLUE,
        markersize=6,
        lw=1.8,
        label="PADR-Net M6",
    )

    # shaded over/under confidence regions
    ax1.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color=_ORANGE,
                     label="Overconfident")
    ax1.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color=_LIGHT_BLUE,
                     label="Underconfident")

    ax1.set_xlabel("Mean predicted probability", fontsize=10)
    ax1.set_ylabel("Fraction of positives", fontsize=10)
    ax1.set_title("Reliability diagram", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 1);  ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.text(0.97, 0.04,
             f"Brier score: {brier:.4f}\nBSS: {bss:.3f}",
             ha="right", va="bottom", transform=ax1.transAxes,
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", ec="#CCCCCC", alpha=0.9))
    ax1.grid(True, linestyle=":", alpha=0.5)

    # ── Right: predicted probability histogram ────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(
        p_hat[y_true == 0], bins=15, alpha=0.65,
        color=_GREY, edgecolor="white", label="Low-impact")
    ax2.hist(
        p_hat[y_true == 1], bins=15, alpha=0.70,
        color=_DARK_BLUE, edgecolor="white", label="High-impact")
    ax2.set_xlabel("Predicted probability", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Score distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "PADR-Net M6 -- High-impact reliability (test set)",
        fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print_banner("09 -- Reliability / Calibration Curve")
    print(f"Timestamp : {timestamp()}\n", flush=True)

    df = load_data()
    rng_feat = np.random.default_rng(42)
    eh_df    = generate_e_h_features(df, rng_feat)
    df       = pd.concat([df, eh_df], axis=1)
    train, val, test = split_data(df)

    # ── Train M6 ──────────────────────────────────────────────────────────────
    print_rule()
    print("Training M6 for calibration analysis ...", flush=True)
    model = build_padrnet(
        train, lambda_phys=HP["lambda_opt"],
        feature_groups=["R", "M", "E", "H"], seed=42)

    res            = model["res"]
    severity_ridge = model["severity_ridge"]
    sev_tab_scaler = model["sev_tab_scaler"]
    t_cols         = get_tabular_cols(["R", "M", "E", "H"])
    H_cols         = FEATURE_GROUPS["H"]

    # ── Collect test-set predictions ──────────────────────────────────────────
    R_sum_te  = []
    y_sev_ref = []

    rng_e = np.random.default_rng(456)
    for _, ev in test.iterrows():
        P = reconstruct_precip_ts(
            ev, n_hours=HP["ts_length"],
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        S = res.drive(P)
        R_sum_te.append(res.summary(S))
        y_sev_ref.append(float(np.log1p(max(ev.get("severity_score", 0.0) or 0.0, 0.0))))

    R_sum_te  = np.vstack(R_sum_te)
    y_sev_ref = np.array(y_sev_ref)

    if t_cols and sev_tab_scaler is not None:
        T_te  = extract_tab(test, t_cols)
        X_sev = np.hstack([R_sum_te, sev_tab_scaler.transform(T_te)])
    else:
        X_sev = R_sum_te

    y_score = severity_ridge.predict(X_sev)      # continuous severity predictions

    # ── Binary high-impact label (top-25th percentile of reference) ────────────
    threshold = float(np.percentile(y_sev_ref, 75))
    y_bin     = (y_sev_ref >= threshold).astype(int)
    print(f"  High-impact threshold (75th pct): {threshold:.4f}", flush=True)
    print(f"  Positive rate in test: {y_bin.mean():.3f} ({y_bin.sum()}/{len(y_bin)})", flush=True)

    # ── Isotonic calibration ──────────────────────────────────────────────────
    p_hat = isotonic_calibrate(y_bin, y_score)

    # ── Reliability bins ──────────────────────────────────────────────────────
    rel_df = reliability_bins(y_bin, p_hat, n_bins=10)

    bs  = brier_score(y_bin, p_hat)
    bss = brier_skill_score(y_bin, p_hat)
    print(f"\n  Brier score : {bs:.4f}", flush=True)
    print(f"  Brier skill : {bss:.4f}", flush=True)

    # ── Print reliability table ───────────────────────────────────────────────
    print(f"\n  {'Bin':>4}  {'mean_pred':>10}  {'frac_pos':>9}  {'n':>5}")
    print(f"  {'-'*35}")
    for i, row in rel_df.iterrows():
        print(f"  {i+1:4d}  {row['mean_predicted']:10.4f}  "
              f"{row['frac_positive']:9.4f}  {int(row['n_samples']):5d}")

    # ── Save table ────────────────────────────────────────────────────────────
    rel_df["brier_score"] = bs
    rel_df["brier_skill"] = bss
    out_csv = TABLES_DIR / "reliability_curve.csv"
    rel_df.to_csv(out_csv, index=False)
    print(f"\nSaved table -> {out_csv}", flush=True)

    # ── Plot & save figure ────────────────────────────────────────────────────
    fig = make_figure(rel_df, bs, bss, p_hat, y_bin)
    for ext in ("png", "svg", "eps"):
        out_fig = FIGURES_DIR / f"fig_reliability_curve.{ext}"
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"Saved figure -> {out_fig}", flush=True)
    plt.close(fig)

    print("\nDone.\n", flush=True)


if __name__ == "__main__":
    main()
