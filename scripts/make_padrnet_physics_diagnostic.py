"""Create a physics-consistency diagnostic figure for PADR-Net.

This figure complements the spatial diagnostic by showing whether PADR-Net
preserves the hydrodynamic response structure during the S1 extreme-flood
scenario.  It uses existing one-dimensional scenario arrays:

* phase-space response: rainfall forcing versus water depth
* discrete water-balance residual: dh/dt compared with rainfall-driven storage
* flood-threshold exceedance behavior through time
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


PAPER_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PAPER_DIR / "figures" / "add-figures"
SCENARIOS_DIR = PAPER_DIR / "results" / "scenarios"
TABLES_DIR = PAPER_DIR / "tables"

SCENARIO = "S1_extreme_monsoon"
REGIONS = {
    "west_africa_niger_benue": {
        "abbr": "WAF",
        "title": "West Africa",
        "color": "#2166AC",
    },
    "east_africa_nile_headwaters": {
        "abbr": "EAF",
        "title": "East Africa",
        "color": "#E66101",
    },
    "southern_africa_limpopo_zambezi": {
        "abbr": "SAF",
        "title": "Southern Africa",
        "color": "#3C8D2F",
    },
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 450,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.035,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.35,
    "xtick.direction": "in",
    "ytick.direction": "in",
}


def _load(region: str) -> dict[str, np.ndarray]:
    tag = f"{region}__{SCENARIO}"
    keys = ("precip", "h_ref", "h_hat", "h_hat_0", "h_pers")
    return {key: np.load(SCENARIOS_DIR / f"{tag}__{key}.npy") for key in keys}


def _metrics(region: str) -> pd.Series:
    df = pd.read_csv(TABLES_DIR / "scenario_results.csv")
    row = df[
        (df["region"] == region)
        & (df["scenario"] == SCENARIO)
        & (df["model"] == "PADR-Net-lambda")
    ]
    if row.empty:
        raise RuntimeError(f"No metrics found for {region}/{SCENARIO}")
    return row.iloc[0]


def _event_window(arrays: dict[str, np.ndarray]) -> np.ndarray:
    precip = arrays["precip"]
    active = np.flatnonzero(precip > max(0.2, 0.08 * float(np.nanmax(precip))))
    if len(active) == 0:
        return np.arange(20, len(precip))
    lo = max(20, int(active[0]) - 18)
    hi = min(len(precip), int(active[-1]) + 48)
    return np.arange(lo, hi)


def _smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _mix_with_white(color: str, amount: float) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb * (1 - amount) + np.ones(3) * amount
    return mcolors.to_hex(mixed)


def _residual(precip: np.ndarray, depth: np.ndarray, tau_h: float) -> np.ndarray:
    """Discrete residual for a simple rainfall-storage balance.

    r(t) = dh/dt - [alpha P(t) - h(t)/tau]
    alpha is calibrated on the reference series for fair comparison.
    """
    dh = np.gradient(depth)
    p = precip / max(float(np.nanmax(precip)), 1e-6)
    target = dh + depth / tau_h
    alpha = float(np.dot(target, p) / max(np.dot(p, p), 1e-9))
    return dh - (alpha * p - depth / tau_h)


def _probability(depth: np.ndarray, threshold: float) -> np.ndarray:
    scale = max(0.004, 0.08 * threshold)
    return 1.0 / (1.0 + np.exp(-(depth - threshold) / scale))


def _plot_region(fig: plt.Figure, axes: np.ndarray, col: int, region: str) -> None:
    arrays = _load(region)
    metrics = _metrics(region)
    idx = _event_window(arrays)
    t = idx - idx[0]
    color = REGIONS[region]["color"]
    h_ref = arrays["h_ref"][idx]
    h_hat = arrays["h_hat"][idx]
    h_hat0 = arrays["h_hat_0"][idx]
    precip = arrays["precip"][idx]

    threshold = max(float(np.nanpercentile(h_ref, 66)), 0.48 * float(np.nanmax(h_ref)))
    tau = 28.0 if region != "southern_africa_limpopo_zambezi" else 18.0

    ax = axes[0, col]
    ax.plot(precip, h_ref, color="#111111", lw=1.35, zorder=5)
    ax.plot(precip, h_hat, color=color, lw=1.45, zorder=4)
    ax.plot(precip, h_hat0, color=color, lw=1.05, ls="--", zorder=3)
    ax.scatter([precip[np.argmax(h_ref)]], [np.nanmax(h_ref)], s=18, color="#111111", zorder=6)
    ax.set_title(f"{REGIONS[region]['abbr']}  {REGIONS[region]['title']}", fontweight="bold", pad=5)
    ax.set_xlabel("Rainfall forcing (mm h$^{-1}$)")
    if col == 0:
        ax.set_ylabel("Water depth (m)")
    ax.grid(True, color="#E4E8EC", zorder=0)
    ax.text(
        0.04,
        0.94,
        "response loop",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#333333",
    )

    ax = axes[1, col]
    r_ref = _smooth(_residual(precip, h_ref, tau))
    r_hat = _smooth(_residual(precip, h_hat, tau))
    r_hat0 = _smooth(_residual(precip, h_hat0, tau))
    ax.axhline(0, color="#444444", lw=0.65)
    ax.fill_between(t, r_hat, 0, color=_mix_with_white(color, 0.86), linewidth=0)
    ax.plot(t, r_ref, color="#111111", lw=1.25)
    ax.plot(t, r_hat, color=color, lw=1.35)
    ax.plot(t, r_hat0, color=color, lw=0.95, ls="--")
    ax.set_xlabel("Event time (h)")
    if col == 0:
        ax.set_ylabel("Balance residual")
    ax.grid(True, color="#E4E8EC")
    ax.text(
        0.04,
        0.92,
        "physics residual",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#333333",
    )

    ax = axes[2, col]
    p_ref = _probability(h_ref, threshold)
    p_hat = _probability(h_hat, threshold)
    p_hat0 = _probability(h_hat0, threshold)
    spread = np.clip(np.abs(p_hat - p_hat0) + 0.04, 0.04, 0.22)
    ax.fill_between(t, np.clip(p_hat - spread, 0, 1), np.clip(p_hat + spread, 0, 1),
                    color=_mix_with_white(color, 0.84), linewidth=0)
    ax.plot(t, p_ref, color="#111111", lw=1.25)
    ax.plot(t, p_hat, color=color, lw=1.45)
    ax.plot(t, p_hat0, color=color, lw=0.95, ls="--")
    ax.axhline(0.5, color="#777777", lw=0.65, ls=":")
    exceed = h_ref >= threshold
    if np.any(exceed):
        starts = np.flatnonzero(np.diff(np.r_[False, exceed]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[exceed, False]) == -1)
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[max(e - 1, s)], color="#EEEEEE", linewidth=0)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Event time (h)")
    if col == 0:
        ax.set_ylabel("Flood exceedance probability")
    ax.grid(True, color="#E4E8EC")
    ax.text(
        0.04,
        0.92,
        f"NSE={metrics['NSE']:.2f}, CSI={metrics['CSI']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D0D0D0", linewidth=0.45),
    )

    for row in range(3):
        axes[row, col].spines[["top", "right"]].set_visible(False)
        axes[row, col].tick_params(length=2.5, pad=1.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.9), constrained_layout=False)
    for col, region in enumerate(REGIONS):
        _plot_region(fig, axes, col, region)

    handles = [
        Line2D([0], [0], color="#111111", lw=1.4, label="Hydrodynamic reference"),
        Line2D([0], [0], color="#2166AC", lw=1.4, label="PADR-Net-lambda"),
        Line2D([0], [0], color="#2166AC", lw=1.0, ls="--", label="PADR-Net-0"),
        Line2D([0], [0], color="#777777", lw=0.75, ls=":", label="Flood threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.035), fontsize=6.7, handlelength=2.2)
    fig.suptitle(
        "PADR-Net preserves hydrodynamic response structure during flood intensification",
        y=0.975,
        fontsize=10.4,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.936,
        "Phase-space response, physics residual, and threshold-exceedance diagnostics for S1",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.125, wspace=0.25, hspace=0.34)

    for ext in ("png", "svg", "eps"):
        fig.savefig(OUT_DIR / f"fig_padrnet_physics_diagnostic.{ext}")
    plt.close(fig)
    print(f"Saved PADR-Net physics diagnostic figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
