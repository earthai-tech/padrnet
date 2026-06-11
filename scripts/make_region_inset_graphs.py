"""Create thin regional flood-signal inset graphs for Fig. 1.

The outputs are designed to be dropped onto the Africa region map as compact
visual summaries.  Each regional inset contains normalized annual signals for:
event frequency, event severity, and event duration, with validation/test years
shown as subtle background bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = PAPER_DIR / "tables"
OUT_DIR = PAPER_DIR / "figures" / "add-figures"

REGIONS = {
    "west_africa_niger_benue": {
        "abbr": "WAF",
        "label": "West Africa",
        "color": "#2166AC",
        "stem": "fig01_inset_waf_flood_signals",
    },
    "east_africa_nile_headwaters": {
        "abbr": "EAF",
        "label": "East Africa",
        "color": "#E66101",
        "stem": "fig01_inset_eaf_flood_signals",
    },
    "southern_africa_limpopo_zambezi": {
        "abbr": "SAF",
        "label": "Southern Africa",
        "color": "#3C8D2F",
        "stem": "fig01_inset_saf_flood_signals",
    },
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 7,
    "axes.linewidth": 0.55,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
}


def _smooth(values: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    """Gaussian smooth without external dependencies."""
    values = np.asarray(values, dtype=float)
    radius = max(2, int(np.ceil(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.zeros_like(values)
    values = np.nan_to_num(values, nan=0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _mix_with_white(color: str, amount: float) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb * (1 - amount) + np.ones(3) * amount
    return mcolors.to_hex(mixed)


def _annual_signals(df: pd.DataFrame, region: str) -> pd.DataFrame:
    years = np.arange(2001, 2025)
    sub = df[df["region"] == region].copy()
    grouped = sub.groupby("year")

    annual = pd.DataFrame(index=years)
    annual["events"] = grouped.size().reindex(years, fill_value=0).astype(float)
    annual["severity"] = (
        grouped["severity_score"].max().reindex(years, fill_value=0).astype(float)
    )
    annual["duration"] = (
        grouped["duration_days"].mean().reindex(years, fill_value=0).astype(float)
    )

    for col in ("events", "severity", "duration"):
        annual[f"{col}_n"] = _smooth(_normalize(annual[col].to_numpy()))

    annual["year"] = years
    return annual


def _plot_region(df: pd.DataFrame, region: str) -> None:
    meta = REGIONS[region]
    color = meta["color"]
    annual = _annual_signals(df, region)
    x = annual["year"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.65, 1.05))
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0.88))

    ax.axvspan(2018, 2019.99, color="#FCE4CA", lw=0)
    ax.axvspan(2020, 2024.7, color="#F8D8DD", lw=0)

    ax.fill_between(
        x,
        annual["events_n"].to_numpy(),
        0,
        color=_mix_with_white(color, 0.82),
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        annual["events_n"].to_numpy(),
        color=color,
        lw=1.65,
        solid_capstyle="round",
        zorder=4,
        label="events",
    )
    ax.plot(
        x,
        annual["severity_n"].to_numpy(),
        color="#B98300",
        lw=0.95,
        solid_capstyle="round",
        zorder=3,
        label="severity",
    )
    ax.plot(
        x,
        annual["duration_n"].to_numpy(),
        color="#4C566A",
        lw=0.85,
        solid_capstyle="round",
        zorder=2,
        label="duration",
    )

    peak_year = int(annual.loc[annual["events"].idxmax(), "year"])
    peak_value = float(annual.loc[peak_year, "events_n"])
    ax.scatter(
        [peak_year],
        [peak_value],
        s=13,
        facecolor="white",
        edgecolor=color,
        linewidth=0.8,
        zorder=5,
    )

    ax.text(
        0.015,
        0.86,
        meta["abbr"],
        transform=ax.transAxes,
        color=color,
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.015,
        0.65,
        meta["label"],
        transform=ax.transAxes,
        color="#222222",
        ha="left",
        va="center",
        fontsize=5.6,
    )
    ax.text(0.705, 0.88, "Val", transform=ax.transAxes, color="#B66F11", fontsize=4.8)
    ax.text(0.84, 0.88, "Test", transform=ax.transAxes, color="#9B1422", fontsize=4.8)

    legend_y = 0.12
    legend_items = [("floods", color), ("severity", "#B98300"), ("duration", "#4C566A")]
    for i, (label, line_color) in enumerate(legend_items):
        x0 = 0.56 + i * 0.135
        ax.plot([x0, x0 + 0.032], [legend_y, legend_y], transform=ax.transAxes,
                color=line_color, lw=0.9, solid_capstyle="round", clip_on=False)
        ax.text(x0 + 0.038, legend_y, label, transform=ax.transAxes,
                color=line_color, fontsize=4.45, va="center", ha="left")

    ax.set_xlim(2001, 2024.75)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([2001, 2007, 2013, 2019, 2024])
    ax.set_xticklabels(["01", "07", "13", "19", "24"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["", "", ""])
    ax.tick_params(length=2.1, pad=1.2, labelsize=4.9)
    ax.grid(axis="y", color="#E4E8EC", linewidth=0.28)
    ax.grid(axis="x", color="#EEF1F4", linewidth=0.24)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#606A73")
    ax.spines["bottom"].set_color("#606A73")

    for ext in ("png", "svg"):
        out = OUT_DIR / f"{meta['stem']}.{ext}"
        fig.savefig(out, transparent=True)

    out = OUT_DIR / f"{meta['stem']}.eps"
    fig.patch.set_alpha(1)
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out, transparent=False)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)
    df = pd.read_csv(TABLES_DIR / "africa_flood_events.csv")
    for region in REGIONS:
        _plot_region(df, region)
    print(f"Saved regional inset graphs to {OUT_DIR}")


if __name__ == "__main__":
    main()
