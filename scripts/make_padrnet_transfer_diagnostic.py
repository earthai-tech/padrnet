"""Create a cross-region and temporal transfer diagnostic for PADR-Net.

The figure is designed to support the generalisation claim without duplicating
the existing LORO radar chart.  It combines:

* an Africa-style transfer schematic for leave-one-region-out evaluation,
* a held-out-region skill matrix,
* leave-one-year-out temporal robustness,
* regime robustness across the three synthetic flood mechanisms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


PAPER_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PAPER_DIR / "figures" / "add-figures"
TABLES_DIR = PAPER_DIR / "tables"

REGIONS = {
    "west_africa_niger_benue": {
        "abbr": "WAF",
        "label": "West Africa",
        "color": "#2166AC",
        "xy": (0.25, 0.63),
    },
    "east_africa_nile_headwaters": {
        "abbr": "EAF",
        "label": "East Africa",
        "color": "#E66101",
        "xy": (0.72, 0.64),
    },
    "southern_africa_limpopo_zambezi": {
        "abbr": "SAF",
        "label": "Southern Africa",
        "color": "#3C8D2F",
        "xy": (0.57, 0.25),
    },
}

REGION_ORDER = list(REGIONS)
REGIME_LABELS = {
    "S1_extreme_monsoon": "S1\nextreme\nmonsoon",
    "S2_cyclone_flash": "S2\nflash /\ncyclone",
    "S3_seasonal_sequence": "S3\nseasonal\nsequence",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "figure.dpi": 300,
    "savefig.dpi": 450,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.035,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

SKILL_CMAP = LinearSegmentedColormap.from_list(
    "transfer_skill",
    ["#FFF7EC", "#FDD49E", "#74ADD1", "#2166AC"],
)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer = pd.read_csv(TABLES_DIR / "transfer_results.csv")
    scenario = pd.read_csv(TABLES_DIR / "scenario_results.csv")
    return transfer, scenario


def _skill_score(row: pd.Series) -> float:
    rmse_skill = max(0.0, 1.0 - float(row["RMSE"]) / 5.0)
    mass = max(0.0, 1.0 - float(row["delta_mass_pct"]) / 100.0)
    return float(
        0.35 * row["NSE_depth"]
        + 0.22 * row["PR_AUC"]
        + 0.18 * row["Spearman"]
        + 0.15 * rmse_skill
        + 0.10 * mass
    )


def _lighten(color: str, amount: float = 0.78) -> str:
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(rgb * (1 - amount) + np.ones(3) * amount)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def _draw_transfer_schematic(ax: plt.Axes, loro: pd.DataFrame) -> None:
    ax.set_title("Leave-one-region-out transfer", fontweight="bold", pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    africa = np.array([
        [0.18, 0.82], [0.35, 0.88], [0.56, 0.82], [0.74, 0.70],
        [0.80, 0.52], [0.72, 0.34], [0.61, 0.16], [0.48, 0.08],
        [0.36, 0.20], [0.28, 0.38], [0.16, 0.55],
    ])
    ax.add_patch(patches.Polygon(africa, closed=True, facecolor="#F7F4E8",
                                 edgecolor="#C8BFA7", linewidth=0.8))

    scores = {str(row["held_out"]): _skill_score(row) for _, row in loro.iterrows()}
    for target in REGION_ORDER:
        tx, ty = REGIONS[target]["xy"]
        for source in REGION_ORDER:
            if source == target:
                continue
            sx, sy = REGIONS[source]["xy"]
            score = scores.get(target, 0.5)
            width = 0.55 + 2.3 * score
            rad = 0.13 if (source, target) != ("east_africa_nile_headwaters", "west_africa_niger_benue") else -0.13
            arrow = patches.FancyArrowPatch(
                (sx, sy),
                (tx, ty),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=width,
                color=_lighten(REGIONS[target]["color"], 0.18),
                shrinkA=13,
                shrinkB=13,
            )
            ax.add_patch(arrow)

    for region, meta in REGIONS.items():
        x, y = meta["xy"]
        ax.scatter([x], [y], s=310, color="white", edgecolor=meta["color"], linewidth=1.8, zorder=4)
        ax.scatter([x], [y], s=170, color=meta["color"], edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(x, y, meta["abbr"], ha="center", va="center", color="white",
                fontsize=7.1, fontweight="bold", zorder=6)
        ax.text(x, y - 0.075, meta["label"], ha="center", va="top",
                fontsize=5.8, color="#222222")

    ax.text(0.05, 0.05, "arrows: train on two regions -> held-out target",
            ha="left", va="bottom", fontsize=5.6, color="#444444")


def _draw_skill_matrix(ax: plt.Axes, loro: pd.DataFrame) -> None:
    metrics = [
        ("NSE_depth", "Depth\nNSE"),
        ("PR_AUC", "PR-\nAUC"),
        ("Spearman", "Rank\nrho"),
        ("RMSE_skill", "RMSE\nskill"),
        ("Mass_balance", "Mass\nbal."),
    ]
    matrix = []
    row_labels = []
    for region in REGION_ORDER:
        row = loro[loro["held_out"] == region].iloc[0].copy()
        row["RMSE_skill"] = max(0.0, 1.0 - float(row["RMSE"]) / 5.0)
        row["Mass_balance"] = max(0.0, 1.0 - float(row["delta_mass_pct"]) / 100.0)
        matrix.append([float(row[m[0]]) for m in metrics])
        row_labels.append(REGIONS[region]["abbr"])

    arr = np.array(matrix)
    im = ax.imshow(arr, cmap=SKILL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Held-out-region skill profile", fontweight="bold", pad=5)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if arr[i, j] > 0.72 else "#202020"
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color=color)
    ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _draw_temporal_transfer(ax: plt.Axes, loyo: pd.DataFrame) -> None:
    loyo = loyo.copy()
    loyo["held_out"] = loyo["held_out"].astype(int)
    loyo = loyo.sort_values("held_out")
    years = loyo["held_out"].to_numpy()
    depth = loyo["NSE_depth"].to_numpy(float)
    pr = loyo["PR_AUC"].to_numpy(float)
    rho = loyo["Spearman"].to_numpy(float)

    ax.set_title("Held-out-year robustness", fontweight="bold", pad=5)
    ax.plot(years, depth, color="#111111", lw=1.55, marker="o", ms=3.5, label="Depth NSE")
    ax.plot(years, pr, color="#2166AC", lw=1.25, marker="s", ms=3.2, label="PR-AUC")
    ax.plot(years, rho, color="#8A6F00", lw=1.15, marker="^", ms=3.2, label="Rank rho")
    for x, n in zip(years, loyo["n_test"].astype(int)):
        ax.text(x, 0.075, f"n={n}", ha="center", va="bottom", fontsize=5.2, color="#555555", rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(years)
    ax.set_ylabel("Skill")
    ax.set_xlabel("Held-out year")
    ax.grid(True, color="#E6EBEF", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=5.8, handlelength=1.8)


def _draw_regime_gain(ax: plt.Axes, scenario: pd.DataFrame) -> None:
    ax.set_title("Regime robustness against persistence", fontweight="bold", pad=5)
    scenarios = list(REGIME_LABELS)
    x = np.arange(len(scenarios))
    offsets = np.linspace(-0.18, 0.18, len(REGION_ORDER))

    for offset, region in zip(offsets, REGION_ORDER):
        sub = scenario[scenario["region"] == region]
        gains = []
        for scen in scenarios:
            lam = sub[(sub["scenario"] == scen) & (sub["model"] == "PADR-Net-lambda")].iloc[0]
            pers = sub[(sub["scenario"] == scen) & (sub["model"] == "Persistence")].iloc[0]
            gain = float(lam["CSI"] - pers["CSI"])
            gains.append(gain)
        color = REGIONS[region]["color"]
        ax.plot(x + offset, gains, color=color, lw=1.2, marker="o", ms=4.2,
                label=REGIONS[region]["abbr"])
        for xi, g in zip(x + offset, gains):
            ax.vlines(xi, 0, g, color=_lighten(color, 0.60), linewidth=1.8)

    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[s] for s in scenarios])
    ax.set_ylabel("CSI gain over persistence")
    ax.set_ylim(-0.08, 0.55)
    ax.grid(True, axis="y", color="#E6EBEF", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=5.8,
              handlelength=1.5, columnspacing=0.8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)
    transfer, scenario = _load_tables()
    loro = transfer[transfer["transfer_type"] == "LORO"].copy()
    loyo = transfer[transfer["transfer_type"] == "LOYO"].copy()

    fig = plt.figure(figsize=(7.3, 5.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.18], height_ratios=[1.05, 1.0])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    ax_year = fig.add_subplot(gs[1, 0])
    ax_regime = fig.add_subplot(gs[1, 1])

    _draw_transfer_schematic(ax_map, loro)
    im = _draw_skill_matrix(ax_matrix, loro)
    _draw_temporal_transfer(ax_year, loyo)
    _draw_regime_gain(ax_regime, scenario)

    for ax, label in zip([ax_map, ax_matrix, ax_year, ax_regime], ["a", "b", "c", "d"]):
        _panel_label(ax, f"({label})")

    cax = fig.add_axes([0.915, 0.575, 0.016, 0.245])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalised transfer skill", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.8, length=2.0)

    fig.suptitle(
        "PADR-Net transfers flood-response skill across regions, years, and regimes",
        y=0.975,
        fontsize=10.3,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Cross-region LORO, temporal LOYO, and scenario-regime diagnostics",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.06, right=0.89, top=0.88, bottom=0.08, wspace=0.28, hspace=0.37)

    for ext in ("png", "svg", "eps"):
        fig.savefig(OUT_DIR / f"fig_padrnet_transfer_diagnostic.{ext}")
    plt.close(fig)
    print(f"Saved PADR-Net transfer diagnostic figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
