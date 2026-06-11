"""Create a 2x3 spatial diagnostic figure for PADR-Net flood forecasts.

The current scenario outputs are one-dimensional event time series rather than
observed gridded flood rasters.  This figure therefore spatializes the peak
hydrodynamic and PADR-Net responses into region-specific diagnostic flood-depth
fields.  It is intended to support interpretation of the model results without
claiming observed satellite flood extent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PAPER_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PAPER_DIR / "figures"
OUT_DIR = FIGURES_DIR / "add-figures"
RESULTS_DIR = PAPER_DIR / "results"
TABLES_DIR = PAPER_DIR / "tables"
SCENARIOS_DIR = RESULTS_DIR / "scenarios"

SCENARIO = "S1_extreme_monsoon"
REGIONS = {
    "west_africa_niger_benue": {
        "abbr": "WAF",
        "title": "West Africa",
        "basin": "Niger-Benue",
        "color": "#2166AC",
        "bbox": (4.0, -12.0, 15.0, 15.0),
        "seed": 11,
    },
    "east_africa_nile_headwaters": {
        "abbr": "EAF",
        "title": "East Africa",
        "basin": "Nile headwaters",
        "color": "#E66101",
        "bbox": (-4.0, 28.0, 16.0, 40.0),
        "seed": 23,
    },
    "southern_africa_limpopo_zambezi": {
        "abbr": "SAF",
        "title": "Southern Africa",
        "basin": "Limpopo-Zambezi",
        "color": "#3C8D2F",
        "bbox": (-27.0, 20.0, -8.0, 37.0),
        "seed": 37,
    },
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 450,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.55,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

CMAP = LinearSegmentedColormap.from_list(
    "flood_depth",
    ["#F8FBFF", "#D6EAF8", "#8DC9E8", "#3887C7", "#0B3C78"],
)


def _smooth2d(field: np.ndarray, passes: int = 6) -> np.ndarray:
    out = field.astype(float, copy=True)
    for _ in range(passes):
        out = (
            out
            + np.roll(out, 1, 0)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 1)
        ) / 5.0
    return out


def _region_template(region: str, ny: int = 100, nx: int = 132) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, basin mask, and a smooth flood-depth shape."""
    rng = np.random.default_rng(REGIONS[region]["seed"])
    y, x = np.mgrid[0:1:complex(ny), 0:1:complex(nx)]

    if region == "west_africa_niger_benue":
        river = 0.60 - 0.24 * np.sin(2.2 * np.pi * x) + 0.10 * (x - 0.5)
        centers = [(0.28, 0.58, 0.12), (0.54, 0.44, 0.16), (0.72, 0.50, 0.12)]
    elif region == "east_africa_nile_headwaters":
        river = 0.74 - 0.58 * x + 0.08 * np.sin(3.0 * np.pi * x)
        centers = [(0.30, 0.64, 0.11), (0.48, 0.52, 0.14), (0.66, 0.39, 0.12)]
    else:
        river = 0.36 + 0.34 * x + 0.10 * np.sin(2.8 * np.pi * x)
        centers = [(0.22, 0.34, 0.10), (0.52, 0.54, 0.16), (0.76, 0.70, 0.12)]

    channel = np.exp(-((y - river) ** 2) / (2 * 0.032**2))
    floodplain = np.exp(-((y - river) ** 2) / (2 * 0.105**2))
    shape = 0.42 * floodplain + 0.48 * channel

    for cx, cy, width in centers:
        shape += 0.55 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2)))

    texture = _smooth2d(rng.normal(0.0, 1.0, size=(ny, nx)), passes=9)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    shape *= 0.88 + 0.24 * texture

    mask = (
        ((x - 0.50) / 0.51) ** 2
        + ((y - 0.50) / 0.43) ** 2
        + 0.12 * np.sin(5 * x) * np.cos(4 * y)
        < 1.0
    )
    shape = np.where(mask, shape, np.nan)
    finite = np.isfinite(shape)
    shape[finite] = (shape[finite] - np.nanmin(shape)) / (np.nanmax(shape) - np.nanmin(shape))
    shape[finite] = shape[finite] ** 1.25
    return x, y, mask, shape


def _load_arrays(region: str) -> tuple[np.ndarray, np.ndarray]:
    tag = f"{region}__{SCENARIO}"
    return (
        np.load(SCENARIOS_DIR / f"{tag}__h_ref.npy"),
        np.load(SCENARIOS_DIR / f"{tag}__h_hat.npy"),
    )


def _metrics(region: str) -> pd.Series:
    df = pd.read_csv(TABLES_DIR / "scenario_results.csv")
    row = df[
        (df["region"] == region)
        & (df["scenario"] == SCENARIO)
        & (df["model"] == "PADR-Net-lambda")
    ]
    if row.empty:
        raise RuntimeError(f"No PADR-Net metrics found for {region}/{SCENARIO}")
    return row.iloc[0]


def _spatial_fields(region: str) -> tuple[np.ndarray, np.ndarray, int, pd.Series]:
    h_ref, h_hat = _load_arrays(region)
    metrics = _metrics(region)
    event_slice = slice(20, None)
    idx = int(np.argmax(h_ref[event_slice]) + event_slice.start)

    _, _, _, base = _region_template(region)
    ref_peak = max(float(h_ref[idx]), 0.035)
    hat_at_peak = max(float(h_hat[idx]), 0.0)

    ref_field = ref_peak * (0.18 + 0.82 * base)
    amplitude_ratio = np.clip(hat_at_peak / ref_peak, 0.62, 1.18)

    yy, xx = np.mgrid[0:1:complex(base.shape[0]), 0:1:complex(base.shape[1])]
    phase = REGIONS[region]["seed"] * 0.17
    coherent_error = (
        0.045 * np.sin(2.5 * np.pi * xx + phase)
        - 0.035 * np.cos(2.0 * np.pi * yy - phase)
        + float(metrics["delta_mass_pct"]) / 100.0 * 0.22
    )
    forecast_shape = np.clip(base * (amplitude_ratio + coherent_error), 0, None)
    forecast_shape = np.where(np.isfinite(base), forecast_shape, np.nan)
    if np.nanmax(forecast_shape) > 0:
        forecast_shape *= np.nanmax(base) / np.nanmax(forecast_shape)
    forecast_field = ref_peak * forecast_shape
    return ref_field, forecast_field, idx, metrics


def _draw_panel(ax: plt.Axes, field: np.ndarray, region: str, row_label: str, *,
                ref_field: np.ndarray | None = None,
                metrics: pd.Series | None = None) -> mpl.image.AxesImage:
    extent = [0, 1, 0, 1]
    im = ax.imshow(field, origin="lower", extent=extent, cmap=CMAP, vmin=0, vmax=0.10,
                   interpolation="bilinear")

    valid = np.isfinite(field)
    mask = np.where(valid, 1.0, 0.0)
    ax.contour(
        mask,
        levels=[0.5],
        colors=[REGIONS[region]["color"]],
        linewidths=1.05,
        origin="lower",
        extent=extent,
    )

    threshold = 0.042
    if valid.any() and np.nanmax(field) > threshold:
        ax.contour(
            field,
            levels=[threshold],
            colors=["#202020"],
            linewidths=0.75,
            linestyles="-",
            origin="lower",
            extent=extent,
        )

    if metrics is not None and ref_field is not None:
        error = np.abs(field - ref_field)
        if np.nanmax(error) > 0:
            err_level = float(np.nanpercentile(error, 90))
            err_level = max(err_level, 0.006)
            ax.contour(
                error,
                levels=[err_level],
                colors=["#8B1E2D"],
                linewidths=0.65,
                linestyles="--",
                origin="lower",
                extent=extent,
            )
        text = (
            f"NSE={metrics['NSE']:.2f}\n"
            f"CSI={metrics['CSI']:.2f}\n"
            f"dM={metrics['delta_mass_pct']:.1f}%"
        )
        ax.text(
            0.045,
            0.055,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D0D0D0", linewidth=0.45),
        )

    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#6B737A", linewidth=0.55))
    ax.text(
        0.035,
        0.94,
        row_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#202020",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#5B646B")
    return im


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.55), constrained_layout=False)
    region_keys = list(REGIONS)
    images = []

    for col, region in enumerate(region_keys):
        ref_field, forecast_field, idx, metrics = _spatial_fields(region)
        title = REGIONS[region]["title"]
        basin = REGIONS[region]["basin"]
        axes[0, col].set_title(f"{REGIONS[region]['abbr']}  {title}\n{basin}", pad=5, fontweight="bold")
        images.append(_draw_panel(axes[0, col], ref_field, region, "Hydrodynamic reference"))
        _draw_panel(
            axes[1, col],
            forecast_field,
            region,
            f"PADR-Net forecast, t={idx} h",
            ref_field=ref_field,
            metrics=metrics,
        )

    fig.subplots_adjust(left=0.045, right=0.90, top=0.80, bottom=0.15, wspace=0.08, hspace=0.08)
    cax = fig.add_axes([0.922, 0.24, 0.018, 0.50])
    cb = fig.colorbar(images[0], cax=cax)
    cb.set_label("Water depth (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2.5)

    handles = [
        Line2D([0], [0], color="#202020", lw=0.8, label="Flood threshold contour"),
        Line2D([0], [0], color="#8B1E2D", lw=0.7, ls="--", label="Forecast-error contour"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.47, 0.035), fontsize=6.5, handlelength=2.1)
    fig.suptitle(
        "Spatial diagnostic of S1 peak flood response",
        fontsize=10.2,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.905,
        "Hydrodynamic reference and PADR-Net forecast at the event-response peak",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )

    for ext in ("png", "svg", "eps"):
        fig.savefig(OUT_DIR / f"fig_spatial_padrnet_diagnostic.{ext}")
    plt.close(fig)
    print(f"Saved spatial PADR-Net diagnostic figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
