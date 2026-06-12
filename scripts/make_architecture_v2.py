"""make_architecture_v2.py
========================
Revised PADR-Net architecture figure for the Mathematical Geosciences
resubmission (round 2).

Corrections from the original submitted Fig. 3:
  * N_res = 200 (was 500)
  * W_in is FIXED random projection (was labelled "learned")
  * Lemma 1 = fading memory (was incorrectly "Lemma 2")
  * Theorem 1 = SWE strict hyperbolicity (physics residual block)
  * Theorem 2 = residual-distance bound d(q*,M)=O(lambda^{-1/2}) (objective)
  * Three observation operators: H_ext, H_depth, H_impact  [Prop. 2]
  * Two readout heads: depth + severity  [Prop. 1 head decoupling]
  * Observation-decomposed data loss, not quantile-pinball
  * Ridge regression readout only — no backprop through W_res or W_in
  * Source-term closure block C_psi in input column

Output
------
    results/figures/fig03_architecture_v2.{png,svg,eps}

Run
---
    python scripts/make_architecture_v2.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES_DIR, print_banner

# ── Journal style ─────────────────────────────────────────────────────────────
JOURNAL_STYLE = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":           8,
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
}

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT   = "#D6E4F7";  C_INPUT_B  = "#2166AC"   # blue  – inputs & closure
C_CLOSE   = "#BBDEFB";  C_CLOSE_B  = "#1565C0"   # deeper blue – source closure
C_RES     = "#FFF3CD";  C_RES_B    = "#9C7A00"   # gold  – reservoir
C_HEAD    = "#D5F0D5";  C_HEAD_B   = "#2E7D32"   # green – readout heads
C_SUBHEAD = "#C8E6C9";                            # lighter green – sub-boxes
C_OBS     = "#E0F4F4";  C_OBS_B    = "#00838F"   # teal  – observation operators
C_SUBOBS  = "#B2EBF2";                            # lighter teal – obs sub-boxes
C_EVAL    = "#F5F5F5";  C_EVAL_B   = "#555555"   # grey  – evaluation
C_PHYS    = "#FDECEA";  C_PHYS_B   = "#C62828"   # red   – SWE residual
C_OBJ     = "#F3EFE0";  C_OBJ_B    = "#5D4037"   # brown – objective
C_THM     = "#EDE7F6";  C_THM_B    = "#6A1B9A"   # purple – theorem / proposition
C_TRAIN   = "#ECEFF1";  C_TRAIN_B  = "#455A64"   # slate – training bar
C_TRANS   = "#E8F5E9";  C_TRANS_B  = "#2E7D32"   # green – transfer bar


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, face, edge, lw=1.0, pad=0.015, zorder=2):
    """Draw a rounded FancyBboxPatch in axes-fraction coordinates."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        linewidth=lw,
        facecolor=face,
        edgecolor=edge,
        transform=ax.transAxes,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(p)
    return p


def _txt(ax, x, y, s, fs=7.0, ha="center", va="center",
         bold=False, italic=False, color="k", zorder=5, **kw):
    """Place text in axes-fraction coordinates."""
    style  = "italic" if italic else "normal"
    weight = "bold"   if bold   else "normal"
    return ax.text(
        x, y, s,
        fontsize=fs, ha=ha, va=va,
        fontstyle=style, fontweight=weight,
        color=color, transform=ax.transAxes,
        zorder=zorder, **kw,
    )


def _arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.0,
           dashed=False, hw=0.006, hl=0.012, zorder=4):
    """Draw a filled-head arrow in axes-fraction coordinates."""
    ls = (0, (4, 3)) if dashed else "-"
    ax.annotate(
        "",
        xy=(x1, y1), xycoords="axes fraction",
        xytext=(x0, y0), textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle=f"-|>, head_width={hw:.4f}, head_length={hl:.4f}",
            color=color, lw=lw, linestyle=ls,
        ),
        zorder=zorder,
    )


# ── Layout constants ──────────────────────────────────────────────────────────

# Column x-positions (left edge) and widths
xA, wA = 0.01, 0.17    # ① Inputs & Closure
xB, wB = 0.22, 0.15    # ② Reservoir Core
xC, wC = 0.40, 0.16    # ③ Readout Heads
xD, wD = 0.60, 0.17    # ④ Obs. Operators
xE, wE = 0.81, 0.18    # ⑤ Evaluation

# y limits of each horizontal zone
Y_TOP    = 0.98   # absolute top (figure title)
Y_TBAR_T = 0.97   # training bar top
Y_TBAR_B = 0.90   # training bar bottom
Y_HDR    = 0.89   # column-header line
Y_MAIN_T = 0.87   # main blocks top
Y_MAIN_B = 0.33   # blocks A/D/E bottom
Y_BC_B   = 0.40   # blocks B/C bottom (raised – leaves room below)
Y_GAP    = 0.31   # gap between main and physics rows
Y_PHYS_T = 0.30   # physics/objective row top
Y_PHYS_B = 0.10   # physics/objective row bottom
Y_LEG_T  = 0.085  # legend top
Y_LEG_B  = 0.065  # legend bottom
Y_TRANS_T = 0.055 # transfer bar top
Y_TRANS_B = 0.01  # transfer bar bottom

H_MAIN  = Y_MAIN_T - Y_MAIN_B   # 0.54 – blocks A/D/E
H_BC    = Y_MAIN_T - Y_BC_B     # 0.47 – blocks B/C
H_PHYS  = Y_PHYS_T - Y_PHYS_B  # 0.20 – physics row


# ── Main drawing function ────────────────────────────────────────────────────

def make_architecture() -> plt.Figure:
    mpl.rcParams.update(JOURNAL_STYLE)

    fig_w = 174 / 25.4   # 6.85 in — full Springer two-column width
    fig_h = 150 / 25.4   # 5.91 in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── ① TRAINING REGIME BAR ────────────────────────────────────────────────
    _box(ax, 0.01, Y_TBAR_B, 0.98, Y_TBAR_T - Y_TBAR_B,
         C_TRAIN, C_TRAIN_B, lw=0.9, pad=0.005)
    _txt(ax, 0.06, (Y_TBAR_T + Y_TBAR_B) / 2,
         "Training regime", fs=7, bold=True, ha="left", color=C_TRAIN_B)
    _txt(ax, 0.52, (Y_TBAR_T + Y_TBAR_B) / 2,
         "Stage 1: Pretrain depth head  (λ = 0, supervised)   →   "
         "Stage 2: Joint optimisation  (λ = λ*,  ridge readout only,  "
         "no backprop through W_res or W_in)",
         fs=6.2, ha="center", color="#333333")

    # ── COLUMN HEADERS ────────────────────────────────────────────────────────
    hdrs = [
        (xA, wA, "(A) Inputs & Closure"),
        (xB, wB, "(B) Reservoir Core"),
        (xC, wC, "(C) Readout Heads"),
        (xD, wD, "(D) Obs. Operators"),
        (xE, wE, "(E) Evaluation"),
    ]
    for xh, wh, label in hdrs:
        _txt(ax, xh + wh / 2, Y_HDR, label, fs=6.5, bold=True,
             color="#333333", va="center")

    # ─────────────────────────────────────────────────────────────────────────
    # ① BLOCK A — Inputs & Source-term Closure
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xA, Y_MAIN_B, wA, H_MAIN, C_INPUT, C_INPUT_B, lw=1.2, pad=0.008)
    _txt(ax, xA + wA / 2, Y_MAIN_T - 0.022,
         "Input stream  x(t)", fs=7.5, bold=True, color=C_INPUT_B)

    input_lines = [
        "ERA5: P, T, u, v, PET",
        "antecedent ω_τ  (τ = 1, 7, 30 d)",
        "DEM · slope · TWI",
        "land cover · Manning n",
        "population · infrastructure",
        "exposure index  E_i",
    ]
    n_in = len(input_lines)
    dy_in = (H_MAIN - 0.20) / n_in
    for k, line in enumerate(input_lines):
        _txt(ax, xA + wA / 2,
             Y_MAIN_T - 0.055 - k * dy_in,
             line, fs=6.3, color="#1a1a1a")

    # Source-term closure sub-box (bottom part of A)
    yC_box = Y_MAIN_B + 0.01
    hC_box = 0.20
    _box(ax, xA + 0.01, yC_box, wA - 0.02, hC_box,
         C_CLOSE, C_CLOSE_B, lw=0.9, pad=0.006)
    _txt(ax, xA + wA / 2, yC_box + hC_box - 0.020,
         "Source closure  Cψ", fs=7, bold=True, color=C_CLOSE_B)
    for k, ln in enumerate([
        "soil + land  →  I, D",
        "terrain + routing  →  Q_lat",
        "land cover  →  Manning n",
        "bathymetry  ->  b,  h_0",
    ]):
        _txt(ax, xA + wA / 2, yC_box + hC_box - 0.055 - k * 0.035,
             ln, fs=6.2, color="#0D47A1")

    # ─────────────────────────────────────────────────────────────────────────
    # ② BLOCK B — Fixed Reservoir
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xB, Y_BC_B, wB, H_BC, C_RES, C_RES_B, lw=1.2, pad=0.008)
    _txt(ax, xB + wB / 2, Y_MAIN_T - 0.020,
         "Fixed reservoir", fs=7.5, bold=True, color=C_RES_B)
    _txt(ax, xB + wB / 2, Y_MAIN_T - 0.044,
         "W_res  (random, fixed)", fs=6.3, color="#4a3000")
    _txt(ax, xB + wB / 2, Y_MAIN_T - 0.063,
         "ρ(W_res) < 1", fs=6.5, color="#4a3000")
    _txt(ax, xB + wB / 2, Y_MAIN_T - 0.082,
         "N_res = 200", fs=7.5, bold=True, color=C_RES_B)

    # Reservoir neuron dots (3 rows × 4 cols)
    dot_y0  = Y_BC_B + 0.28
    dot_dy  = 0.048
    dot_x0  = xB + 0.025
    dot_dx  = (wB - 0.05) / 3.5
    for row in range(3):
        for col in range(4):
            ax.plot(dot_x0 + col * dot_dx, dot_y0 + row * dot_dy,
                    "o", ms=4, color=C_RES_B,
                    transform=ax.transAxes, zorder=4, clip_on=False)
        for col in range(3):
            ax.annotate(
                "",
                xy=(dot_x0 + (col + 1) * dot_dx, dot_y0 + row * dot_dy),
                xytext=(dot_x0 + col * dot_dx, dot_y0 + row * dot_dy),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=C_RES_B, lw=0.35),
                zorder=3,
            )
    # recurrent arrow above dots
    dot_top  = dot_y0 + 2 * dot_dy
    ax.annotate(
        "",
        xy=(dot_x0, dot_top + 0.04),
        xytext=(dot_x0 + 3 * dot_dx, dot_top + 0.04),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>, head_width=0.004, head_length=0.008",
            color=C_RES_B, lw=0.7,
            connectionstyle="arc3,rad=-0.35",
        ),
        zorder=4,
    )
    _txt(ax, xB + wB / 2, dot_top + 0.06, "W_res", fs=5.8, color=C_RES_B)

    _txt(ax, xB + wB / 2, Y_BC_B + 0.22,
         "temporal memory", fs=6.3, italic=True, color="#5D4037")
    _txt(ax, xB + wB / 2, Y_BC_B + 0.185,
         "h(t) = σ(W_in x(t) + W_res h(t−1))", fs=5.8, color="#333333")

    # Lemma 1 tag
    _box(ax, xB + 0.01, Y_BC_B + 0.01, wB - 0.02, 0.065,
         "#FFF8E1", C_RES_B, lw=0.8, pad=0.005)
    _txt(ax, xB + wB / 2, Y_BC_B + 0.052,
         "[Lemma 1]", fs=6.5, bold=True, color=C_RES_B)
    _txt(ax, xB + wB / 2, Y_BC_B + 0.030,
         "fading memory / echo-state", fs=5.8, italic=True, color=C_RES_B)

    # ─────────────────────────────────────────────────────────────────────────
    # ③ BLOCK C — Readout Heads
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xC, Y_BC_B, wC, H_BC, C_HEAD, C_HEAD_B, lw=1.2, pad=0.008)
    _txt(ax, xC + wC / 2, Y_MAIN_T - 0.020,
         "Readout heads", fs=7.5, bold=True, color=C_HEAD_B)

    # Depth head sub-box (upper half of C)
    yDH = Y_BC_B + H_BC / 2 + 0.015
    hDH = H_BC / 2 - 0.055
    _box(ax, xC + 0.01, yDH, wC - 0.02, hDH,
         C_SUBHEAD, C_HEAD_B, lw=0.8, pad=0.006)
    _txt(ax, xC + wC / 2, yDH + hDH - 0.020,
         "Depth head", fs=7, bold=True, color=C_HEAD_B)
    _txt(ax, xC + wC / 2, yDH + hDH - 0.045,
         "W_out,d", fs=6.5, color="#1B5E20")
    _txt(ax, xC + wC / 2, yDH + hDH - 0.065,
         "q̂ = (h, uh, vh)", fs=6.3, color="#1B5E20")
    _txt(ax, xC + wC / 2, yDH + hDH - 0.085,
         "ĥ = exp(ŷ_h) − 1 ≥ 0", fs=6.0, color="#1B5E20")

    # Severity head sub-box (lower half of C)
    ySH = Y_BC_B + 0.03
    hSH = H_BC / 2 - 0.055
    _box(ax, xC + 0.01, ySH, wC - 0.02, hSH,
         C_SUBHEAD, C_HEAD_B, lw=0.8, pad=0.006)
    _txt(ax, xC + wC / 2, ySH + hSH - 0.020,
         "Severity head", fs=7, bold=True, color=C_HEAD_B)
    _txt(ax, xC + wC / 2, ySH + hSH - 0.045,
         "β_s", fs=6.5, color="#1B5E20")
    _txt(ax, xC + wC / 2, ySH + hSH - 0.065,
         "y_A = beta_s^T  s_i", fs=6.3, color="#1B5E20")

    # Prop. 1 tag between the two sub-boxes
    yP1 = ySH + hSH + 0.002
    _txt(ax, xC + wC / 2, yP1 + 0.005,
         "[Prop. 1: head decoupling]", fs=5.8, bold=True, color=C_HEAD_B)

    # ─────────────────────────────────────────────────────────────────────────
    # ④ BLOCK D — Observation Operators
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xD, Y_MAIN_B, wD, H_MAIN, C_OBS, C_OBS_B, lw=1.2, pad=0.008)
    _txt(ax, xD + wD / 2, Y_MAIN_T - 0.020,
         "Observation operators", fs=7.5, bold=True, color=C_OBS_B)
    _txt(ax, xD + wD / 2, Y_MAIN_T - 0.042,
         "[Prop. 2: obs. decomp.]", fs=6.2, bold=True, color=C_OBS_B)

    obs_specs = [
        ("H_ext(q̂)",
         "satellite wet/dry extent", "→ ℓ_ext"),
        ("H_depth(q̂)",
         "benchmark / synthetic depth", "→ ℓ_depth"),
        ("H_impact(q̂, E)",
         "event-level severity", "→ ℓ_impact"),
    ]
    n_obs   = len(obs_specs)
    hObs    = (H_MAIN - 0.10) / n_obs - 0.012
    obs_y   = []   # store bottom-y of each obs sub-box
    for k, (op, desc, loss) in enumerate(obs_specs):
        yk = Y_MAIN_T - 0.075 - k * (hObs + 0.012) - hObs
        obs_y.append(yk)
        _box(ax, xD + 0.01, yk, wD - 0.02, hObs,
             C_SUBOBS, C_OBS_B, lw=0.7, pad=0.005)
        _txt(ax, xD + wD / 2, yk + hObs - 0.018,
             op, fs=6.8, bold=True, color=C_OBS_B)
        _txt(ax, xD + wD / 2, yk + hObs / 2,
             desc, fs=6.0, color="#004D55")
        _txt(ax, xD + wD / 2, yk + 0.012,
             loss, fs=6.3, bold=True, color=C_OBS_B)

    # ─────────────────────────────────────────────────────────────────────────
    # ⑤ BLOCK E — Evaluation Outputs
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xE, Y_MAIN_B, wE, H_MAIN, C_EVAL, C_EVAL_B, lw=1.2, pad=0.008)
    _txt(ax, xE + wE / 2, Y_MAIN_T - 0.020,
         "Evaluation", fs=7.5, bold=True, color=C_EVAL_B)

    eval_specs = [
        ("Extent",       "CSI · TSS",              C_OBS_B),
        ("Depth skill",  "NSE_depth · RMSE · MAE",  C_HEAD_B),
        ("Impact rank",  "ρ_s · PR-AUC · MAE", "#E65100"),
    ]
    n_eval  = len(eval_specs)
    hEval   = (H_MAIN - 0.07) / n_eval - 0.012
    eval_y  = []
    for k, (title, metrics, col) in enumerate(eval_specs):
        yk = Y_MAIN_T - 0.055 - k * (hEval + 0.012) - hEval
        eval_y.append(yk)
        _box(ax, xE + 0.01, yk, wE - 0.02, hEval,
             C_EVAL, col, lw=0.8, pad=0.005)
        _txt(ax, xE + wE / 2, yk + hEval - 0.022,
             title, fs=6.8, bold=True, color=col)
        _txt(ax, xE + wE / 2, yk + hEval / 2 - 0.005,
             metrics, fs=6.1, color="#333333")

    # ─────────────────────────────────────────────────────────────────────────
    # SWE RESIDUAL BLOCK (below Reservoir, B)
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, xB, Y_PHYS_B, wB, H_PHYS, C_PHYS, C_PHYS_B, lw=1.2, pad=0.008)
    _txt(ax, xB + wB / 2, Y_PHYS_T - 0.020,
         "SWE residual", fs=7.5, bold=True, color=C_PHYS_B)
    _txt(ax, xB + wB / 2, Y_PHYS_T - 0.044,
         "ℱ(q̂; P, I, D, Q_lat, n, b)", fs=6.0, color="#8B0000")
    _txt(ax, xB + wB / 2, Y_PHYS_T - 0.065,
         "ℓ_phys = ‖ℱ(q̂)‖²", fs=7, bold=True,
         color=C_PHYS_B)

    # Theorem 1 tag
    _box(ax, xB + 0.01, Y_PHYS_B + 0.01, wB - 0.02, 0.060,
         "#FFEBEE", C_PHYS_B, lw=0.8, pad=0.005)
    _txt(ax, xB + wB / 2, Y_PHYS_B + 0.052,
         "[Theorem 1]", fs=6.5, bold=True, color=C_PHYS_B)
    _txt(ax, xB + wB / 2, Y_PHYS_B + 0.030,
         "SWE strict hyperbolicity", fs=5.8, italic=True, color=C_PHYS_B)

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECTIVE BLOCK (below C + D)
    # ─────────────────────────────────────────────────────────────────────────
    xObj = xC
    wObj = xD + wD - xC     # spans C → right edge of D
    _box(ax, xObj, Y_PHYS_B, wObj, H_PHYS, C_OBJ, C_OBJ_B, lw=1.2, pad=0.008)
    _txt(ax, xObj + wObj / 2, Y_PHYS_T - 0.020,
         "Objective  ℓ", fs=7.5, bold=True, color=C_OBJ_B)
    _txt(ax, xObj + wObj / 2, Y_PHYS_T - 0.044,
         "L_data = w_ext L_ext  +  w_depth L_depth  +  w_impact L_impact",
         fs=6.3, color="#3E2723")
    _txt(ax, xObj + wObj / 2, Y_PHYS_T - 0.064,
         "L = L_data  +  lambda_phys L_phys  "
         "+  a_0 ||W_out||^2  +  a_s ||beta_s||^2",
         fs=6.8, bold=True, color=C_OBJ_B)
    _txt(ax, xObj + wObj / 2, Y_PHYS_T - 0.083,
         "Ridge regression readout only  —  no backprop through W_res or W_in",
         fs=5.9, italic=True, color="#555555")

    # Theorem 2 sub-box inside objective
    _box(ax, xObj + 0.01, Y_PHYS_B + 0.01, wObj - 0.02, 0.070,
         C_THM, C_THM_B, lw=0.9, pad=0.006)
    _txt(ax, xObj + wObj / 2, Y_PHYS_B + 0.062,
         "[Theorem 2]", fs=6.5, bold=True, color=C_THM_B)
    _txt(ax, xObj + wObj / 2, Y_PHYS_B + 0.044,
         "d( q_hat*, M ) = O( lambda_phys^(-1/2) )",
         fs=7.5, bold=True, color=C_THM_B)
    _txt(ax, xObj + wObj / 2, Y_PHYS_B + 0.025,
         "residual-distance bound", fs=6.0, italic=True, color=C_THM_B)

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSFER VALIDATION BAR
    # ─────────────────────────────────────────────────────────────────────────
    _box(ax, 0.01, Y_TRANS_B, 0.98, Y_TRANS_T - Y_TRANS_B,
         C_TRANS, C_TRANS_B, lw=0.8, pad=0.005)
    _txt(ax, 0.09, (Y_TRANS_T + Y_TRANS_B) / 2,
         "Transfer validation", fs=7, bold=True, ha="left", color=C_TRANS_B)
    _txt(ax, 0.54, (Y_TRANS_T + Y_TRANS_B) / 2,
         "Chronological holdout  ·  LORO: WAF · EAF · SAF  ·  "
         "LOYO: 2020–2024  ·  Impact quartile  ·  Label source",
         fs=6.3, ha="center", color="#1B5E20")

    # ─────────────────────────────────────────────────────────────────────────
    # LEGEND
    # ─────────────────────────────────────────────────────────────────────────
    leg_items = [
        (C_INPUT,  C_INPUT_B,  "Input / closure"),
        (C_RES,    C_RES_B,    "Reservoir (fixed)"),
        (C_HEAD,   C_HEAD_B,   "Readout (ridge)"),
        (C_OBS,    C_OBS_B,    "Obs. operator"),
        (C_PHYS,   C_PHYS_B,   "Physics residual"),
        (C_OBJ,    C_OBJ_B,    "Objective"),
        (C_THM,    C_THM_B,    "Theorem / Prop."),
    ]
    lh = Y_LEG_T - Y_LEG_B
    lw_sq = 0.012
    gap = (0.98 - 0.01 - len(leg_items) * (lw_sq + 0.060)) / (len(leg_items) - 1)
    lx0 = 0.01
    for k, (fc, ec, label) in enumerate(leg_items):
        lx = lx0 + k * (lw_sq + 0.060 + gap)
        _box(ax, lx, Y_LEG_B, lw_sq, lh, fc, ec, lw=0.7, pad=0.003)
        _txt(ax, lx + lw_sq + 0.005, Y_LEG_B + lh / 2,
             label, fs=5.8, ha="left", color="#333333")

    # ─────────────────────────────────────────────────────────────────────────
    # ARROWS
    # ─────────────────────────────────────────────────────────────────────────

    # ① A  →  ② B (W_in, fixed)
    yA_mid = (Y_BC_B + Y_MAIN_T) / 2
    _arrow(ax, xA + wA, yA_mid, xB, yA_mid, color=C_INPUT_B, lw=1.1)
    _txt(ax, (xA + wA + xB) / 2, yA_mid + 0.025,
         "W_in (fixed)", fs=5.8, color=C_INPUT_B)

    # ② B  →  ③ C (reservoir state h(t))
    yBC = (Y_BC_B + Y_MAIN_T) / 2
    _arrow(ax, xB + wB, yBC, xC, yBC, color=C_RES_B, lw=1.1)
    _txt(ax, (xB + wB + xC) / 2, yBC + 0.022,
         "h(t)", fs=6, color=C_RES_B)

    # Pre-compute obs sub-box mid-y values (from obs_y list)
    obs_mids  = [obs_y[k] + hObs / 2 for k in range(n_obs)]
    eval_mids = [eval_y[k] + hEval / 2 for k in range(n_eval)]

    # Depth head mid-y
    yDH_mid = yDH + hDH / 2
    ySH_mid = ySH + hSH / 2

    # ③ depth head  →  ④ H_ext  (q̂ into satellite extent)
    _arrow(ax, xC + wC, yDH_mid, xD, obs_mids[0],
           color=C_HEAD_B, lw=0.9)

    # ③ depth head  →  ④ H_depth  (q̂ into benchmark depth)
    _arrow(ax, xC + wC, yDH_mid, xD, obs_mids[1],
           color=C_HEAD_B, lw=0.9)

    # ③ severity head  →  ④ H_impact  (ŷ_A into impact operator)
    _arrow(ax, xC + wC, ySH_mid, xD, obs_mids[2],
           color=C_HEAD_B, lw=0.9)

    # ④ H_ext  →  ⑤ Extent
    _arrow(ax, xD + wD, obs_mids[0], xE, eval_mids[0],
           color=C_OBS_B, lw=0.9)

    # ④ H_depth  →  ⑤ Depth skill
    _arrow(ax, xD + wD, obs_mids[1], xE, eval_mids[1],
           color=C_OBS_B, lw=0.9)

    # ④ H_impact  →  ⑤ Impact rank
    _arrow(ax, xD + wD, obs_mids[2], xE, eval_mids[2],
           color=C_OBS_B, lw=0.9)

    # ① A (closure bottom)  →  SWE residual (dashed)
    yA_close_mid = Y_MAIN_B + 0.01 + 0.20 / 2
    _arrow(ax, xA + wA, yA_close_mid,
           xB, Y_PHYS_T - 0.002,
           color=C_CLOSE_B, lw=0.9, dashed=True)

    # ② B (reservoir) ↓  SWE residual (vertical dashed)
    _arrow(ax, xB + wB / 2, Y_BC_B, xB + wB / 2, Y_PHYS_T,
           color=C_RES_B, lw=1.0, dashed=True)

    # ③ C (readout) ↓  Objective (vertical dashed)
    _arrow(ax, xC + wC / 2, Y_BC_B, xC + wC / 2, Y_PHYS_T,
           color=C_HEAD_B, lw=1.0, dashed=True)

    # ④ D (obs) ↓  Objective (vertical dashed)
    _arrow(ax, xD + wD / 2, Y_MAIN_B, xD + wD / 2, Y_PHYS_T,
           color=C_OBS_B, lw=1.0, dashed=True)

    # SWE residual  →  Objective (horizontal, λ_phys label)
    _arrow(ax, xB + wB, Y_PHYS_B + H_PHYS / 2,
           xObj, Y_PHYS_B + H_PHYS / 2,
           color=C_PHYS_B, lw=1.1)
    _txt(ax, (xB + wB + xObj) / 2, Y_PHYS_B + H_PHYS / 2 + 0.020,
         "λ_phys ℓ_phys", fs=5.8, color=C_PHYS_B)

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE TITLE
    # ─────────────────────────────────────────────────────────────────────────
    _txt(ax, 0.5, 0.998,
         "PADR-Net  ·  Physics-Aware Deep Reservoir Network",
         fs=8.5, bold=True, ha="center", va="top", color="#1A237E")

    return fig


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg", "eps"):
        out = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"  Saved -> {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner("make_architecture_v2  –  revised PADR-Net architecture figure")
    fig = make_architecture()
    save_fig(fig, "fig03_architecture_v2")
    plt.close(fig)
    print("Done.")
