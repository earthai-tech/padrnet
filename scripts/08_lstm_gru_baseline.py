"""08_lstm_gru_baseline.py
==========================
LSTM and GRU numerical baselines for the PADR-Net comparison (Table 4).

Architecture: random-weight recurrent cell + Ridge readout.

Following the reservoir-computing comparison convention (Lukoševičius &
Jaeger, 2009; Gallicchio & Micheli, 2017), the recurrent weights are
randomly initialised and kept fixed; only the readout (Ridge regression)
is trained.  This directly isolates the contribution of the ESN's
structural constraints (spectral radius, sparsity) versus the general
capacity of tanh-gated recurrences.

Networks
--------
  RAND-LSTM : random-weight LSTM cell, hidden_dim=200
  RAND-GRU  : random-weight GRU cell,  hidden_dim=200
  ESN       : PADR-Net M6 (reference, re-run with same seed)

All three share the same readout paradigm:
  features = [final_hidden | time-mean_hidden]   (2 * hidden_dim)
  depth_head    : Ridge(alpha=alpha_aug) -> log(max_depth)
  severity_head : Ridge(alpha=1e-3)     -> log1p(severity)

Outputs
-------
results/tables/lstm_gru_baseline.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TABLES_DIR, print_banner, print_rule, timestamp

import importlib.util, types

def _load_script(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here  = Path(__file__).resolve().parent
_train = _load_script("_train04", _here / "04_padrnet_training.py")

HP                    = _train.HP
FRICTION_CF           = _train.FRICTION_CF
P_SCALE               = _train.P_SCALE
DT                    = _train.DT
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
evaluate_model        = _train.evaluate_model

nse            = _train.nse
pr_auc         = _train.pr_auc
csi_tss        = _train.csi_tss
delta_mass_pct = _train.delta_mass_pct


# =============================================================================
# Random-weight LSTM cell
# =============================================================================

class RandomLSTM:
    """LSTM with fixed random weights; only readout is trained.

    Gates use the full input-and-hidden formulation::

        i = sigmoid(W_i x + U_i h + b_i)
        f = sigmoid(W_f x + U_f h + b_f)
        g = tanh(W_g x + U_g h + b_g)
        o = sigmoid(W_o x + U_o h + b_o)
        c_new = f * c + i * g
        h_new = o * tanh(c_new)
    """

    def __init__(self, n_inputs: int = 1, n_hid: int = 200, seed: int = 0):
        self.N = n_hid
        rng = np.random.default_rng(seed)
        sc  = 1.0 / np.sqrt(n_hid)

        def W(rows, cols):
            return rng.standard_normal((rows, cols)).astype(np.float32) * sc

        self.W_i = W(n_hid, n_inputs);   self.U_i = W(n_hid, n_hid)
        self.W_f = W(n_hid, n_inputs);   self.U_f = W(n_hid, n_hid)
        self.W_g = W(n_hid, n_inputs);   self.U_g = W(n_hid, n_hid)
        self.W_o = W(n_hid, n_inputs);   self.U_o = W(n_hid, n_hid)
        self.b_i = rng.standard_normal(n_hid).astype(np.float32) * 0.05
        self.b_f = rng.standard_normal(n_hid).astype(np.float32) * 0.05 + 1.0  # forget-gate bias
        self.b_g = rng.standard_normal(n_hid).astype(np.float32) * 0.05
        self.b_o = rng.standard_normal(n_hid).astype(np.float32) * 0.05

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def drive(self, P: np.ndarray) -> np.ndarray:
        T = len(P)
        S = np.zeros((T, self.N), dtype=np.float32)
        h = np.zeros(self.N, dtype=np.float32)
        c = np.zeros(self.N, dtype=np.float32)
        for t in range(T):
            x  = np.array([P[t]], dtype=np.float32)
            i_ = self._sigmoid(self.W_i @ x + self.U_i @ h + self.b_i)
            f_ = self._sigmoid(self.W_f @ x + self.U_f @ h + self.b_f)
            g_ = np.tanh     (self.W_g @ x + self.U_g @ h + self.b_g)
            o_ = self._sigmoid(self.W_o @ x + self.U_o @ h + self.b_o)
            c  = f_ * c + i_ * g_
            h  = o_ * np.tanh(c)
            S[t] = h
        return S

    def summary(self, S: np.ndarray) -> np.ndarray:
        return np.concatenate([S[-1], S.mean(axis=0)])


# =============================================================================
# Random-weight GRU cell
# =============================================================================

class RandomGRU:
    """GRU with fixed random weights; only readout is trained.

        z = sigmoid(W_z x + U_z h + b_z)
        r = sigmoid(W_r x + U_r h + b_r)
        n = tanh(W_n x + U_n (r * h) + b_n)
        h_new = (1 - z) * h + z * n
    """

    def __init__(self, n_inputs: int = 1, n_hid: int = 200, seed: int = 0):
        self.N = n_hid
        rng = np.random.default_rng(seed)
        sc  = 1.0 / np.sqrt(n_hid)

        def W(rows, cols):
            return rng.standard_normal((rows, cols)).astype(np.float32) * sc

        self.W_z = W(n_hid, n_inputs);   self.U_z = W(n_hid, n_hid)
        self.W_r = W(n_hid, n_inputs);   self.U_r = W(n_hid, n_hid)
        self.W_n = W(n_hid, n_inputs);   self.U_n = W(n_hid, n_hid)
        self.b_z = rng.standard_normal(n_hid).astype(np.float32) * 0.05
        self.b_r = rng.standard_normal(n_hid).astype(np.float32) * 0.05
        self.b_n = rng.standard_normal(n_hid).astype(np.float32) * 0.05

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def drive(self, P: np.ndarray) -> np.ndarray:
        T = len(P)
        S = np.zeros((T, self.N), dtype=np.float32)
        h = np.zeros(self.N, dtype=np.float32)
        for t in range(T):
            x  = np.array([P[t]], dtype=np.float32)
            z_ = self._sigmoid(self.W_z @ x + self.U_z @ h + self.b_z)
            r_ = self._sigmoid(self.W_r @ x + self.U_r @ h + self.b_r)
            n_ = np.tanh(self.W_n @ x + self.U_n @ (r_ * h) + self.b_n)
            h  = (1.0 - z_) * h + z_ * n_
            S[t] = h
        return S

    def summary(self, S: np.ndarray) -> np.ndarray:
        return np.concatenate([S[-1], S.mean(axis=0)])


# =============================================================================
# Generic Ridge readout trainer / evaluator
# =============================================================================

def build_rnn_model(
    df_tr: pd.DataFrame,
    rnn_cell,
    lambda_phys: float,
    feature_groups: list[str],
) -> dict:
    use_h  = "H" in feature_groups
    H_cols = FEATURE_GROUPS["H"] if use_h else []
    t_cols = get_tabular_cols(feature_groups)

    r_sum_list = []
    maxh_list  = []
    y_sev_list = []

    rng_e = np.random.default_rng(123)
    for _, ev in df_tr.iterrows():
        P = reconstruct_precip_ts(
            ev, n_hours=HP["ts_length"],
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        c_f, p_sc = h_swe_params(ev) if use_h else (FRICTION_CF, P_SCALE)
        h = swe_depth_ts(P, c_f=c_f, p_scale=p_sc)
        S = rnn_cell.drive(P)
        r_sum_list.append(rnn_cell.summary(S))
        maxh_list.append(float(np.max(h)))
        y_sev_list.append(float(np.log1p(max(ev.get("severity_score", 0.0) or 0.0, 0.0))))

    R_sum  = np.vstack(r_sum_list)
    maxh   = np.array(maxh_list)
    y_sev  = np.array(y_sev_list)
    log_maxh = np.log1p(maxh)

    alpha_0 = HP["ridge_alpha"]
    # ── Depth head ────────────────────────────────────────────────────────────
    if H_cols:
        H_tr          = extract_tab(df_tr, H_cols)
        depth_H_scaler = StandardScaler().fit(H_tr)
        X_depth       = np.hstack([R_sum, depth_H_scaler.transform(H_tr)])
    else:
        depth_H_scaler = None
        X_depth        = R_sum

    depth_ridge = Ridge(alpha=alpha_0, fit_intercept=True).fit(X_depth, log_maxh)

    # ── Severity head ─────────────────────────────────────────────────────────
    if t_cols:
        T_tr          = extract_tab(df_tr, t_cols)
        sev_tab_scaler = StandardScaler().fit(T_tr)
        X_sev         = np.hstack([R_sum, sev_tab_scaler.transform(T_tr)])
    else:
        sev_tab_scaler = None
        X_sev          = R_sum

    severity_ridge = Ridge(alpha=alpha_0, fit_intercept=True).fit(X_sev, y_sev)

    return {
        "rnn":              rnn_cell,
        "depth_ridge":      depth_ridge,
        "severity_ridge":   severity_ridge,
        "depth_H_scaler":   depth_H_scaler,
        "sev_tab_scaler":   sev_tab_scaler,
    }


def evaluate_rnn_model(
    model: dict,
    df_te: pd.DataFrame,
    feature_groups: list[str],
) -> dict:
    rnn            = model["rnn"]
    depth_ridge    = model["depth_ridge"]
    severity_ridge = model["severity_ridge"]
    depth_H_scaler = model["depth_H_scaler"]
    sev_tab_scaler = model["sev_tab_scaler"]
    use_h  = "H" in feature_groups
    H_cols = FEATURE_GROUPS["H"] if use_h else []
    t_cols = get_tabular_cols(feature_groups)

    R_sum_te = []
    maxh_ref = []
    y_sev_ref = []

    rng_e = np.random.default_rng(456)
    for _, ev in df_te.iterrows():
        P = reconstruct_precip_ts(
            ev, n_hours=HP["ts_length"],
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        c_f, p_sc = h_swe_params(ev) if use_h else (FRICTION_CF, P_SCALE)
        h = swe_depth_ts(P, c_f=c_f, p_scale=p_sc)
        S = rnn.drive(P)
        R_sum_te.append(rnn.summary(S))
        maxh_ref.append(float(np.max(h)))
        y_sev_ref.append(float(np.log1p(max(ev.get("severity_score", 0.0) or 0.0, 0.0))))

    R_sum_te  = np.vstack(R_sum_te)
    maxh_ref  = np.array(maxh_ref)
    y_sev_ref = np.array(y_sev_ref)

    if H_cols and depth_H_scaler is not None:
        H_te    = extract_tab(df_te, H_cols)
        X_depth = np.hstack([R_sum_te, depth_H_scaler.transform(H_te)])
    else:
        X_depth = R_sum_te

    log_maxh_hat = depth_ridge.predict(X_depth)
    nse_depth    = nse(np.log1p(maxh_ref), log_maxh_hat)

    if t_cols and sev_tab_scaler is not None:
        T_te  = extract_tab(df_te, t_cols)
        X_sev = np.hstack([R_sum_te, sev_tab_scaler.transform(T_te)])
    else:
        X_sev = R_sum_te

    y_sev_hat      = severity_ridge.predict(X_sev)
    sp_rho, _      = scipy_stats.spearmanr(y_sev_ref, y_sev_hat)
    pr_auc_val     = pr_auc(y_sev_ref, y_sev_hat)
    mae_val        = float(np.mean(np.abs(y_sev_ref - y_sev_hat)))
    dm_sev         = delta_mass_pct(y_sev_ref, y_sev_hat)

    return {
        "NSE_depth":      round(nse_depth,  4),
        "Spearman":       round(float(sp_rho), 4),
        "PR_AUC":         round(pr_auc_val, 4),
        "MAE":            round(mae_val,    4),
        "delta_mass_pct": round(dm_sev,     4),
    }


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print_banner("08 -- LSTM / GRU Numerical Baseline")
    print(f"Timestamp : {timestamp()}\n", flush=True)

    df = load_data()
    rng_feat = np.random.default_rng(42)
    eh_df    = generate_e_h_features(df, rng_feat)
    df       = pd.concat([df, eh_df], axis=1)
    train, val, test = split_data(df)

    feat_groups = ["R", "M", "E", "H"]

    rows = []

    # ── RAND-LSTM ───────────────────────────────────────────────────────────────
    print_rule()
    print("Training RAND-LSTM (hidden=200) ...", flush=True)
    lstm_cell = RandomLSTM(n_inputs=1, n_hid=HP["N_res"], seed=42)
    lstm_model = build_rnn_model(train, lstm_cell,
                                  lambda_phys=HP["lambda_opt"],
                                  feature_groups=feat_groups)
    lstm_m = evaluate_rnn_model(lstm_model, test, feat_groups)
    print(f"  NSE_depth={lstm_m['NSE_depth']:.3f}  rho={lstm_m['Spearman']:.3f}  "
          f"PR-AUC={lstm_m['PR_AUC']:.3f}  MAE={lstm_m['MAE']:.3f}", flush=True)
    rows.append({"model": "RAND-LSTM", "hidden_dim": HP["N_res"],
                 "readout": "Ridge", **lstm_m})

    # ── RAND-GRU ────────────────────────────────────────────────────────────────
    print_rule()
    print("Training RAND-GRU (hidden=200) ...", flush=True)
    gru_cell  = RandomGRU(n_inputs=1, n_hid=HP["N_res"], seed=42)
    gru_model = build_rnn_model(train, gru_cell,
                                 lambda_phys=HP["lambda_opt"],
                                 feature_groups=feat_groups)
    gru_m = evaluate_rnn_model(gru_model, test, feat_groups)
    print(f"  NSE_depth={gru_m['NSE_depth']:.3f}  rho={gru_m['Spearman']:.3f}  "
          f"PR-AUC={gru_m['PR_AUC']:.3f}  MAE={gru_m['MAE']:.3f}", flush=True)
    rows.append({"model": "RAND-GRU", "hidden_dim": HP["N_res"],
                 "readout": "Ridge", **gru_m})

    # ── PADR-Net (ESN M6) — reference ──────────────────────────────────────────
    print_rule()
    print("Re-evaluating PADR-Net ESN M6 (reference) ...", flush=True)
    esn_model = build_padrnet(train, lambda_phys=HP["lambda_opt"],
                               feature_groups=feat_groups, seed=42)
    esn_m     = evaluate_model(esn_model, test, feat_groups)
    print(f"  NSE_depth={esn_m['NSE_depth']:.3f}  rho={esn_m['Spearman']:.3f}  "
          f"PR-AUC={esn_m['PR_AUC']:.3f}  MAE={esn_m['MAE']:.3f}", flush=True)
    rows.append({
        "model": "PADR-Net",
        "hidden_dim": HP["N_res"],
        "readout": "Ridge",
        "NSE_depth":       round(esn_m["NSE_depth"], 4),
        "Spearman":        round(esn_m["Spearman"], 4),
        "PR_AUC":          round(esn_m["PR_AUC"], 4),
        "MAE":             round(esn_m["MAE"], 4),
        "delta_mass_pct":  round(esn_m["delta_mass_pct"], 4),
    })

    # ── Summary table ────────────────────────────────────────────────────────────
    print_rule()
    print(f"\n  {'Model':18s}  {'NSE_depth':>9}  {'Spearman':>8}  "
          f"{'PR-AUC':>7}  {'MAE':>6}")
    print(f"  {'-'*55}")
    for r in rows:
        print(f"  {r['model']:18s}  {r['NSE_depth']:9.3f}  {r['Spearman']:8.3f}  "
              f"{r['PR_AUC']:7.3f}  {r['MAE']:6.3f}")

    df_out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "lstm_gru_baseline.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}", flush=True)
    print("Done.\n", flush=True)


if __name__ == "__main__":
    main()
