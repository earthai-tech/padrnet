"""padrnet_config.py
===================
Central hyperparameter configuration for PADR-Net.

All scripts import HP and related constants from here.
Modify this file to reproduce sensitivity experiments
without editing the training script.
"""

# =============================================================================
# Reservoir hyperparameters
# =============================================================================
HP = {
    "N_res":            200,      # reservoir size (neurons)
    "spectral_radius":   0.90,    # rho(W_res) — must be < 1 (echo-state property)
    "input_scaling":     0.60,    # W_in scaling
    "leaking_rate":      0.25,    # leaking-rate ESN (alpha in state update)
    "sparsity":          0.12,    # fraction of non-zero entries in W_res
    "ridge_alpha":       1e-3,    # base Ridge regularisation (alpha_0)
    "ts_length":         168,     # event time-series length (hours; 7 days)
    "lambda_opt":        0.10,    # optimal physics weight (lambda*)
    "lambda_grid": [0.0, 0.01, 0.05, 0.10, 0.50, 1.00, 5.00],  # sensitivity grid
}

# =============================================================================
# SWE physics parameters
# =============================================================================
FRICTION_CF = 0.05   # linearised SWE friction coefficient (s^{-1})
DT          = 1.0    # time step (hours)
P_SCALE     = 1e-3   # precipitation to dimensionless depth scale factor

# =============================================================================
# Ablation models M0 – M8
# =============================================================================
# Each entry: (feature_groups, lambda_phys)
ABLATION_MODELS = {
    "M0": (["R"],                0.10),   # rainfall-only baseline
    "M1": (["R", "M"],           0.10),   # + antecedent memory
    "M2": (["R", "E"],           0.10),   # + exposure (no memory)
    "M3": (["R", "H"],           0.10),   # + hydrodynamics (no memory)
    "M4": (["R", "M", "E"],      0.10),   # socio-hydrological baseline
    "M5": (["R", "M", "H"],      0.10),   # physical-hydrological model
    "M6": (["R", "M", "E", "H"], 0.10),   # full model (lambda = lambda_opt)
    "M7": (["R", "M", "E", "H"], 0.00),   # full features, no physics
    "M8": (["R", "M", "E", "H"], 1.00),   # full features, strong physics
}

# Table 2 nested models (incremental predictor comparison)
NESTED_MODELS = ["M0", "M1", "M4", "M6"]

# =============================================================================
# Random seed
# =============================================================================
RANDOM_SEED = 2024
