"""run_all.py
===========
Master runner: executes all PADR-Net pipeline scripts in order.

Usage
-----
    python scripts/run_all.py              # full pipeline
    python scripts/run_all.py --from 03   # resume from script 03
    python scripts/run_all.py --only 06   # run only script 06

Scripts
-------
  00 -- Generate synthetic event table (offline / reproducibility mode)
  01 -- Africa data audit
  02 -- Build Africa event table
  03 -- Build ERA5 covariates
  04 -- PADR-Net training + evaluation
  05 -- Flood scenario generation
  06 -- Publication figures
  07 -- Source-closure sensitivity analysis (Reviewer 3)
  08 -- LSTM / GRU random-weight baseline (Reviewer 3)
  09 -- Reliability / calibration curve (Reviewer 3)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE = [
    ("00", "00_generate_synthetic_events.py"),
    ("01", "01_africa_data_audit.py"),
    ("02", "02_build_africa_event_table.py"),
    ("03", "03_build_era5_covariates.py"),
    ("04", "04_padrnet_training.py"),
    ("05", "05_make_flood_scenarios.py"),
    ("06", "06_make_figures.py"),
    ("07", "07_source_closure_sensitivity.py"),
    ("08", "08_lstm_gru_baseline.py"),
    ("09", "09_reliability_curve.py"),
]


def run_script(script_name: str) -> bool:
    path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}")
    t0     = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(SCRIPTS_DIR.parent))
    elapsed = time.time() - t0
    status  = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}]  {script_name}  ({elapsed:.1f}s)")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PADR-Net pipeline")
    parser.add_argument("--from",  dest="from_step", default=None,
                        help="Start from script number (e.g. 03)")
    parser.add_argument("--only",  dest="only_step", default=None,
                        help="Run only this script number (e.g. 06)")
    args = parser.parse_args()

    to_run = PIPELINE
    if args.only_step:
        to_run = [(num, name) for num, name in PIPELINE if num == args.only_step]
    elif args.from_step:
        to_run = [(num, name) for num, name in PIPELINE if num >= args.from_step]

    if not to_run:
        print(f"No scripts matched -- check --from/--only arguments.")
        sys.exit(1)

    results = []
    t_total = time.time()
    for num, name in to_run:
        ok = run_script(name)
        results.append((name, ok))
        if not ok:
            print(f"\n[!] Script {name} failed. Stopping pipeline.\n")
            break

    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = "[OK]" if ok else "[FAILED]"
        print(f"  {status}  {name}")
    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    all_ok = all(ok for _, ok in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
