"""Run the full pipeline end-to-end: sampling → ETL → analysis → model.

Usage:
    python run_all.py              # run all steps
    python run_all.py --steps 01 03   # run only steps 01 and 03

Each step script is idempotent and re-runs overwrite their outputs, so you
can re-run any subset after changing a parameter.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

STEPS = {
    "01": ("scripts/01_sample_users.py", "sample complete users from the raw dataset"),
    "02": ("scripts/02_build_features.py", "ETL: events -> sessions -> sequences -> features"),
    "03": ("scripts/03_analysis.py", "funnels, loss paths, feature comparison, sankey"),
    "04": ("scripts/04_churn_model.py", "churn prediction (group split)"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS),
                    help="steps to run (default: all)")
    args = ap.parse_args()

    print(f"root: {REPO_ROOT}")
    raw = REPO_ROOT / "data" / "raw" / "UserBehavior.csv"
    if not raw.exists():
        print(f"ERROR: raw data not found at {raw}. Place the dataset there first.")
        sys.exit(1)

    for step in args.steps:
        script = REPO_ROOT / STEPS[step][0]
        print(f"\n=== step {step}: {STEPS[step][1]} ===")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"step {step} FAILED (exit {proc.returncode})")
            sys.exit(proc.returncode)
        print(f"step {step} ok ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
