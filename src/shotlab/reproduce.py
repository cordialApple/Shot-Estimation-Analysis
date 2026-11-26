import argparse
import subprocess
import sys

from shotlab.data import ROOT

NOTEBOOKS = {
    "models": ["00_data_audit", "01_logistic_regression", "02_gradient_boosting"],
    "comparison": ["05_model_comparison", "06_basketball_insights"],
}


def execute(name):
    path = ROOT / "notebooks" / f"{name}.ipynb"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Re-execute the local notebooks in place.")
    parser.add_argument("stage", nargs="?", choices=["models", "comparison", "all"], default="all")
    args = parser.parse_args()
    for stage in NOTEBOOKS if args.stage == "all" else [args.stage]:
        for name in NOTEBOOKS[stage]:
            print("executing", name, flush=True)
            execute(name)


if __name__ == "__main__":
    main()
