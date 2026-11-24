import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time

import pandas as pd

from shotlab.data import ARTIFACTS, ROOT, config

PREDICTIONS = ARTIFACTS / "predictions"
COLUMNS = ["shot_id", "model", "arm", "config_hash", "direction", "role", "fold", "n_train", "replicate",
           "context_seed", "p_raw", "p_cal"]
PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "tabfm", "tabicl"]


def config_hash(run_config):
    canonical = json.dumps(run_config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def stem(run_config, role):
    return f"{run_config['arm'].replace(':', '-')}__{run_config['direction']}__{role}__{config_hash(run_config)}"


def path_for(run_config, role):
    return PREDICTIONS / run_config["model"] / f"{stem(run_config, role)}.parquet"


def exists(run_config, role):
    return path_for(run_config, role).exists()


def write(frame, run_config, role, extra=None):
    path = path_for(run_config, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["model"] = run_config["model"]
    out["arm"] = run_config["arm"]
    out["config_hash"] = config_hash(run_config)
    out["direction"] = run_config["direction"]
    out["role"] = role
    out["n_train"] = run_config.get("n_train") or 0
    out["replicate"] = run_config.get("replicate", 0)
    for column, default in [("fold", -1), ("context_seed", -1), ("p_cal", float("nan"))]:
        if column not in out:
            out[column] = default
    partial = path.with_suffix(".partial")
    out[COLUMNS].to_parquet(partial, index=False)
    partial.replace(path)
    sidecar = {
        "run_config": run_config, "role": role, "rows": len(out), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(), "data_sha256": config()["data_sha256"], "manifest_sha256": config()["manifest_sha256"],
        "environment": environment(), **(extra or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))
    return path


def catalog():
    rows = []
    for sidecar in sorted(PREDICTIONS.glob("*/*.json")):
        meta = json.loads(sidecar.read_text())
        run = meta["run_config"]
        rows.append({"path": sidecar.with_suffix(".parquet"), "model": run["model"], "arm": run["arm"],
                     "direction": run["direction"], "role": meta["role"], "n_train": run.get("n_train") or 0,
                     "replicate": run.get("replicate", 0), "config_hash": config_hash(run), "rows": meta["rows"]})
    return pd.DataFrame(rows)


def read(**filters):
    listing = catalog()
    for column, value in filters.items():
        listing = listing[listing[column] == value]
    return pd.concat([pd.read_parquet(path) for path in listing["path"]], ignore_index=True)
