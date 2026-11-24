import hashlib
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(os.environ.get("SHOTLAB_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
ARTIFACTS = Path(os.environ.get("SHOTLAB_ARTIFACTS", ROOT / "artifacts"))
RESULTS = Path(os.environ.get("SHOTLAB_RESULTS", ROOT / "results"))
CONFIGS = ROOT / "configs"
TARGET = "outcome"
RAW_FILES = ("training.csv.gz", "testing.csv.gz")


@lru_cache
def config():
    return yaml.safe_load((CONFIGS / "study.yaml").read_text())


def load_grid(model):
    return yaml.safe_load((CONFIGS / f"{model}.yaml").read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums():
    expected = config()["data_sha256"]
    for name in RAW_FILES:
        found = sha256_file(DATA_DIR / name)
        if found != expected[name]:
            raise ValueError(f"checksum mismatch for {name}: {found}")


def season_labels():
    return {raw: label for label, raw in config()["seasons"].items()}


def load_shots(refresh=False):
    cache = ARTIFACTS / "shots.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    verify_checksums()
    frames = [pd.read_csv(DATA_DIR / name) for name in RAW_FILES]
    shots = pd.concat(frames, ignore_index=True)
    shots["season"] = shots["season_id"].map(season_labels())
    shots[TARGET] = shots[TARGET].astype("float64")
    shots = shots.sort_values("shot_id").reset_index(drop=True)
    ARTIFACTS.mkdir(exist_ok=True)
    shots.to_parquet(cache, index=False)
    return shots
