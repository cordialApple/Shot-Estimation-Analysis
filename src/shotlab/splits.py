import hashlib

import numpy as np
import pandas as pd

from shotlab.data import ARTIFACTS, config, load_shots

LABELED = ("S1", "S2")


def hash_uniform(shot_ids, tag):
    salt = config()["salt"]
    values = np.empty(len(shot_ids))
    for i, shot_id in enumerate(np.asarray(shot_ids)):
        digest = hashlib.sha256(f"{salt}|{tag}|{int(shot_id)}".encode()).digest()
        values[i] = int.from_bytes(digest[:8], "big") / 2**64
    return values


def smallest_within(values, groups, size):
    ranks = pd.Series(values).groupby(np.asarray(groups)).rank(method="first")
    return (ranks <= size).to_numpy()


def build_manifest(shots):
    cfg = config()
    manifest = pd.DataFrame({
        "shot_id": shots["shot_id"].to_numpy(),
        "season": shots["season"].to_numpy(),
        "month_block": shots["month"].map(cfg["month_blocks"]).to_numpy(),
    })
    manifest["labeled"] = manifest["season"].isin(LABELED)
    manifest["u_panel"] = hash_uniform(manifest["shot_id"], "panel")
    manifest["u_tune"] = hash_uniform(manifest["shot_id"], "tune")
    manifest["panel"] = manifest["labeled"] & smallest_within(
        manifest["u_panel"], manifest["season"], cfg["panel_size"])
    block_key = manifest["season"] + "_" + manifest["month_block"].astype(str)
    manifest["tuning_query"] = manifest["labeled"] & smallest_within(
        manifest["u_tune"], block_key, cfg["tuning_query_size"])
    return manifest


def manifest_digest(manifest):
    payload = pd.util.hash_pandas_object(manifest, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(refresh=False):
    cache = ARTIFACTS / "split_manifest.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    manifest = build_manifest(load_shots())
    ARTIFACTS.mkdir(exist_ok=True)
    manifest.to_parquet(cache, index=False)
    return manifest


def month_folds(month_block):
    month_block = np.asarray(month_block)
    return [(np.flatnonzero(month_block != b), np.flatnonzero(month_block == b))
            for b in np.unique(month_block)]


def nested_subset(shot_ids, size, replicate):
    order = np.argsort(hash_uniform(shot_ids, f"subset{replicate}"), kind="stable")
    mask = np.zeros(len(shot_ids), dtype=bool)
    mask[order[:size]] = True
    return mask
