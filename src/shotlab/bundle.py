import json
import zipfile

from shotlab import features, splits, store
from shotlab.data import ARTIFACTS, ROOT, TARGET, sha256_file

BUNDLE = ARTIFACTS / "bundle"
ZIP_PATH = ARTIFACTS / "shotlab_bundle.zip"
CODE = ["pyproject.toml", "configs", "src/shotlab"]


def build_data():
    """Training files carry outcomes; query files carry none. The T file is the whole test season."""
    table = features.load_features()
    manifest = splits.load_manifest()
    table = table.merge(manifest[["shot_id", "panel", "tuning_query"]], on="shot_id", validate="one_to_one")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for season in splits.LABELED:
        rows = table[table["season"] == season].reset_index(drop=True)
        rows.drop(columns="panel").to_parquet(BUNDLE / f"train_{season}.parquet", index=False)
        rows[rows["panel"]].drop(columns=[TARGET, "panel", "tuning_query"]).to_parquet(BUNDLE / f"panel_{season}.parquet", index=False)
    unlabeled = table[table["season"] == "T"].reset_index(drop=True)
    unlabeled.drop(columns=[TARGET, "panel", "tuning_query"]).to_parquet(BUNDLE / "panel_T.parquet", index=False)
    info = {"git_commit": store.git_commit(),
            "sha256": {path.name: sha256_file(path) for path in sorted(BUNDLE.glob("*.parquet"))}}
    (BUNDLE / "bundle_info.json").write_text(json.dumps(info, indent=2))
    return info


def build_zip():
    info = build_data()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in CODE:
            path = ROOT / entry
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
        for file in sorted(BUNDLE.iterdir()):
            archive.write(file, f"artifacts/bundle/{file.name}")
    return ZIP_PATH, info


def verify(bundle=BUNDLE):
    info = json.loads((bundle / "bundle_info.json").read_text())
    for name, expected in info["sha256"].items():
        if sha256_file(bundle / name) != expected:
            raise ValueError(f"checksum mismatch for {name}")
    return info


if __name__ == "__main__":
    print(build_zip())
