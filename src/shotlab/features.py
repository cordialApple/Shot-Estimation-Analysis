import numpy as np
import pandas as pd

from shotlab.data import ARTIFACTS, TARGET, config, load_shots

RIM_X = -41.75
SHOTTYPES = ["dunk", "floater", "heave", "jumper", "layup", "post", "tip"]
GAMESTATES = ["DIRECTSECONDCHANCE", "HALFCOURT", "TRANSITION"]
MONTHS = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR"]
FAMILY_OF = {"dunk": "rim", "layup": "rim", "tip": "rim", "floater": "short", "post": "short", "heave": "heave"}
FAMILIES = ["rim", "short", "mid", "three", "heave"]
MAX_CONTESTERS = 3

BLOCKS = {
    "location": ["locationx", "locationy", "distance", "shot_angle", "three"],
    "shot_type": ["shottype"],
    "creation": ["dribblesbefore", "catch_and_shoot", "shooterspeed", "shotclock", "sc_zero", "sc_reset", "gamestate"],
    "defense": ["closestdefdist", "num_contesters", "distcont2"],
    "closeout": ["closeout_1", "closeout_2", "closeout_3", "closeout_4"],
    "identity": ["shooter_edge", "defender_edge"],
    "team_month": ["off_team_edge", "def_team_edge", "month_idx"],
}
PRIMARY_BLOCKS = ["location", "shot_type", "creation", "defense", "closeout", "identity"]
LADDER = ["location", "shot_type", "creation", "defense", "closeout", "identity"]
KEYS = ["shooter_id", "closestdef_id", "off_team_id", "def_team_id", "shot_family", "month_block"]
META = ["shot_id", "season", TARGET]
CATEGORICAL = ["shottype", "gamestate"]
RAW_FEATURES = [c for block in BLOCKS.values() for c in block if not c.endswith("_edge")]


def columns_for(blocks):
    return [c for block in blocks for c in BLOCKS[block]]


def shot_family(shottype, three):
    family = shottype.map(FAMILY_OF)
    jumper = np.where(three, "three", "mid")
    return family.where(family.notna(), pd.Series(jumper, index=shottype.index))


def build_features(shots):
    approach = shots["closestdefapproach"].str.strip("{}").str.split(",", expand=True).astype(float)
    table = shots[META + ["shooter_id", "closestdef_id", "off_team_id", "def_team_id"]].copy()
    table["shot_family"] = pd.Categorical(shot_family(shots["shottype"], shots["three"]), FAMILIES)
    table["month_block"] = shots["month"].map(config()["month_blocks"]).astype("int8")
    for column in ["locationx", "locationy", "distance", "shooterspeed", "shotclock", "closestdefdist"]:
        table[column] = shots[column].astype("float64")
    table["shot_angle"] = np.degrees(np.arctan2(shots["locationy"].abs(), shots["locationx"] - RIM_X))
    table["three"] = shots["three"].astype("int8")
    table["shottype"] = pd.Categorical(shots["shottype"], SHOTTYPES)
    table["dribblesbefore"] = shots["dribblesbefore"].astype("float64")
    table["catch_and_shoot"] = (shots["dribblesbefore"] == 0).astype("int8")
    table["sc_zero"] = (shots["shotclock"] == 0).astype("int8")
    table["sc_reset"] = (shots["shotclock"] == 24).astype("int8")
    table["gamestate"] = pd.Categorical(shots["gamestate"], GAMESTATES)
    table["num_contesters"] = shots["num_contesters"].clip(upper=MAX_CONTESTERS).astype("float64")
    table["distcont2"] = shots["distcont2"].astype("float64")
    for step in range(4):
        table[f"closeout_{step + 1}"] = approach[step] - shots["closestdefdist"]
    table["month_idx"] = shots["month"].map({m: i for i, m in enumerate(MONTHS)}).astype("float64")
    return table


def load_features(refresh=False):
    cache = ARTIFACTS / "features.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    table = build_features(load_shots())
    table.to_parquet(cache, index=False)
    return table


def season_frame(table, season):
    return table[table["season"] == season].reset_index(drop=True)


def model_input(frame):
    return frame[KEYS + RAW_FEATURES]


def target(frame):
    return frame[TARGET].to_numpy().astype(int)
