from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict
from ..utils.logging import get_logger

logger = get_logger("features")

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df

def build_week1_features(
    matches: pd.DataFrame,
    events: pd.DataFrame,
    squads: pd.DataFrame,
    players: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """
    Minimal, leakage-safe features for Stage-1 baseline:
      - rolling player xG, shots, minutes over last N matches
      - venue, opponent_id, season, formation (categorical)
      - target label: selected_squad (in_xi or on_bench)
    """
    # Ensure datetime
    matches = ensure_datetime(matches, "match_date")

    # Label: player is in preliminary squad (XI or bench)
    label_df = squads.copy()
    label_df["selected_squad"] = ((label_df["in_xi"].astype(int) == 1) | (label_df["on_bench"].astype(int) == 1)).astype(int)
    label_df = label_df[["match_id", "team_id", "player_id", "selected_squad", "minutes", "position"]]

    # Basic per-player per-match aggregates from events
    ev = events.copy()
    if "xg" not in ev.columns:
        ev["xg"] = 0.0
    ev_counts = (
        ev.groupby(["match_id", "team_id", "player_id"])
          .agg(shots=("event_type", lambda s: (s == "shot").sum()),
               xg=("xg", "sum"),
               passes=("event_type", lambda s: (s == "pass").sum()),
               duels=("event_type", lambda s: (s == "duel").sum()))
          .reset_index()
    )

    # Merge with matches to get opponent/venue/date
    m = matches[["match_id", "team_id", "opponent_id", "venue", "season", "competition", "formation", "match_date"]]
    base = (label_df
            .merge(m, on=["match_id", "team_id"], how="left")
            .merge(ev_counts, on=["match_id", "team_id", "player_id"], how="left")
            .fillna({"shots": 0, "xg": 0.0, "passes": 0, "duels": 0}))

    # Sort for rolling windows
    base = base.sort_values(["player_id", "match_date"])

    # Rolling features by player (strictly past matches only, shift=1)
    def add_roll(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
        for col in cols:
            for w in windows:
                df[f"roll_{col}_{w}"] = (df.groupby("player_id")[col]
                                           .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean()))
        return df

    base = add_roll(base, ["xg", "shots", "passes", "duels", "minutes"], windows)

    # Categorical encodings (leave as strings; RF handles them via one-hot later)
    for cat in ["venue", "season", "competition", "formation", "position", "opponent_id"]:
        base[cat] = base[cat].astype("category")

    # Drop rows without label (shouldn't happen) or missing match_date
    base = base.dropna(subset=["selected_squad", "match_date"])

    # Minimal feature set for Week-1
    feature_cols = [c for c in base.columns if c.startswith("roll_")] + \
                   ["venue", "season", "competition", "formation", "position", "opponent_id"]

    keep = ["match_id", "team_id", "player_id", "match_date", "selected_squad"] + feature_cols
    return base[keep].reset_index(drop=True)
