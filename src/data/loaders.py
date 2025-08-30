from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import yaml
from ..utils.logging import get_logger

logger = get_logger("data.loaders")

@dataclass
class DataPaths:
    matches: Path
    events: Path
    squads: Path
    injuries: Path | None
    players: Path

def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config_data_paths(cfg_path: str | Path) -> DataPaths:
    cfg = read_yaml(cfg_path)
    p = cfg["paths"]
    return DataPaths(
        matches=Path(p["matches"]),
        events=Path(p["events"]),
        squads=Path(p["squads"]),
        injuries=Path(p.get("injuries")) if p.get("injuries") else None,
        players=Path(p["players"]),
    )

def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format for {path}")

def load_tables(paths: DataPaths) -> dict[str, pd.DataFrame]:
    logger.info("Loading data tables…")
    matches = _read_any(paths.matches)
    events = _read_any(paths.events)
    squads = _read_any(paths.squads)
    players = _read_any(paths.players)
    return {"matches": matches, "events": events, "squads": squads, "players": players}
