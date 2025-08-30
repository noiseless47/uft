#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport
import yaml

from src.data.loaders import load_config_data_paths, load_tables
from src.utils.logging import get_logger
from src.config import REPORT_DIR

logger = get_logger("profile")

def main(cfg_path: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths = load_config_data_paths(cfg_path)
    tables = load_tables(paths)

    # Basic checks: required columns presence
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    sch = cfg["schema"]

    errors = []
    for name, df in tables.items():
        if name not in sch:  # players/events/matches/squads expected
            continue
        req = sch[name].get("required", [])
        missing = [c for c in req if c not in df.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            if df["match_date"].isna().all():
                errors.append(f"{name}: match_date entirely NaT — check format")

    if errors:
        logger.warning("SCHEMA WARNINGS/ERRORS:")
        for e in errors:
            logger.warning(f"  - {e}")

    # Merge quick sample for profile
    # Keep sample to reduce HTML size if data is huge
    sample = tables["matches"].copy()
    sample = sample.sample(min(len(sample), 5000), random_state=17)
    report_path = Path(cfg["profile_report_path"]) if "profile_report_path" in cfg else (REPORT_DIR / "data_profile.html")
    logger.info(f"Generating profiling report at: {report_path}")
    profile = ProfileReport(sample, title="Week-1 Data Profile", minimal=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(report_path)

    logger.info("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
