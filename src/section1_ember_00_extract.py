# src/section1_ember_00_extract.py
"""
Section 1 (Ember) — Step 0: extract clean subset + basic audit.

Outputs:
  - data/interim/ember/ember_generation_pct_lac_2018_2025.csv
  - outputs/tables/ember_section1_coverage.csv

Run (from repo root Auctions/):
  python -m src.section1_ember_00_extract
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EmberSection1Config

REQUIRED_COLUMNS = {
    "Area",
    "Ember region",
    "Date",
    "Category",
    "Subcategory",
    "Variable",
    "Unit",
    "Value",
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_ember(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Ember file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    bad = int(df["Date"].isna().sum())
    if bad:
        raise ValueError(f"Found {bad} rows with unparseable Date values.")
    return df


def extract_generation_share_pct(df: pd.DataFrame, cfg: EmberSection1Config) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Ember region"].eq("Latin America and Caribbean")]
    out = out[out["Area"].isin(cfg.countries)]
    out = out[out["Date"].between(pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date))]

    out = out[out["Category"].eq("Electricity generation")]
    out = out[out["Unit"].eq("%")]

    out = out[["Area", "Date", "Category", "Subcategory", "Variable", "Unit", "Value"]].copy()
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out


def build_coverage_table(gen_pct: pd.DataFrame) -> pd.DataFrame:
    return (
        gen_pct.groupby("Area")["Date"]
        .agg(start="min", end="max", months="nunique")
        .reset_index()
        .rename(columns={"Area": "Country"})
        .sort_values(["start", "Country"])
    )


def main() -> int:
    cfg = EmberSection1Config()

    df = load_ember(cfg.ember_raw_csv)
    gen_pct = extract_generation_share_pct(df, cfg)

    if gen_pct.empty:
        raise RuntimeError("Filtered Ember subset is empty. Check filters and file path.")

    ensure_parent_dir(cfg.ember_interim_out)
    gen_pct.to_csv(cfg.ember_interim_out, index=False)

    coverage = build_coverage_table(gen_pct)
    ensure_parent_dir(cfg.coverage_table_out)
    coverage.to_csv(cfg.coverage_table_out, index=False)

    print("OK")
    print(f"Saved: {cfg.ember_interim_out}")
    print(f"Saved: {cfg.coverage_table_out}")
    print(f"Rows: {len(gen_pct):,} | Countries: {gen_pct['Area'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
