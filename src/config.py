# src/config.py
"""
Project config: paths + constants.

Run scripts from repo root (Auctions/), e.g.:
  python -m src.section1_ember_00_extract
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES_DRAFT_DIR = OUTPUTS_DIR / "figures_draft"


@dataclass(frozen=True)
class EmberSection1Config:
    """Config for Section 1 Ember extraction."""
    ember_raw_csv: Path = RAW_DIR / "ember" / "ember_monthly_dec2025.csv"
    start_date: str = "2018-01-01"
    end_date: str = "2025-10-01"

    countries: tuple[str, ...] = (
        "Uruguay",
        "Peru",
        "Mexico",
        "Ecuador",
        "Dominican Republic (the)",
        "Costa Rica",
        "Chile",
        "Colombia",
        "Argentina",
        "Bolivia",
        "Brazil",
        "El Salvador",
    )

    ember_interim_out: Path = INTERIM_DIR / "ember" / "ember_generation_pct_lac_2018_2025.csv"
    coverage_table_out: Path = OUTPUT_TABLES_DIR / "ember_section1_coverage.csv"