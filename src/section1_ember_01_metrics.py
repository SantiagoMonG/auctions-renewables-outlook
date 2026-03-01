# src/section1_ember_01_metrics.py
"""
Section 1 (Ember) — Step 1: compute descriptive system metrics.

Reads:
  - data/interim/ember/ember_generation_pct_lac_2018_2025.csv

Writes:
  - data/processed/ember/section1_ember_monthly_mix_panel.csv
  - data/processed/ember/section1_ember_mix_metrics.csv
  - data/processed/ember/section1_ember_qc.csv  (new: optional QA)

Run (from repo root Auctions/):
  python -m src.section1_ember_01_metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import EmberSection1Config, PROCESSED_DIR


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_interim(cfg: EmberSection1Config) -> pd.DataFrame:
    if not cfg.ember_interim_out.exists():
        raise FileNotFoundError(
            f"Interim Ember file not found: {cfg.ember_interim_out}\n"
            "Run: python -m src.section1_ember_00_extract"
        )
    df = pd.read_csv(cfg.ember_interim_out)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    if df["Date"].isna().any():
        bad = int(df["Date"].isna().sum())
        raise ValueError(f"Found {bad} rows with unparseable Date values.")
    df["Area"] = df["Area"].replace({"Dominican Republic (the)":"Dominican Republic"})
    return df


def pivot_country_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide panel: one row per Country-Month, columns are Ember Variables (Wind, Solar, Hydro, Fossil, etc.).
    """
    panel = (
        df.pivot_table(
            index=["Area", "Date"],
            columns="Variable",
            values="Value",
            aggfunc="mean",
        )
        .sort_index()
        .reset_index()
        .rename(columns={"Area": "Country"})
    )
    return panel


def compute_mix_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute four buckets (shares, %) per Country-Month:
      - VRE = Wind + Solar
      - Hydro = Hydro
      - Thermal = Fossil (fallback: sum fossil components if Fossil missing)
      - Other = Bioenergy + Other Renewables + Nuclear (if present; else 0)
    """
    df = panel.copy()

    for c in [
        "Wind",
        "Solar",
        "Hydro",
        "Fossil",
        "Gas",
        "Coal",
        "Other Fossil",
        "Gas and Other Fossil",
        "Bioenergy",
        "Other Renewables",
        "Nuclear",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    out = df[["Country", "Date"]].copy()
    out["VRE"] = df["Wind"].fillna(0) + df["Solar"].fillna(0)

    # Thermal fallback
    fossil_fallback = df["Gas"].fillna(0) + df["Coal"].fillna(0) + df["Other Fossil"].fillna(0)
    out["Thermal"] = np.where(df["Fossil"].notna(), df["Fossil"], fossil_fallback)

    out["Hydro"] = df["Hydro"]
    out["Other"] = (
        df["Bioenergy"].fillna(0) + df["Other Renewables"].fillna(0) + df["Nuclear"].fillna(0)
    )

    out["sum_check_pct"] = out[["VRE", "Hydro", "Thermal", "Other"]].sum(axis=1)

    out = out.sort_values(["Country","Date"]).reset_index(drop=True)
    return out


def regularize_monthly(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Reindex each country's data to a complete MS monthly range. Avoids irregular windows.
    """
    parts: List[pd.DataFrame] = []
    for country, g in df.groupby("Country", sort=False):
        g = g.sort_values(date_col).set_index(date_col)
        if len(g.index) == 0:
            continue
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx)
        g["Country"] = country
        parts.append(g.reset_index().rename(columns={"index": date_col}))
    return pd.concat(parts, ignore_index=True) if parts else df.copy()

def add_rolling_means(mix: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    out = mix.sort_values(["Country", "Date"]).copy()
    for col in ["VRE", "Hydro", "Thermal", "Other"]:
        out[f"{col}_12MA"] = (
            out.groupby("Country", sort=False)[col]
            .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
        )
    return out


def add_hydro_swing(
    mix: pd.DataFrame, trailing_months: int = 36
) -> pd.DataFrame:
    """
    HydroSwing(t) = Hydro%(t) - mean(Hydro%(t-36..t-1))
    Grouped rolling + grouped shift(1) to avoid cross-country spillover.
    """
    out = mix.sort_values(["Country", "Date"]).copy()
    baseline = out.groupby("Country", sort=False)["Hydro"].transform(
        lambda s: s.rolling(window=trailing_months, min_periods=trailing_months).mean().shift(1)
    )
    out["HydroSwing"] = out["Hydro"] - baseline
    return out


def compute_backup_response(mix: pd.DataFrame) -> pd.DataFrame:
    """
    Backup response on monthly changes (ΔHydro vs ΔThermal):
      - Pearson corr of Δs
      - Slope from ΔThermal ~ b*(−ΔHydro)
    """
    rows = []
    for country, g in mix.groupby("Country", sort=False):
        g = g.sort_values("Date")
        dh = g["Hydro"].diff()
        dt = g["Thermal"].diff()
        valid = pd.concat([dh, dt], axis=1, keys=["dHydro", "dThermal"]).dropna()
        if len(valid) >= 24:
            corr = float(valid["dHydro"].corr(valid["dThermal"]))
            x = (-valid["dHydro"]).to_numpy()
            y = valid["dThermal"].to_numpy()
            x = x - x.mean()  # center (why: stable slope)
            denom = float((x ** 2).sum())
            slope = float((x * y).sum() / denom) if denom != 0 else float("nan")
        else:
            corr = float("nan")
            slope = float("nan")
        rows.append(
            {
                "Country": country,
                "Backup response corr(ΔHydro,ΔThermal)": corr,
                "Backup slope: ΔThermal per 1pp Hydro drop": slope,
                "N_changes": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def summarize_country_metrics(mix: pd.DataFrame) -> pd.DataFrame:
    """
    Country-level summary:
      - VRE ramp-up (12MA end-start), using first/last non-NaN 12MA
      - Hydro swing min/max + dates
      - Hydro stress months (< -10 pp)
      - Coverage months (distinct monthly observations)
    """
    rows = []
    for country, g in mix.groupby("Country", sort=False):
        g = g.sort_values("Date").copy()

        # VRE ramp (12MA)
        vre_ma = g["VRE_12MA"].dropna()
        if len(vre_ma) >= 2:
            vre_ramp = float(vre_ma.iloc[-1] - vre_ma.iloc[0])
            start_date = str(g.loc[g["VRE_12MA"].notna(), "Date"].iloc[0].date())
        else:
            vre_ramp = float("nan")
            start_date = str(g["Date"].min().date()) if len(g) else ""

        end_date = str(g["Date"].max().date()) if len(g) else ""

        # Hydro swing
        hs = g["HydroSwing"].dropna()
        if len(hs):
            hs_min = float(hs.min())
            hs_max = float(hs.max())
            hs_min_date = str(g.loc[hs.idxmin(), "Date"].date())
            hs_max_date = str(g.loc[hs.idxmax(), "Date"].date())
        else:
            hs_min = float("nan")
            hs_max = float("nan")
            hs_min_date = ""
            hs_max_date = ""

        stress_count = int((g["HydroSwing"] < -10).sum(skipna=True))
        coverage_months = int(g["Date"].nunique())

        rows.append(
            {
                "Country": country,
                "Start (data)": start_date,
                "End (data)": end_date,
                "VRE ramp-up (pp, 12MA end-start)": vre_ramp,
                "Hydro swing min (pp)": hs_min,
                "Hydro swing min date": hs_min_date,
                "Hydro swing max (pp)": hs_max,
                "Hydro swing max date": hs_max_date,
                "Hydro stress months (<-10pp)": stress_count,
                "Coverage months": coverage_months,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("VRE ramp-up (pp, 12MA end-start)", ascending=False)


def build_qc_table(mix: pd.DataFrame) -> pd.DataFrame:
    """
    Simple QC: share sum close to 100 by month, and yearly coverage.
    """
    df = mix.copy()
    df["sum_dev_pp"] = df["sum_check_pct"] - 100.0
    # Yearly coverage
    df["Year"] = df["Date"].dt.year
    cov = (
        df.groupby(["Country", "Year"])["Date"]
        .nunique()
        .reset_index(name="months_available")
    )
    # Aggregate sum deviation per country-year
    dev = (
        df.groupby(["Country", "Year"])["sum_dev_pp"]
        .agg(avg_dev_pp="mean", p95_dev_pp=lambda s: float(np.nanpercentile(np.abs(s.dropna()), 95)) if s.notna().any() else np.nan)
        .reset_index()
    )
    return cov.merge(dev, on=["Country", "Year"], how="left")


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    cfg = EmberSection1Config()

    # Load
    df = load_interim(cfg)

    # Pivot (shares only already ensured upstream)
    panel = pivot_country_month(df)

    # Buckets
    mix = compute_mix_buckets(panel)

    # Regularize monthly index per country (why: stable rolling windows)
    mix = regularize_monthly(mix)

    # 12MA
    mix = add_rolling_means(mix, window=12)

    # HydroSwing (trailing 36 mo baseline, exclude current via grouped shift)
    mix = add_hydro_swing(mix, trailing_months=36)

    # Backup stats
    backup = compute_backup_response(mix)

    # Country metrics
    metrics = summarize_country_metrics(mix).merge(backup, on="Country", how="left")

    # QC table
    qc = build_qc_table(mix)

    # Save processed monthly panel
    panel_out = PROCESSED_DIR / "ember" / "section1_ember_monthly_mix_panel.csv"
    ensure_parent_dir(panel_out)
    mix.to_csv(panel_out, index=False)

    # Save country metrics
    metrics_out = PROCESSED_DIR / "ember" / "section1_ember_mix_metrics.csv"
    ensure_parent_dir(metrics_out)
    metrics.to_csv(metrics_out, index=False)

    # Save QC table (new)
    qc_out = PROCESSED_DIR / "ember" / "section1_ember_qc.csv"
    ensure_parent_dir(qc_out)
    qc.to_csv(qc_out, index=False)

    print("OK")
    print(f"Saved monthly panel: {panel_out}")
    print(f"Saved metrics:       {metrics_out}")
    print(f"Saved QC:            {qc_out}")
    print(f"Countries: {metrics['Country'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())