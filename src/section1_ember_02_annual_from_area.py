# src/section1_ember_02_annual_from_area.py
"""
Build annual LAC deltas from the single Area row (Latin America & the Caribbean).

Outputs:
  - data/processed/ember/section1_ember_annual_LAC_area_row.csv
  - data/processed/ember/section1_ember_annual_LAC_area_row_deltas.csv
  - outputs/figures_draft/section1_lac_annual_deltas_area_row.png

Run (from repo root):
  python -m src.section1_ember_02_annual_from_area
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DIR, PROCESSED_DIR, OUTPUT_FIGURES_DRAFT_DIR

RAW_ANNUAL = DATA_DIR / "raw" / "ember" / "ember_yearly_december_2025.csv"

# Common label variants; your file likely uses "Latin America and Caribbean"
AREA_LABEL_PREFERRED = [
    "Latin America and Caribbean",
    "Latin America and the Caribbean",
    "Latin America & the Caribbean",
    "Latin America & Caribbean",
]

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Try common encodings—Ember files vary."""
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    for col in ("Area", "Category", "Variable", "Unit"):
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()  # <-- FIXED: .str.strip()
            # collapse internal whitespace (defensive)
            d[col] = d[col].str.replace(r"\s+", " ", regex=True)
    d["Year"] = pd.to_numeric(d.get("Year", pd.NA), errors="coerce").astype("Int64")
    d["Value"] = pd.to_numeric(d.get("Value", pd.NA), errors="coerce")
    return d

def pick_area_label(df: pd.DataFrame) -> str:
    """Pick the exact LAC Area row from known variants; else fallback to first containing 'Latin America'."""
    areas = set(df["Area"].dropna().astype(str).tolist())
    for p in AREA_LABEL_PREFERRED:
        if p in areas:
            return p
    candidates = sorted([a for a in areas if "Latin America" in a])
    if candidates:
        return candidates[0]
    raise RuntimeError("No Area value containing 'Latin America' found in the file.")

def getcol(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    names_lower = [n.strip().lower() for n in names]
    matches = [c for c in frame.columns if str(c).strip().lower() in names_lower]
    if not matches:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[matches[0]], errors="coerce").fillna(0.0)

def build_buckets_from_area(df: pd.DataFrame, area_label: str) -> pd.DataFrame:
    """Compute annual buckets (TWh) and YoY deltas for the given Area label."""
    # Filter to area + needed slices
    use = df[
        (df["Area"] == area_label)
        & (df["Unit"] == "TWh")
        & (df["Category"].isin(["Electricity generation", "Electricity demand"]))
    ].copy()
    if use.empty:
        raise RuntimeError(
            f"No rows after filtering for Area='{area_label}', Unit='TWh', and generation/demand."
        )

    gen = use[use["Category"] == "Electricity generation"].copy()
    dem = use[use["Category"] == "Electricity demand"][["Year", "Value"]].rename(columns={"Value": "Demand_TWh"})

    # Pivot generation by Variable
    gen_p = (
        gen.pivot_table(index=["Year"], columns="Variable", values="Value", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    # Buckets
    wind = getcol(gen_p, ["Wind"])
    solar = getcol(gen_p, ["Solar"])
    hydro = getcol(gen_p, ["Hydro"])

    fossil_pref = getcol(gen_p, ["Fossil"])
    gas = getcol(gen_p, ["Gas"])
    coal = getcol(gen_p, ["Coal"])
    other_fossil = getcol(gen_p, ["Other Fossil"])
    fossil_fb = gas + coal + other_fossil
    fossil = np.where(fossil_pref.to_numpy() > 0, fossil_pref.to_numpy(), fossil_fb.to_numpy())

    bio = getcol(gen_p, ["Bioenergy"])
    other_ren = getcol(gen_p, ["Other Renewables", "Other Renewable"])
    nuclear = getcol(gen_p, ["Nuclear"])

    vre = wind + solar
    other_clean = bio + other_ren + nuclear  # Hydro excluded

    ann = (
        pd.DataFrame(
            {
                "Year": gen_p.index,
                "VRE_TWh": vre.values,
                "Hydro_TWh": hydro.values,
                "OtherClean_TWh": other_clean.values,
                "Fossil_TWh": fossil,
            }
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )

    # Merge demand
    ann = ann.merge(dem, on="Year", how="left")

    # YoY deltas
    for c in ["VRE_TWh", "Hydro_TWh", "OtherClean_TWh", "Fossil_TWh", "Demand_TWh"]:
        ann[f"Δ{c}"] = ann[c].diff()

    # Residual check (sum of bucket deltas)
    ann["ΔGen_sum_TWh"] = ann[["ΔVRE_TWh", "ΔHydro_TWh", "ΔOtherClean_TWh", "ΔFossil_TWh"]].sum(axis=1)

    return ann

def save_outputs(area_label: str, ann: pd.DataFrame) -> None:
    full_out = PROCESSED_DIR / "ember" / "section1_ember_annual_LAC_area_row.csv"
    deltas_out = PROCESSED_DIR / "ember" / "section1_ember_annual_LAC_area_row_deltas.csv"
    fig_out = OUTPUT_FIGURES_DRAFT_DIR / "section1_lac_annual_deltas_area_row.png"

    ensure_parent(full_out)
    ensure_parent(deltas_out)
    ensure_parent(fig_out)

    ann.to_csv(full_out, index=False)
    ann[
        ["Year", "ΔVRE_TWh", "ΔHydro_TWh", "ΔOtherClean_TWh", "ΔFossil_TWh", "ΔDemand_TWh", "ΔGen_sum_TWh"]
    ].to_csv(deltas_out, index=False)

    # Plot (single-axis; no explicit colors)
    years = ann["Year"]
    plt.figure(figsize=(10, 5))
    s1 = ann["ΔFossil_TWh"]
    plt.bar(years, s1, label="Δ Fossil")
    s2 = s1 + ann["ΔHydro_TWh"].fillna(0)
    plt.bar(years, ann["ΔHydro_TWh"], bottom=s1, label="Δ Hydro")
    s3 = s2 + ann["ΔVRE_TWh"].fillna(0)
    plt.bar(years, ann["ΔVRE_TWh"], bottom=s2, label="Δ VRE")
    plt.bar(years, ann["ΔOtherClean_TWh"], bottom=s3, label="Δ Other clean (ex-hydro)")
    plt.plot(years, ann["ΔDemand_TWh"], marker="o", label="Δ Demand")
    plt.title(f"{area_label} — Annual Δ generation by source vs demand (TWh)")
    plt.xlabel("Year")
    plt.ylabel("Change (TWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_out, dpi=160)
    plt.close()

    print("OK")
    print(f"Area used:           {area_label}")
    print(f"Saved table:         {full_out}")
    print(f"Saved deltas table:  {deltas_out}")
    print(f"Saved figure:        {fig_out}")

def main() -> int:
    if not RAW_ANNUAL.exists():
        raise FileNotFoundError(f"Missing annual Ember file: {RAW_ANNUAL}")
    df = safe_read_csv(RAW_ANNUAL)
    df = normalize_columns(df)
    area_label = pick_area_label(df)
    ann = build_buckets_from_area(df, area_label)
    save_outputs(area_label, ann)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
