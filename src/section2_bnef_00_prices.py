"""
BNEF Auctions - Section 2 (Prices)

This script produces robust descriptive price statistics for competitive clean energy auctions
using BloombergNEF "Average price (US dollars per megawatt-hour)" (USD/MWh).

Design choices (agreed in the note):
- Universe: completed auctions, 2004–2024
- Price sample: drop rows without a USD/MWh price
- Core price analysis: Solar and Wind only (exclude multi-tech/storage-backed records)
- Annual reporting rules:
    - show median + IQR when N>=20
    - show median only when 10<=N<20
    - omit annual stats when N<10
- Global chart start years: Solar 2014; Wind 2015
- Robustness:
    - Core screen: price >= $1/MWh
    - Sensitivity: exclude price < $10/MWh (strict)
    - Within-group winsorization on CORE by (tech, year) for N>=20

Outputs:
- outputs/tables/*.csv
- outputs/figures_draft/*.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# ## 0) Configuration (column names, paths, and analysis rules)
# =============================================================================

PriceTrack = Literal["raw", "core", "strict", "winsorized"]

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    input_csv: Path
    out_tables: Path
    out_figs: Path

PRICE_COL = "Average price (US dollars per megawatt-hour)"
YEAR_COL = "Auction year"
STATUS_COL = "Status"
REGION_COL = "Region"
SUBSECTOR_COL = "Subsector of awarded capacity"

@dataclass(frozen=True)
class GlobalPriceRules:
    solar_start_year: int = 2014
    wind_start_year: int = 2015
    min_price_core: float = 1.0
    min_price_strict: float = 10.0
    show_median_n: int = 10
    show_iqr_n: int = 20

# =============================================================================
# ## 1) CLI / path resolution
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute robust BNEF auction price statistics (Solar/Wind)."
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of src/).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to auctions_bnef.csv (default: <repo-root>/data/raw/bnef/auctions_bnef.csv).",
    )
    p.add_argument(
        "--out-tables",
        type=Path,
        default=None,
        help="Output tables directory (default: <repo-root>/outputs/tables).",
    )
    p.add_argument(
        "--out-figs",
        type=Path,
        default=None,
        help="Output figures directory (default: <repo-root>/outputs/figures_draft).",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    repo_root: Path = args.repo_root
    input_csv = args.input or (repo_root / "data" / "raw" / "bnef" / "auctions_bnef.csv")
    out_tables = args.out_tables or (repo_root / "outputs" / "tables")
    out_figs = args.out_figs or (repo_root / "outputs" / "figures_draft")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    return Paths(repo_root=repo_root, input_csv=input_csv, out_tables=out_tables, out_figs=out_figs)

# =============================================================================
# ## 2) Load & clean raw data into the analysis universe (Completed, 2004–2024)
# =============================================================================

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def load_universe(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="cp1252")

    for col in [STATUS_COL, REGION_COL, SUBSECTOR_COL]:
        if col in df.columns:
            df[col] = _clean_str(df[col])

    df[YEAR_COL] = pd.to_numeric(df.get(YEAR_COL), errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df.get(PRICE_COL), errors="coerce")

    u = df[_clean_str(df[STATUS_COL]).str.lower().eq("completed")].copy()
    u = u[(u[YEAR_COL] >= 2004) & (u[YEAR_COL] <= 2024)].copy()
    u[YEAR_COL] = u[YEAR_COL].astype(int)
    return u

# =============================================================================
# ## 3) Build the price sample (drop rows with missing USD/MWh)
# =============================================================================

def drop_missing_prices(u: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    n_universe = int(len(u))
    priced = u[u[PRICE_COL].notna()].copy()
    n_priced = int(len(priced))
    meta = {
        "n_universe_completed_2004_2024": n_universe,
        "n_priced_rows": n_priced,
        "n_missing_price_rows": n_universe - n_priced,
    }
    return priced, meta

# =============================================================================
# ## 4) Classify Solar vs Wind for the price section (exclude multi-tech rows)
# =============================================================================

def classify_pv_onshore_wind_only(priced: pd.DataFrame) -> pd.DataFrame:

    s = priced[SUBSECTOR_COL].astype(str).str.strip().str.lower()

    is_pv = s.eq("pv")
    is_onshore_wind = s.eq("onshore wind")

    out = priced[is_pv | is_onshore_wind].copy()
    out["tech"] = np.where(is_pv.loc[out.index], "PV", "Onshore wind")
    return out


def audit_subsector_counts(priced: pd.DataFrame) -> pd.DataFrame:
    """
    Optional QA helper: show priced observation counts by subsector label
    (useful to document what is excluded: offshore, floating offshore, csp, agrivoltaic, etc.).
    """
    s = priced[SUBSECTOR_COL].astype(str).str.strip().str.lower()
    return (
        s.value_counts(dropna=False)
         .rename_axis("subsector_norm")
         .reset_index(name="n_priced_obs")
    )

# =============================================================================
# ## 5) Define analysis tracks (RAW vs CORE vs STRICT)
# =============================================================================

def apply_track(df: pd.DataFrame, track: PriceTrack, rules: GlobalPriceRules) -> pd.DataFrame:
    if track == "raw":
        return df.copy()
    if track == "core":
        return df[df[PRICE_COL] >= rules.min_price_core].copy()
    if track == "strict":
        return df[df[PRICE_COL] >= rules.min_price_strict].copy()
    raise ValueError(f"Unknown track: {track}")

# =============================================================================
# ## 6) Robustness: within-(tech×year) winsorization on CORE
# =============================================================================


def winsorize_within_tech_year(core_df: pd.DataFrame, rules: GlobalPriceRules) -> pd.DataFrame:
    d = core_df.copy()

    def _winsorize(x: pd.Series) -> pd.Series:
        n = int(x.size)
        if n < rules.show_iqr_n:
            return x
        lo_q, hi_q = (0.05, 0.95) if n < 100 else (0.01, 0.99)
        lo = float(x.quantile(lo_q))
        hi = float(x.quantile(hi_q))
        return x.clip(lower=lo, upper=hi)

    d["price_wins"] = d.groupby(["tech", YEAR_COL], group_keys=False)[PRICE_COL].apply(_winsorize)
    return d

# =============================================================================
# ## 7) Compute annual statistics under reporting rules (median, IQR, display flags)
# =============================================================================

def annual_stats(
    df: pd.DataFrame,
    tech: Literal["PV", "Onshore wind"],
    start_year: int,
    price_field: str,
    rules: GlobalPriceRules,
) -> pd.DataFrame:
    d = df[df["tech"].eq(tech)].copy()
    d = d[d[YEAR_COL] >= start_year].copy()

    g = d.groupby(YEAR_COL)[price_field]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "median": g.median(),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
        }
    ).reset_index()

    out.rename(columns={YEAR_COL: "year"}, inplace=True)
    out["IQR"] = out["p75"] - out["p25"]
    out["show_median"] = out["n"] >= rules.show_median_n
    out["show_IQR"] = out["n"] >= rules.show_iqr_n
    return out.sort_values("year").reset_index(drop=True)


# =============================================================================
# ## 8) Make figures (median line + IQR whiskers only when N>=20)
# =============================================================================


def plot_median_iqr(ts: pd.DataFrame, title: str, out_path: Path) -> None:
    d = ts[ts["show_median"]].copy()
    if d.empty:
        return

    years = d["year"].to_numpy()
    med = d["median"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(years, med, marker="o")

    d_i = d[d["show_IQR"]].copy()
    if not d_i.empty:
        yerr_lower = (d_i["median"] - d_i["p25"]).to_numpy()
        yerr_upper = (d_i["p75"] - d_i["median"]).to_numpy()
        ax.errorbar(d_i["year"], d_i["median"], yerr=[yerr_lower, yerr_upper], fmt="none", capsize=3)

    ax.set_title(title)
    ax.set_xlabel("Auction year")
    ax.set_ylabel("Average price (USD/MWh) — median (IQR where n≥20)")
    ax.set_xticks(sorted(set(years.tolist())))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# =============================================================================
# ## 9) Robustness summaries (strict-vs-core and winsorized-vs-core max shifts)
# =============================================================================

def sensitivity_max_median_shift(core_ts: pd.DataFrame, strict_ts: pd.DataFrame) -> float:
    a = core_ts[["year", "median", "show_median"]].copy()
    b = strict_ts[["year", "median"]].copy()
    m = a.merge(b, on="year", how="left", suffixes=("_core", "_strict"))
    m = m[m["show_median"]].copy()
    if m.empty:
        return 0.0
    return float((m["median_strict"] - m["median_core"]).abs().max())


def winsorization_max_shifts(core_ts: pd.DataFrame, wins_ts: pd.DataFrame) -> dict[str, float]:
    a = core_ts[["year", "median", "p25", "p75", "show_median"]].copy()
    b = wins_ts[["year", "median", "p25", "p75"]].copy()
    m = a.merge(b, on="year", how="left", suffixes=("_core", "_wins"))
    m = m[m["show_median"]].copy()
    if m.empty:
        return {"max_median_diff": 0.0, "max_p25_diff": 0.0, "max_p75_diff": 0.0}
    return {
        "max_median_diff": float((m["median_wins"] - m["median_core"]).abs().max()),
        "max_p25_diff": float((m["p25_wins"] - m["p25_core"]).abs().max()),
        "max_p75_diff": float((m["p75_wins"] - m["p75_core"]).abs().max()),
    }


# =============================================================================
# ## 10) Orchestrate: run the full pipeline and write outputs
# =============================================================================

def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    rules = GlobalPriceRules()

    # ## 10.1 Load base universe (Completed, 2004–2024)
    universe = load_universe(paths.input_csv)

    # ## 10.2 Drop missing prices (price section sample)
    priced, meta = drop_missing_prices(universe)

    # ## 10.3 Keep only Solar/Wind (exclude multi-tech)
    sw = classify_pv_onshore_wind_only(priced)

    # ## 10.4 Build analysis tracks
    raw = apply_track(sw, "raw", rules)
    core = apply_track(sw, "core", rules)
    strict = apply_track(core, "strict", rules)

    # ## 10.5 Winsorize within (tech×year) on CORE for robustness
    core_w = winsorize_within_tech_year(core, rules)

    # ## 10.6 Compute annual stats
    raw_solar = annual_stats(raw, "PV", rules.solar_start_year, PRICE_COL, rules)
    raw_wind = annual_stats(raw, "Onshore wind", rules.wind_start_year, PRICE_COL, rules)

    core_solar = annual_stats(core, "PV", rules.solar_start_year, PRICE_COL, rules)
    core_wind = annual_stats(core, "Onshore wind", rules.wind_start_year, PRICE_COL, rules)

    strict_solar = annual_stats(strict, "PV", rules.solar_start_year, PRICE_COL, rules)
    strict_wind = annual_stats(strict, "Onshore wind", rules.wind_start_year, PRICE_COL, rules)

    wins_solar = annual_stats(core_w, "PV", rules.solar_start_year, "price_wins", rules)
    wins_wind = annual_stats(core_w, "Onshore wind", rules.wind_start_year, "price_wins", rules)

    # ## 10.7 Robustness summaries for the technical note
    sens = pd.DataFrame(
        [
            {
                "tech": "PV",
                "max_abs_median_shift_strict_vs_core": round(
                    sensitivity_max_median_shift(core_solar, strict_solar), 4
                ),
            },
            {
                "tech": "Onshore wind",
                "max_abs_median_shift_strict_vs_core": round(
                    sensitivity_max_median_shift(core_wind, strict_wind), 4
                ),
            },
        ]
    )

    win_imp = pd.DataFrame(
        [
            {"tech": "PV", **winsorization_max_shifts(core_solar, wins_solar)},
            {"tech": "Onshore wind", **winsorization_max_shifts(core_wind, wins_wind)},
        ]
    )

    # ## 10.8 Write tables
    pd.DataFrame([meta]).to_csv(paths.out_tables / "prices_step1_price_availability.csv", index=False)

    raw_solar.to_csv(paths.out_tables / "prices_global_raw_solar_annual.csv", index=False)
    raw_wind.to_csv(paths.out_tables / "prices_global_raw_wind_annual.csv", index=False)

    core_solar.to_csv(paths.out_tables / "prices_global_core_ge1_solar_annual.csv", index=False)
    core_wind.to_csv(paths.out_tables / "prices_global_core_ge1_wind_annual.csv", index=False)

    strict_solar.to_csv(paths.out_tables / "prices_global_strict_ge10_solar_annual.csv", index=False)
    strict_wind.to_csv(paths.out_tables / "prices_global_strict_ge10_wind_annual.csv", index=False)

    wins_solar.to_csv(paths.out_tables / "prices_global_core_ge1_winsor_solar_annual.csv", index=False)
    wins_wind.to_csv(paths.out_tables / "prices_global_core_ge1_winsor_wind_annual.csv", index=False)

    sens.to_csv(paths.out_tables / "prices_global_sensitivity_max_median_shift.csv", index=False)
    win_imp.to_csv(paths.out_tables / "prices_global_winsorization_impact_max_shifts.csv", index=False)

    # ## 10.9 Write figures (winsorized core series)
    plot_median_iqr(
        wins_solar,
        "Global solar auction prices (median; IQR when n≥20) — core ($≥1) + within-year winsorization",
        paths.out_figs / "fig_section2_global_prices_solar_median_iqr.png",
    )
    plot_median_iqr(
        wins_wind,
        "Global wind auction prices (median; IQR when n≥20) — core ($≥1) + within-year winsorization",
        paths.out_figs / "fig_section2_global_prices_wind_median_iqr.png",
    )

    # ## 10.10 Console summary for quick QA
    miss_pct = meta["n_missing_price_rows"] / meta["n_universe_completed_2004_2024"] * 100
    print("=== BNEF Prices (Global, Solar/Wind) ===")
    print(f"Input: {paths.input_csv}")
    print(f"Universe (Completed, 2004–2024): {meta['n_universe_completed_2004_2024']}")
    print(f"Missing price dropped (for price section): {meta['n_missing_price_rows']} ({miss_pct:.1f}%)")
    print(f"Price observations kept (any tech): {meta['n_priced_rows']}")
    print(f"Solar/Wind priced observations (raw): {len(raw)}")
    print(f"Core (price>=1): {len(core)}; Strict (price>=10): {len(strict)}")
    print(f"Start years: Solar {rules.solar_start_year}; Wind {rules.wind_start_year}")
    print("Robustness:")
    print(sens.to_string(index=False))
    print(win_imp.round(4).to_string(index=False))
    print(f"Tables -> {paths.out_tables}")
    print(f"Figures -> {paths.out_figs}")


if __name__ == "__main__":
    main()