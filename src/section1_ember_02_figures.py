# src/section1_ember_02_figures.py
"""
Section 1 (Ember) — Step 2: generate the hero scatter plot.

Reads:
  - data/processed/ember/section1_ember_mix_metrics.csv

Writes:
  - outputs/figures_draft/fig_section1_hero_scatter_vre_vs_hydrostress.png

Run (from repo root Auctions/):
  python -m src.section1_ember_02_figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, OUTPUT_FIGURES_DRAFT_DIR


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_metrics() -> pd.DataFrame:
    metrics_path = PROCESSED_DIR / "ember" / "section1_ember_mix_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics: {metrics_path}\n"
            "Run first: python -m src.section1_ember_01_metrics"
        )
    df = pd.read_csv(metrics_path)

    needed = [
        "Country",
        "VRE ramp-up (pp, 12MA end-start)",
        "Hydro swing min (pp)",
        "Backup response corr(ΔHydro,ΔThermal)",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics file missing columns: {missing}")

    return df


def make_hero_scatter(df: pd.DataFrame, out_path: Path) -> None:
    x = df["VRE ramp-up (pp, 12MA end-start)"].astype(float)
    y = df["Hydro swing min (pp)"].astype(float)  # more negative = worse stress
    backup_strength = df["Backup response corr(ΔHydro,ΔThermal)"].abs().astype(float)

    # Bubble size: keep stable and readable across datasets
    sizes = 600 * (backup_strength.clip(0, 1) ** 2 + 0.05)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=sizes)

    for _, r in df.iterrows():
        plt.annotate(
            r["Country"],
            (float(r["VRE ramp-up (pp, 12MA end-start)"]), float(r["Hydro swing min (pp)"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    plt.axvline(0, linewidth=1)
    plt.axhline(0, linewidth=1)

    plt.title("VRE ramp-up vs hydro stress (selected LAC countries, Ember monthly data)")
    plt.xlabel("VRE ramp-up (pp, Wind+Solar share; 12-month rolling avg, end minus start)")
    plt.ylabel("Worst hydro stress (pp, minimum Hydro swing vs trailing norm)\n(more negative = deeper hydro shortfalls)")
    plt.tight_layout()

    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main() -> int:
    out_path = OUTPUT_FIGURES_DRAFT_DIR / "fig_section1_hero_scatter_vre_vs_hydrostress.png"
    df = load_metrics()
    make_hero_scatter(df, out_path)
    print("OK")
    print(f"Saved hero scatter: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
