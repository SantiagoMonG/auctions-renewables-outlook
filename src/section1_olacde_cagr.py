import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
DATA_PATH = Path("data/processed/olacde/installed_capacity_cagr.csv")
OUTPUT_PATH = Path("data/processed/olacde/section1_capacity_growth_by_period.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Separate utility and distributed solar before altering original
df_utility = df[df['Source'] == 'Solar'].copy()
df_distributed = df[df['Source'] == 'Distributed solar'].copy()

# Combine their numeric values
df_total_solar = df_utility.drop(columns='Source').copy()
df_total_solar.iloc[:, :] += df_distributed.drop(columns='Source').values
df_total_solar.insert(0, 'Source', 'Total Solar')

# Filter out individual solar rows
df_filtered = df[~df['Source'].isin(['Solar', 'Distributed solar'])]

# Final dataset includes all original sources except solar, plus aggregated total solar
df_final = pd.concat([df_filtered, df_total_solar], ignore_index=True)

# Period definitions
periods = {
    "2000-2024": ("2000", "2024"),
    "2000-2012": ("2000", "2012"),
    "2012-2018": ("2012", "2018"),
    "2018-2024": ("2018", "2024")
}

# CAGR function
def calc_cagr_safe(start, end, years):
    if pd.isna(start) or pd.isna(end) or start <= 0 or years <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1

# Recalculate with smarter solar handling
results = []
for _, row in df_final.iterrows():
    tech = row['Source']
    for period, (start_y, end_y) in periods.items():
        start_val = row.get(start_y)
        end_val = row.get(end_y)

        # Adjust if total solar and start year is before solar existed
        if tech == "Total Solar" and (pd.isna(start_val) or start_val == 0):
            # Find first nonzero year for solar
            nonzero_years = [y for y in row.index[1:] if row[y] > 0 and y <= end_y]
            if nonzero_years:
                adjusted_start_y = nonzero_years[0]
                start_val = row[adjusted_start_y]
                years = int(end_y) - int(adjusted_start_y)
            else:
                start_val, years = np.nan, 0
        else:
            years = int(end_y) - int(start_y)

        cagr = calc_cagr_safe(start_val, end_val, years)
        net_add = end_val - start_val if pd.notnull(start_val) and pd.notnull(end_val) else np.nan

        results.append({
            "Technology": tech,
            "Period": period,
            "Start (MW)": round(start_val, 2) if pd.notnull(start_val) else "—",
            "End (MW)": round(end_val, 2) if pd.notnull(end_val) else "—",
            "Net Additions (MW)": round(net_add, 2) if pd.notnull(net_add) else "—",
            "CAGR (%)": round(cagr * 100, 2) if pd.notnull(cagr) else "—"
        })

# Save results
df_out = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"CAGR results saved to {OUTPUT_PATH}")
