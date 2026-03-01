import os, glob, re
import pandas as pd
from datetime import datetime

RAW_DIR = "data/raw/country_case/Uruguay/eolica1"
YEAR_FILTER = "2022"          # set to None if you want full period
STEP_MIN = 10                 # these are 10-min series
MISSING_SENTINEL = -99999999

def read_text_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()

def parse_old_format(path: str) -> pd.DataFrame:
    """
    Old format example: first lines like '9,NSeries,...' then 2019,1,1,0,0,0 then step_h then numbers.
    Returns df with columns s1..sN and timestamp.
    """
    lines = read_text_lines(path)
    clean = []
    for ln in lines:
        ln = ln.split("//")[0].strip()
        if ln:
            clean.append(ln)

    nseries = int(clean[0].split(",")[0])
    y, m, d, hh, mm, ss = map(int, clean[1].split(",")[:6])
    dt0 = datetime(y, m, d, hh, mm, ss)

    # step hours (should be 0.1666..)
    step_h = float(clean[2].split(",")[0])
    minutes = int(round(step_h * 60))

    data_lines = []
    num_pat = re.compile(r"-?\d+(?:\.\d+)?")
    for ln in clean:
        parts = [p for p in ln.split(",") if p != ""]
        if len(parts) >= nseries and all(re.fullmatch(r"-?\d+(\.\d+)?", p) for p in parts[:nseries]):
            data_lines.append(parts[:nseries])

    df = pd.DataFrame(data_lines, columns=[f"s{i+1}" for i in range(nseries)]).astype(float)
    df["timestamp"] = pd.date_range(dt0, periods=len(df), freq=f"{minutes}min")
    return df

def parse_version_format(path: str) -> pd.DataFrame:
    """
    New format example: starts with 'VERSION_FORMATO_SERIES:' and has a names line with 'H:' and a t_serial column.
    Returns df with columns (t_serial + short signal names) and timestamp.
    """
    lines = read_text_lines(path)
    # locate names line (starts with comma and includes H:)
    names_idx = None
    for i, ln in enumerate(lines[:200]):
        if ln.lstrip().startswith(",") and ("H:" in ln):
            names_idx = i
            names_line = ln.strip()
            break
    if names_idx is None:
        raise ValueError("No names line found")

    raw_names = [s.strip() for s in names_line.split(",") if s.strip()]
    short_names = []
    for rn in raw_names:
        parts = rn.split(":")
        short_names.append(parts[1].strip() if len(parts) >= 2 else rn)

    # data block begins after names line
    colnames = ["t_serial"] + short_names
    df = pd.read_csv(path, skiprows=names_idx + 1, header=None, names=colnames, engine="python")
    df = df.replace(MISSING_SENTINEL, pd.NA)

    # timestamp from Excel serial day
    df["timestamp"] = pd.to_datetime(df["t_serial"], unit="D", origin="1899-12-30", errors="coerce")
    return df

def load_power_series(path: str) -> pd.Series:
    """
    Returns a MW series indexed by timestamp.
    Heuristic:
      - if file contains 'VERSION_FORMATO_SERIES:' => use version format and power column 'pot' if present
      - else old format and power column s3 (based on your inspection)
    """
    # quick peek
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()

    if "VERSION_FORMATO_SERIES" in first:
        df = parse_version_format(path)
        if "pot" not in df.columns:
            raise ValueError(f"{os.path.basename(path)}: no 'pot' column found. cols={df.columns.tolist()}")
        mw = pd.to_numeric(df["pot"], errors="coerce")
    else:
        df = parse_old_format(path)
        if "s3" not in df.columns:
            raise ValueError(f"{os.path.basename(path)}: no 's3' column found. cols={df.columns.tolist()}")
        mw = pd.to_numeric(df["s3"], errors="coerce")

    s = pd.Series(mw.values, index=pd.to_datetime(df["timestamp"], errors="coerce"))
    s = s.dropna().clip(lower=0)
    if YEAR_FILTER:
        s = s.loc[YEAR_FILTER]
    return s.sort_index()

def daily_mwh_from_mw_10min(mw: pd.Series) -> pd.Series:
    # 10-min energy: MW * (10/60) hours
    return mw.resample("D").sum() * (STEP_MIN / 60)

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*_series10min.sas")))
    if not files:
        raise FileNotFoundError(f"No files found in {RAW_DIR}")

    rows = []
    for f in files:
        try:
            mw = load_power_series(f)
            dmwh = daily_mwh_from_mw_10min(mw).dropna()
            if len(dmwh) < 10:
                continue
            rows.append({
                "file": os.path.basename(f),
                "median_daily_MWh": float(dmwh.median()),
                "mean_daily_MWh": float(dmwh.mean()),
                "max_daily_MWh": float(dmwh.max()),
                "n_days": int(dmwh.shape[0]),
                "period_start": str(dmwh.index.min().date()),
                "period_end": str(dmwh.index.max().date()),
            })
        except Exception as e:
            rows.append({"file": os.path.basename(f), "error": str(e)})

    out = pd.DataFrame(rows)
    ok = out[out["error"].isna()] if "error" in out.columns else out
    ok = ok.sort_values("median_daily_MWh", ascending=False)

    print("\nTop 10 by MEDIAN daily MWh:")
    print(ok.head(10).to_string(index=False))

    # Save ranking
    os.makedirs("outputs/tables", exist_ok=True)
    ok.to_csv("outputs/tables/uruguay_wind_rank_by_median_daily_MWh.csv", index=False)
    print("\nSaved: outputs/tables/uruguay_wind_rank_by_median_daily_MWh.csv")