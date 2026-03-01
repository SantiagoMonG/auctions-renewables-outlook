import re
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

def read_adme_series10min_text(path: str) -> pd.DataFrame:
    # Lee como texto (son CSV “disfrazados”)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Limpia comentarios // y líneas vacías
    clean = []
    for ln in lines:
        ln = ln.split("//")[0].strip()
        if ln:
            clean.append(ln)

    # Header mínimo conocido (según lo que viste)
    nseries = int(clean[0].split(",")[0])                 # "9,NSeries,..."
    y, m, d, hh, mm, ss = map(int, clean[1].split(",")[:6])  # "2019,1,1,0,0,0,..."
    dt0 = datetime(y, m, d, hh, mm, ss)
    step_h = float(clean[2].split(",")[0])                # "0.166666667,..."
    step = timedelta(hours=step_h)

    # Ahora buscamos dónde empiezan los datos numéricos “puros”
    data_lines = []
    for ln in clean:
        parts = [p for p in ln.split(",") if p != ""]
        # línea de datos típica: solo números (y puntos) y muchas columnas
        if len(parts) >= nseries and all(re.fullmatch(r"-?\d+(\.\d+)?", p) for p in parts[:nseries]):
            data_lines.append(parts[:nseries])

    if not data_lines:
        raise ValueError(f"No encontré bloque de datos numéricos en {path}")

    # Construye dataframe
    df = pd.DataFrame(data_lines, columns=[f"s{i+1}" for i in range(nseries)]).astype(float)
    df["timestamp"] = [dt0 + i*step for i in range(len(df))]
    return df

if __name__ == "__main__":
    import os, glob

    RAW_DIR = "data/raw/country_case/Uruguay/eolica1"

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*_series10min.sas")))
    if not files:
        raise FileNotFoundError(f"No encontré *_series10min.sas en {RAW_DIR}")

    f = files[0]
    df = read_adme_series10min_text(f)

    print("\n=== Archivo (ejemplo):", os.path.basename(f), "===")
    print("Columnas:", df.columns.tolist())
    print(df.head())
    print(df.describe())

    # 🔴 Por ahora paramos aquí para inspeccionar (luego lo quitamos)
    # Cuando ya sepamos qué columna s# es la potencia MW, seguimos con el agregado.