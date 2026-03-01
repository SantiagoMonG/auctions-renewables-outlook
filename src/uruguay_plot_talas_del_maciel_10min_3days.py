import os
import pandas as pd
import matplotlib.pyplot as plt

FILE = "data/raw/country_case/Uruguay/eolica1/c13_series10min.sas"
PLANT_NAME = "Talas del Maciel I"
YEAR_FILTER = "2022"

OUT_CSV = "outputs/tables/talas_del_maciel_I_10min_3days_2022.csv"
OUT_PNG = "outputs/figures_draft/talas_del_maciel_I_10min_3days_2022.png"

# Sentinelas observados en el archivo (valores absurdos)
POT_MISSING_THRESHOLD = -1000       # pot ~ [0, 50], así que < -1000 es missing
GENERIC_MISSING_THRESHOLD = -1e6    # para otras variables si se necesitara

def read_adme_old_named_series(path: str) -> pd.DataFrame:
    """
    Lee archivos tipo c13_series10min.sas (formato 'viejo' con fila de nombres).
    Devuelve DataFrame con columnas:
      t_serial, vel, dir, pot, tem, pre, hum, cgm, dis, Var(vel), timestamp
    """
    # La fila 6 (1-index) contiene nombres y la data empieza justo después.
    # En tu archivo: las primeras 5 líneas son metadatos, la 6 es nombres.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = [next(f).strip("\n") for _ in range(6)]

    names_line = header[5]  # ",vel,dir,pot,..."
    names = [n.strip() for n in names_line.split(",") if n.strip()]
    colnames = ["t_serial"] + names

    df = pd.read_csv(path, skiprows=6, header=None, names=colnames, engine="python")

    # Convertir a numérico
    for c in colnames:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Timestamp desde Excel serial day
    df["timestamp"] = pd.to_datetime(df["t_serial"], unit="D", origin="1899-12-30", errors="coerce")
    return df

def build_day_frame(series_mw: pd.Series, day_date, label: str, daily_mwh_value: float) -> pd.DataFrame:
    day = series_mw.loc[str(day_date)].copy()
    minutes_since_midnight = (day.index.hour * 60 + day.index.minute).astype(int)
    out = pd.DataFrame({
        "central": PLANT_NAME,
        "archivo": os.path.basename(FILE),
        "tipo_dia": label,
        "fecha": str(day_date),
        "dia_semana": pd.Timestamp(day_date).day_name(),
        "hora_min_desde_00": minutes_since_midnight,
        "hhmm": [f"{h:02d}:{m:02d}" for h, m in zip(day.index.hour, day.index.minute)],
        "mw": day.values,
        "energia_diaria_mwh": daily_mwh_value,
    })
    return out

if __name__ == "__main__":
    df = read_adme_old_named_series(FILE)

    # Filtrar año
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if YEAR_FILTER:
        df = df.loc[YEAR_FILTER]

    # Potencia MW: 'pot' (limpiar sentinelas)
    mw = df["pot"].copy()
    mw = mw.mask(mw < POT_MISSING_THRESHOLD)  # -556000 -> NaN
    mw = mw.clip(lower=0)                     # potencia no negativa

    # Para elegir días "representativos", evitamos días con muchos faltantes:
    expected_points_per_day = int((24 * 60) / 10)  # 144 para 10-min
    valid_counts = mw.resample("D").count()
    good_days = valid_counts[valid_counts >= 0.95 * expected_points_per_day].index  # >=95% puntos válidos

    mw_good = mw[mw.index.normalize().isin(good_days)]

    # Energía diaria (MWh): sum(MW)*10/60
    daily_mwh = mw_good.resample("D").sum() * (10 / 60)
    daily_mwh = daily_mwh.dropna()

    calm_day = daily_mwh.idxmin().date()
    windy_day = daily_mwh.idxmax().date()
    median_day = daily_mwh.sort_values().index[len(daily_mwh) // 2].date()

    print("Calmado:", calm_day, "MWh:", float(daily_mwh.loc[str(calm_day)]))
    print("Mediano:", median_day, "MWh:", float(daily_mwh.loc[str(median_day)]))
    print("Ventoso:", windy_day, "MWh:", float(daily_mwh.loc[str(windy_day)]))

    # Dataset tidy
    calm_df = build_day_frame(mw, calm_day, "Calmado", float(daily_mwh.loc[str(calm_day)]))
    med_df  = build_day_frame(mw, median_day, "Mediano", float(daily_mwh.loc[str(median_day)]))
    wnd_df  = build_day_frame(mw, windy_day, "Ventoso", float(daily_mwh.loc[str(windy_day)]))
    plot_df = pd.concat([calm_df, med_df, wnd_df], ignore_index=True)

    # Guardar CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    plot_df.to_csv(OUT_CSV, index=False)
    print("CSV guardado en:", OUT_CSV)

    # Plot con eje X = hora del día
    plt.figure(figsize=(11, 4))

    def etiqueta(sub):
        d = sub["fecha"].iloc[0]
        wd = sub["dia_semana"].iloc[0]
        typ = sub["tipo_dia"].iloc[0]
        mwh_val = sub["energia_diaria_mwh"].iloc[0]
        return f"{typ} {d} ({wd}) – {mwh_val:.0f} MWh"

    for typ in ["Calmado", "Mediano", "Ventoso"]:
        sub = plot_df[plot_df["tipo_dia"] == typ].sort_values("hora_min_desde_00")
        plt.plot(sub["hora_min_desde_00"], sub["mw"], label=etiqueta(sub))

    xticks = list(range(0, 24 * 60, 120))
    xticklabels = [f"{h:02d}:00" for h in range(0, 24, 2)]
    plt.xticks(xticks, xticklabels)

    plt.title(f"{PLANT_NAME} – Generación eólica cada 10 minutos: día calmado, mediano y ventoso (2022)")
    plt.xlabel("Hora del día")
    plt.ylabel("MW")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    print("PNG guardado en:", OUT_PNG)

    plt.show()