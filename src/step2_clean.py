import pandas as pd
from pathlib import Path

# Percorsi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_FILE = DATA_DIR / "EURUSD_M1.csv"
PARQUET_FILE = DATA_DIR / "EURUSD_M1_clean.parquet"

# Leggi CSV
df = pd.read_csv(CSV_FILE, sep="\t", skipinitialspace=True)
df.columns = [c.replace("<", "").replace(">", "").strip() for c in df.columns]
df["time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])

# Rinomina colonne
df = (
    df.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "TICKVOL": "Volume"})
    .set_index("time")
    .sort_index()
    [["Open", "High", "Low", "Close", "Volume"]]
)

# Salva parquet
df.to_parquet(PARQUET_FILE)
print(f"✅ Salvato file pulito: {PARQUET_FILE}")