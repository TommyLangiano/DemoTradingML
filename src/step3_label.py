import pandas as pd
from pathlib import Path

# Percorsi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "EURUSD_M1_clean.parquet"
OUTPUT_FILE = DATA_DIR / "EURUSD_M1_labeled.parquet"

# Carica parquet pulito
df = pd.read_parquet(INPUT_FILE)

# Parametri
tp_pct = 0.001
sl_pct = 0.0005
horizon = 20

# Funzione etichettatura
def label_row(close_price, future_highs, future_lows, tp_pct, sl_pct):
    tp_level = close_price * (1 + tp_pct)
    sl_level = close_price * (1 - sl_pct)
    for h, l in zip(future_highs, future_lows):
        if h >= tp_level:
            return 1
        if l <= sl_level:
            return 0
    return 0

# Calcola etichette
highs = df["High"].to_numpy()
lows = df["Low"].to_numpy()
closes = df["Close"].to_numpy()

labels = []

for i in range(len(df) - horizon):
    future_highs = highs[i+1 : i+1+horizon]
    future_lows  = lows[i+1 : i+1+horizon]
    close_price  = closes[i]
    lbl = label_row(close_price, future_highs, future_lows, tp_pct, sl_pct)
    labels.append(lbl)

labels += [None] * horizon
df["Label"] = labels

df.to_parquet(OUTPUT_FILE)
print(f"✅ Salvato file etichettato: {OUTPUT_FILE}")