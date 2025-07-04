#!/usr/bin/env python3
"""
Filtra il dataset etichettato in funzione del trend MA.
Conserva solo:
  • Label = 1 se Close > MA  → LONG context
  • Label = 0 se Close < MA  → SHORT context

Output: parquet filtrato + statistiche in console.
"""

import argparse, logging, sys
from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────
# Logging
# ─────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "filter_trend.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("filter_trend")

# ─────────────────────────────
# CLI
# ─────────────────────────────
p = argparse.ArgumentParser(description="Trend filter via Moving Average")
p.add_argument("--input",  default="data/processed/EURUSD_M1_labeled.parquet")
p.add_argument("--output", default="data/processed/EURUSD_M1_filtered.parquet")
p.add_argument("--ma", type=int, default=50, help="MA window length (bars)")
args = p.parse_args()

SRC = Path(args.input)
if not SRC.exists():
    L.error(f"Parquet etichettato non trovato: {SRC}")
    sys.exit(1)

# ─────────────────────────────
# Carica e calcola MA
# ─────────────────────────────
df = pd.read_parquet(SRC)
L.info(f"Rows loaded (labeled): {len(df):,}")

ma_len = args.ma
df["MA"] = df["Close"].rolling(window=ma_len).mean()
df.dropna(inplace=True)

# Trend flags
df["Trend_Long"]  = df["Close"] > df["MA"]
df["Trend_Short"] = df["Close"] < df["MA"]

# Filtra: label 1 con Trend_Long, label 0 con Trend_Short
pre_rows = len(df)
df = df[(df["Trend_Long"] & (df["Label"] == 1)) |
        (df["Trend_Short"] & (df["Label"] == 0))]
L.info(f"Rows after trend filter: {len(df):,} (da {pre_rows:,})")

# ─────────────────────────────
# Statistiche
# ─────────────────────────────
pos_pct = 100 * df["Label"].mean()
df["date"] = df.index.date
daily_counts = df.groupby("date").size()
L.info(f"Positivi dopo filtro: {pos_pct:.2f}%")
L.info(f"Trade/giorno — mean: {daily_counts.mean():.2f}, "
       f"median: {daily_counts.median():.2f}, max: {daily_counts.max()}")

# ─────────────────────────────
# Salva parquet filtrato
# ─────────────────────────────
OUT = Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.drop(columns=["MA","Trend_Long","Trend_Short","date"]).to_parquet(OUT)
L.info(f"✅ Parquet filtrato salvato: {OUT}")