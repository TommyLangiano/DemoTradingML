#!/usr/bin/env python3
"""
Crea la colonna Label (1 = TP prima di SL, 0 = SL prima di TP).
Output: parquet con colonna 'Label'.
"""

import argparse, logging, sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ─────────────────────────────
# 1️⃣  Logging
# ─────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "label.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("label")

# ─────────────────────────────
# 2️⃣  Parametri CLI
# ─────────────────────────────
p = argparse.ArgumentParser(description="Label TP/SL")
p.add_argument("--input",  default="data/processed/EURUSD_M1_clean.parquet")
p.add_argument("--output", default="data/processed/EURUSD_M1_labeled.parquet")
p.add_argument("--tp", type=float, default=0.001,  help="TP pct (+)")
p.add_argument("--sl", type=float, default=0.001,  help="SL pct (-)")
p.add_argument("--horizon", type=int, default=20,  help="Barre future da osservare")
args = p.parse_args()

# ─────────────────────────────
# 3️⃣  Carica dati
# ─────────────────────────────
SRC = Path(args.input)
if not SRC.exists():
    L.error(f"Clean parquet non trovato: {SRC}")
    sys.exit(1)
df = pd.read_parquet(SRC)
L.info(f"Rows loaded: {len(df):,}")

# ─────────────────────────────
# 4️⃣  Calcolo label
# ─────────────────────────────
tp_pct, sl_pct, H = args.tp, args.sl, args.horizon
highs, lows, closes = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()

def label_row(i: int) -> int:
    tp_level = closes[i] * (1 + tp_pct)
    sl_level = closes[i] * (1 - sl_pct)
    for h, l in zip(highs[i+1:i+1+H], lows[i+1:i+1+H]):
        if h >= tp_level:
            return 1
        if l <= sl_level:
            return 0
    return 0  # fallback: nessuno colpito → SL

L.info(f"Start labeling... Horizon: {H}, TP: {tp_pct}, SL: {sl_pct}")

labels = [label_row(i) for i in range(len(df) - H)] + [None] * H
df["Label"] = labels

L.info(f"Done labeling. Dropping last {H} rows (tail horizon)...")
df.dropna(inplace=True)
df["Label"] = df["Label"].astype(int)
L.info(f"Rows after dropping tail: {len(df):,}")

# ─────────────────────────────
# 5️⃣  Validazione schema Pandera
# ─────────────────────────────
schema = DataFrameSchema({
    "Label": Column(int, Check(lambda s: s.isin([0, 1])))
})
try:
    schema.validate(df[["Label"]].sample(min(len(df), 500)))
    pct_pos = 100 * df["Label"].mean()
    L.info(f"Schema ✅ OK – Positives: {pct_pos:.2f}%")
except pa.errors.SchemaError as e:
    L.error(f"Schema validation failed:\n{e}")
    sys.exit(1)

# ─────────────────────────────
# 6️⃣  Salva parquet
# ─────────────────────────────
OUT = Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)
L.info(f"✅ Label parquet salvato: {OUT}")