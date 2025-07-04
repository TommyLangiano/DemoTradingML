#!/usr/bin/env python3
"""
Calcola le feature base (HL_range, OC_change, Body_pct, Volume_log),
le normalizza (z-score) e salva il Parquet in feature_store.
"""

import argparse, logging, sys, os
from pathlib import Path
import pandas as pd, numpy as np
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
        logging.FileHandler(LOG_DIR / "feature_engineering.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("feature")

# ─────────────────────────────
# 2️⃣  CLI
# ─────────────────────────────
p = argparse.ArgumentParser(description="Feature Engineering M1")
p.add_argument("--input",  default="data/processed/EURUSD_M1_labeled.parquet",
               help="Parquet con colonna Label")
p.add_argument("--output", default="data/feature_store/EURUSD_M1_features.parquet",
               help="Parquet con feature normalizzate")
args = p.parse_args()

in_path  = Path(args.input)
out_path = Path(args.output)

if not in_path.exists():
    L.error(f"⚠️  Labeled parquet non trovato: {in_path}")
    sys.exit(1)

# ─────────────────────────────
# 3️⃣  Carica + calcola feature
# ─────────────────────────────
df = pd.read_parquet(in_path)
L.info(f"Rows loaded: {len(df):,}")

df["HL_range"]  = df["High"] - df["Low"]
df["OC_change"] = df["Close"] - df["Open"]
df["Body_pct"]  = np.abs(df["Close"]-df["Open"]) / (df["HL_range"] + 1e-6)
df["Volume_log"]= np.log1p(df["Volume"])

features = ["HL_range","OC_change","Body_pct","Volume_log"]

# ─────────────────────────────
# 4️⃣  Normalizzazione z-score
# ─────────────────────────────
for col in features:
    mean, std = df[col].mean(), df[col].std()
    df[col] = (df[col] - mean) / std
L.info("Feature normalizzate (z-score)")

# ─────────────────────────────
# 5️⃣  Validazione schema
# ─────────────────────────────
schema = DataFrameSchema({
    "Label": Column(int, Check(lambda s: s.isin([0,1]))),
    **{c: Column(float, Check(lambda s: s.notna())) for c in features}
})

try:
    schema.validate(df[["Label"]+features].sample(100))
    L.info("Schema Pandera → OK")
except pa.errors.SchemaError as e:
    L.error(f"Schema validation failed:\n{e}")
    sys.exit(1)

# ─────────────────────────────
# 6️⃣  Salva parquet
# ─────────────────────────────
out_path.parent.mkdir(parents=True, exist_ok=True)
df[features + ["Label"]].to_parquet(out_path)
L.info(f"✅ Features parquet salvato: {out_path}")