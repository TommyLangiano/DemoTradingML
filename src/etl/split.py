#!/usr/bin/env python3
"""
Divide il parquet delle feature in tre blocchi temporali:
70 % train • 15 % valid • 15 % test.
Salva in data/splits/.
"""

import argparse, logging, os, sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# ────────────────────────────────
# Logging
# ────────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "split.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("split")

# ────────────────────────────────
# CLI
# ────────────────────────────────
p = argparse.ArgumentParser(description="Temporal train/valid/test split")
p.add_argument("--input",  default="data/feature_store/EURUSD_M1_features.parquet")
p.add_argument("--outdir", default="data/splits")
p.add_argument("--train",  type=float, default=0.70, help="Quota train")
p.add_argument("--valid",  type=float, default=0.15, help="Quota valid")
args = p.parse_args()

IN  = Path(args.input)
OUT = Path(args.outdir)

if not IN.exists():
    L.error(f"Feature parquet non trovato: {IN}")
    sys.exit(1)

df = pd.read_parquet(IN).sort_index()
n  = len(df)
a  = int(n * args.train)
b  = int(n * (args.train + args.valid))
L.info(f"Rows total: {n:,}  → train:{a:,}  valid:{b-a:,}  test:{n-b:,}")

OUT.mkdir(parents=True, exist_ok=True)
df.iloc[:a].to_parquet(OUT/"train.parquet")
df.iloc[a:b].to_parquet(OUT/"valid.parquet")
df.iloc[b:].to_parquet(OUT/"test.parquet")
L.info(f"✅ Split salvati in {OUT.resolve()}")