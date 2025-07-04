#!/usr/bin/env python3
"""
Pulizia e deduplicazione dati M1 o tick aggregato.
- Logging a file + console
- Validazione schema Pandera (Volume opzionale)
- Gestisce timestamp in ms (Dukascopy) o DATE+TIME (MT5)
"""

import argparse, logging, sys
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ────────────────────────────────
# 1️⃣  Logging
# ────────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "clean.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("clean")

# ────────────────────────────────
# 2️⃣  Schema Pandera (Volume non obbligatorio)
# ────────────────────────────────
schema = DataFrameSchema({
    "Open":   Column(float, Check(lambda s: s.notna())),
    "High":   Column(float, Check(lambda s: s.notna())),
    "Low":    Column(float, Check(lambda s: s.notna())),
    "Close":  Column(float, Check(lambda s: s.notna())),
    "Volume": Column(int,   Check.ge(0), nullable=True)
})

# ────────────────────────────────
# 3️⃣  Funzione load
# ────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    """Carica CSV da Dukascopy-node (timestamp) o MT5 (DATE+TIME)."""
    sep_guess = "\t" if "\t" in open(path, "r").readline() else ","
    df = pd.read_csv(path, sep=sep_guess)

    if "DATE" in df.columns:     # Export MT5
        df.columns = [c.replace("<", "").replace(">", "").strip() for c in df.columns]
        df["time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
        df = df.rename(columns={
            "OPEN": "Open", "HIGH": "High", "LOW": "Low",
            "CLOSE": "Close", "TICKVOL": "Volume"
        })
    elif "timestamp" in df.columns:  # Dukascopy timestamp ms
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close"
        })
        # Se Volume non esiste, crealo placeholder
        if "volume" in df.columns:
            df = df.rename(columns={"volume": "Volume"})
        else:
            df["Volume"] = 0
    else:
        L.error("⚠️  CSV non contiene colonne DATE+TIME o timestamp.")
        sys.exit(1)
    return df

# ────────────────────────────────
# 4️⃣  Funzione clean
# ────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = (df
        .set_index("time")
        [["Open", "High", "Low", "Close", "Volume"]]
        .sort_index())
    df = df.loc[~df.index.duplicated()]
    return df

# ────────────────────────────────
# 5️⃣  Main
# ────────────────────────────────
def main(args):
    L.info(f"Start clean for {args.symbol}")

    raw_csv = Path(args.input)
    if not raw_csv.exists():
        L.error(f"CSV non trovato: {raw_csv}")
        sys.exit(1)

    df = load_csv(raw_csv)
    L.info(f"Rows loaded: {len(df):,}")

    df = clean(df)
    L.info(f"Rows after dedup: {len(df):,}")

    # Validate schema
    try:
        schema.validate(df.sample(min(len(df), 500)))  # valida max 500 righe
        L.info("Schema Pandera → ✅ OK")
    except pa.errors.SchemaError as e:
        L.error(f"Schema validation failed:\n{e}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    L.info(f"✅ Clean parquet salvato: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean M1 CSV -> Parquet")
    parser.add_argument("--symbol", default="EURUSD", help="Simbolo (solo log)")
    parser.add_argument("--input",  default="data/raw/EURUSD_M1.csv", help="Path CSV raw")
    parser.add_argument("--output", default="data/processed/EURUSD_M1_clean.parquet",
                        help="Parquet pulito")
    args = parser.parse_args()
    main(args)