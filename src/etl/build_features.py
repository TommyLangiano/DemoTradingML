import pandas as pd
import numpy as np
from pathlib import Path

DATA_IN  = Path(__file__).resolve().parents[2] / "data/processed/EURUSD_M1_clean.parquet"
DATA_OUT = Path(__file__).resolve().parents[2] / "data/feature_store/EURUSD_M1_features.parquet"

def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["HL_range"]   = df["High"] - df["Low"]
    df["OC_change"]  = df["Close"] - df["Open"]
    df["Body_pct"]   = df["OC_change"] / (df["HL_range"] + 1e-6)
    df["Volume_log"] = np.log1p(df["Volume"])
    return df

if __name__ == "__main__":
    df = pd.read_parquet(DATA_IN)
    df = build_basic_features(df)
    df.to_parquet(DATA_OUT)
    print(f"✅ Features salvate in {DATA_OUT}")