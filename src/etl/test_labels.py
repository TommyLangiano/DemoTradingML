#!/usr/bin/env python3
"""
Testa diverse configurazioni TP/SL e Horizon.
Stampa % positivi per ogni combinazione.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Parametri
INPUT = "data/processed/EURUSD_M1_clean.parquet"
tp_list = [0.001, 0.002, 0.003]  # TP: 0.1%, 0.2%, 0.3%
sl_list = [0.001, 0.002]         # SL: 0.1%, 0.2%
horizon_list = [20, 50]          # Barre future

# Carica dati
df = pd.read_parquet(INPUT)
highs, lows, closes = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()

def label_row(i, tp_pct, sl_pct, H):
    tp_level = closes[i] * (1 + tp_pct)
    sl_level = closes[i] * (1 - sl_pct)
    for h, l in zip(highs[i+1:i+1+H], lows[i+1:i+1+H]):
        if h >= tp_level:
            return 1
        if l <= sl_level:
            return 0
    return 0

print("🔎 Start TP/SL test...\n")

# Loop su combinazioni
for tp in tp_list:
    for sl in sl_list:
        for H in horizon_list:
            labels = [label_row(i, tp, sl, H) for i in range(len(df) - H)]
            pct_pos = 100 * np.mean(labels)
            print(f"TP: {tp:.3f} | SL: {sl:.3f} | Horizon: {H:3d} → Positivi: {pct_pos:.2f}%")