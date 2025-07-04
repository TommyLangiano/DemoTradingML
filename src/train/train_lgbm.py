#!/usr/bin/env python3
"""
Train LightGBM binario con monitor AUC e early stopping. Salva .pkl.
"""

import argparse, logging, os, sys
from pathlib import Path
import pandas as pd, lightgbm as lgb, joblib, json

# Logging
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(sys.stdout)
    ],
)
L = logging.getLogger("train")

# CLI
p = argparse.ArgumentParser(description="Train LightGBM")
p.add_argument("--splits", default="data/splits")
p.add_argument("--out",    default="models/checkpoints/lgbm_model.pkl")
p.add_argument("--metric", default="auc")
args = p.parse_args()

# Percorsi
SPL  = Path(args.splits)
TRAIN = SPL/"train.parquet"; VALID = SPL/"valid.parquet"
if not TRAIN.exists() or not VALID.exists():
    L.error("⚠️  train/valid parquet mancano, run split.py prima.")
    sys.exit(1)

# Carica
FEAT = ["HL_range","OC_change","Body_pct","Volume_log"]
train = pd.read_parquet(TRAIN)
valid = pd.read_parquet(VALID)
dtrain = lgb.Dataset(train[FEAT], label=train["Label"])
dvalid = lgb.Dataset(valid[FEAT], label=valid["Label"])

params = dict(objective="binary", metric=args.metric,
              learning_rate=0.02, num_leaves=31, verbose=-1)

model = lgb.train(params, dtrain,
                  num_boost_round=1000,
                  valid_sets=[dtrain, dvalid],
                  valid_names=["train","valid"],
                  early_stopping_rounds=50,
                  callbacks=[lgb.log_evaluation(period=50)])

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, args.out)
L.info(f"✅ Modello salvato: {args.out}")

# Salva metriche JSON
metrics = dict(best_iter=model.best_iteration,
               best_score=model.best_score["valid"][args.metric])
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
with open("models/checkpoints/metrics.json","w") as f:
    json.dump(metrics,f,indent=2)
L.info(f"AUC valido: {metrics['best_score']:.4f}")