import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path

FEATURES = Path(__file__).resolve().parents[2] / "data/feature_store/EURUSD_M1_features.parquet"
MODEL_OUT = Path(__file__).resolve().parents[2] / "models/checkpoints/lgbm_model.pkl"

def main():
    df = pd.read_parquet(FEATURES).dropna(subset=["Label"])   # assicurati che ci sia la colonna Label
    X = df[["HL_range", "OC_change", "Body_pct", "Volume_log"]]
    y = df["Label"]

    dtrain = lgb.Dataset(X, label=y)
    params = dict(objective="binary", metric="auc", learning_rate=0.02,
                  num_leaves=31, verbose=-1)
    model = lgb.train(params, dtrain, num_boost_round=400)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"✅ Modello LGBM salvato in {MODEL_OUT}")

if __name__ == "__main__":
    main()