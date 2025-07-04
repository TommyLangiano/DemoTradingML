import pandas as pd
from pathlib import Path
import lightgbm as lgb
import joblib

# Percorsi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "EURUSD_M1_labeled.parquet"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "lgbm_model.pkl"

# Carica dati
df = pd.read_parquet(INPUT_FILE)
print("✅ Dati caricati")

# Rimuovi righe finali (Label == None)
df = df.dropna(subset=["Label"])
print("✅ Righe senza label rimosse")

# Seleziona feature base
features = ["Open", "High", "Low", "Close", "Volume"]
target = "Label"

X = df[features]
y = df[target]

# Dataset LightGBM
dtrain = lgb.Dataset(X, label=y)

# Parametri base
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "verbose": -1,
    "force_col_wise": True
}

# Allenamento
model = lgb.train(params, dtrain, num_boost_round=100)
print("✅ Modello allenato")

# Salva pkl
joblib.dump(model, MODEL_FILE)
print(f"✅ Modello salvato: {MODEL_FILE}")