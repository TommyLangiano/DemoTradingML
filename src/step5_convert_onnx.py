import joblib
import numpy as np
import onnx
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from pathlib import Path

# Percorsi
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODEL_DIR / "lgbm_model.pkl"
ONNX_FILE = MODEL_DIR / "lgbm_model.onnx"

# Carica modello
model = joblib.load(MODEL_FILE)
print("✅ Modello caricato")

# Definisci input shape (numero colonne = 5)
initial_type = [("float_input", FloatTensorType([None, 5]))]

# Conversione
onnx_model = convert_lightgbm(model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, str(ONNX_FILE))

print(f"✅ Modello convertito e salvato in ONNX: {ONNX_FILE}")