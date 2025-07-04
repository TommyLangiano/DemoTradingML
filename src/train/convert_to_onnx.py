#!/usr/bin/env python3
import joblib, onnxmltools, os, pathlib, logging, sys
from onnxmltools.convert.common.data_types import FloatTensorType
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL = "models/checkpoints/lgbm_model.pkl"
ONNX  = "models/onnx/lgbm_model.onnx"

if not os.path.exists(MODEL):
    logging.error("⚠️  .pkl non trovato, run train_lgbm.py prima.")
    sys.exit(1)

model = joblib.load(MODEL)
initial = [("float_input", FloatTensorType([None,4]))]
onnx = onnxmltools.convert_lightgbm(model, initial_types=initial)
pathlib.Path("models/onnx").mkdir(parents=True, exist_ok=True)
onnxmltools.utils.save_model(onnx, ONNX)
logging.info(f"✅ ONNX salvato: {ONNX}")