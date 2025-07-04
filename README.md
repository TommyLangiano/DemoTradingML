# 🚀 ML-EA-Project — MetaTrader 5 + Machine Learning Trading Pipeline

**Versione 1.0 — Luglio 2025**

Questo progetto unisce la potenza di **Machine Learning** e la velocità di esecuzione di **MetaTrader 5**, creando un *Expert Advisor* che utilizza modelli predittivi basati su dati storici per prendere decisioni di trading automatico.

---

## 💡 Obiettivo

- Prevedere se il prezzo raggiungerà il **Take-Profit** prima dello **Stop-Loss** in un orizzonte definito.
- Massimizzare la probabilità di trade profittevoli con risk-management avanzato.
- Integrazione seamless con MT5 tramite un server FastAPI e modello ONNX.

---

## ⚙️ Architettura del progetto

ML-EA-Project/
├─ data/
│   ├─ raw/               → CSV originali (M1 o tick)
│   ├─ processed/         → file puliti e labelizzati (.parquet)
│   ├─ feature_store/     → feature normalizzate
│   └─ splits/           → train / valid / test
│
├─ models/
│   ├─ checkpoints/       → LightGBM .pkl + metriche
│   └─ onnx/              → Modello convertito per FastAPI
│
├─ src/
│   ├─ etl/               → script ETL: clean.py, label.py, feature_engineering.py, split.py
│   ├─ train/            → training & conversione modello
│   └─ deploy/          → fastapi_gateway.py
│
├─ MQL5/                 → EA_PRO.mq5 + moduli risk & utils
├─ logs/                → log di esecuzione
├─ README.md
└─ requirements.txt

---

## 📥 Dati

### Dukascopy-node
```bash
npx dukascopy-node -i eurusd -from 2022-01-01 -to 2025-06-30 -t m1 -f csv -dir data/raw
mv data/raw/eurusd_m1_*.csv data/raw/EURUSD_M1.csv


⸻

⚙️ Setup ambiente

python3 -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate
pip install -r requirements.txt


⸻

🛠️ Pipeline end-to-end

python src/etl/clean.py
python src/etl/label.py          # parametri: --tp --sl --horizon
python src/etl/feature_engineering.py
python src/etl/split.py
python src/train/train_lgbm.py
python src/train/convert_to_onnx.py

Output finale:
	•	models/checkpoints/lgbm_model.pkl
	•	models/checkpoints/metrics.json
	•	models/onnx/lgbm_model.onnx

⸻

🌐 FastAPI Gateway

python src/deploy/fastapi_gateway.py

	•	URL: http://127.0.0.1:8000/predict
	•	Esempio richiesta JSON:

{ "features": [-0.15, 0.10, 0.50, 1.05] }

	•	Risposta:

{ "prediction": 0.73 }


⸻

🤖 Expert Advisor (EA_PRO.mq5)

Strategia di base
	•	Trend Filter: usa MA 200 su M15 per determinare bias.
	•	Session Filter: attivo solo in sessioni Londra / New York.
	•	Regole apertura:
	•	prediction > 0.6 & bias long → BUY
	•	prediction < 0.4 & bias short → SELL
	•	Risk Manager:
	•	TP/SL fissi
	•	Trailing stop (50% del TP)
	•	Equity guard (-3% daily stop-loss)

Setup
	1.	Abilita “Trading algoritmico”.
	2.	Aggiungi http://127.0.0.1 alle Opzioni → Expert Advisors → WebRequest.
	3.	Ricompila EA_PRO.mq5 in MetaEditor.

⸻

🗂️ Monitoraggio

Componente	Log
ETL & training	logs/*.log
Prediction FastAPI	logs/pred_log.csv
MT5 Diario	dettagli ordini & equity-guard


⸻

🚩 Prossimi upgrade
	•	Tick aggregation: dollar bars o volume bars.
	•	Feature engineering avanzato (ATR, macro-news flag, MA-diff).
	•	Hyperparameter tuning con Optuna.
	•	Docker deployment per FastAPI.
	•	CI/CD pipeline per validazione modelli.

