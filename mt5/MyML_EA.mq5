//+------------------------------------------------------------------+
//|                                             MyML_EA.mq5          |
//+------------------------------------------------------------------+
#property copyright "Tommy Langiano"
#property link      "https://tuo-sito.com"
#property version   "1.00"
#property strict

#include <ONNX/onnxruntime.mqh>

#resource "\\Files\\lgbm_model.onnx" as uchar ModelONNX[]
COnnx OnnxRuntime;

input double LotSize      = 0.1;
input double StopLossPips = 50;
input double TakeProfitPips = 100;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!OnnxRuntime.LoadFromBuffer(ModelONNX, sizeof(ModelONNX)))
     {
      Print("Errore caricamento modello ONNX: ", OnnxRuntime.GetErrorMessage());
      return INIT_FAILED;
     }
   Print("✅ Modello ONNX caricato correttamente");
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double features[5];

   // Esempio: usiamo valori live della barra corrente
   features[0] = iOpen(Symbol(), PERIOD_M1, 0);
   features[1] = iHigh(Symbol(), PERIOD_M1, 0);
   features[2] = iLow(Symbol(), PERIOD_M1, 0);
   features[3] = iClose(Symbol(), PERIOD_M1, 0);
   features[4] = iVolume(Symbol(), PERIOD_M1, 0);

   double result[];
   if(!OnnxRuntime.Predict(features, result))
     {
      Print("Errore durante inferenza: ", OnnxRuntime.GetErrorMessage());
      return;
     }

   double prediction = result[0];
   // Debug
   Print("Prediction: ", prediction);

   // Esempio logica semplice
   if(prediction > 0.5 && PositionsTotal() == 0)
     {
      trade_open(true);
     }
   else if(prediction <= 0.5 && PositionsTotal() == 0)
     {
      trade_open(false);
     }
  }
//+------------------------------------------------------------------+
//| Funzione per aprire trade                                        |
//+------------------------------------------------------------------+
void trade_open(bool isBuy)
  {
   double price = isBuy ? SymbolInfoDouble(Symbol(), SYMBOL_ASK) : SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double sl = isBuy ? price - StopLossPips * _Point : price + StopLossPips * _Point;
   double tp = isBuy ? price + TakeProfitPips * _Point : price - TakeProfitPips * _Point;
   int    type = isBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   trade.PositionOpen(Symbol(), type, LotSize, price, sl, tp, "ML-EA");
  }
//+------------------------------------------------------------------+