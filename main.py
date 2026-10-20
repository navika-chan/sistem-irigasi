"""
Microservice Cloud REST API Sistem Rekomendasi Irigasi Presisi
Model: SimpleRNN (1.991 parameter, Seed 42 - Best Validation Checkpoint)
Engine: Pure-NumPy High-Performance Inference (Zero-TensorFlow Overhead)
Framework: FastAPI + Uvicorn
"""

import os
import json
import time
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Sistem Rekomendasi Irigasi Presisi Berbasis SimpleRNN Cloud",
    description="REST API untuk peramalan ETo multi-step dan rekomendasi volume irigasi tanaman tomat berbasis neraca air FAO-56.",
    version="1.0.0"
)

# Muat Bobot Model Terlatih (Pure-NumPy)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "best_rnn_weights.json")

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Berkas bobot model tidak ditemukan di: {WEIGHTS_PATH}")

with open(WEIGHTS_PATH, "r") as f:
    WEIGHTS = json.load(f)

# Ekstrak Matriks Bobot & Bias
W_X1 = np.array(WEIGHTS['rnn1_Wx'], dtype=np.float32) # (1, 32)
W_H1 = np.array(WEIGHTS['rnn1_Wh'], dtype=np.float32) # (32, 32)
B1   = np.array(WEIGHTS['rnn1_b'], dtype=np.float32)   # (32,)

W_X2 = np.array(WEIGHTS['rnn2_Wx'], dtype=np.float32) # (32, 16)
W_H2 = np.array(WEIGHTS['rnn2_Wh'], dtype=np.float32) # (16, 16)
B2   = np.array(WEIGHTS['rnn2_b'], dtype=np.float32)   # (16,)

W_D  = np.array(WEIGHTS['dense_W'], dtype=np.float32)  # (16, 7)
B_D  = np.array(WEIGHTS['dense_b'], dtype=np.float32)  # (7,)

S_MIN   = float(WEIGHTS['scaler_min'])
S_SCALE = float(WEIGHTS['scaler_scale'])

# Konstanta Fisik Neraca Air Tomat
RAW = 48.33         # Ambang deplesi stres air (mm)
TAW = 120.83        # Kapasitas air total zona perakaran (mm)
ML_PER_MM = 96.0    # Konversi pot 0.096 m2: 1 mm setara 96 mL/pot

def run_pure_numpy_inference(history_7d: List[float]) -> List[float]:
    """
    Menjalankan forward pass SimpleRNN 2-layer dengan aktivasi ReLU dan Dense linear
    menggunakan matriks Pure-NumPy tanpa ketergantungan TensorFlow.
    """
    x_raw = np.array(history_7d, dtype=np.float32).reshape(1, 7, 1)
    
    # 1. Penskalaan MinMax
    x_scaled = x_raw * S_SCALE + S_MIN
    
    # 2. SimpleRNN Layer 1 (32 unit, ReLU, return_sequences=True)
    h1 = np.zeros((1, 32), dtype=np.float32)
    h1_seq = np.zeros((1, 7, 32), dtype=np.float32)
    for t in range(7):
        xt = x_scaled[:, t, :]
        h1 = np.maximum(0.0, np.dot(xt, W_X1) + np.dot(h1, W_H1) + B1)
        h1_seq[:, t, :] = h1
        
    # 3. SimpleRNN Layer 2 (16 unit, ReLU, return_sequences=False)
    h2 = np.zeros((1, 16), dtype=np.float32)
    for t in range(7):
        xt2 = h1_seq[:, t, :]
        h2 = np.maximum(0.0, np.dot(xt2, W_X2) + np.dot(h2, W_H2) + B2)
        
    # 4. Dense Output Layer (7 unit, Linear)
    out_scaled = np.dot(h2, W_D) + B_D
    
    # 5. Inverse MinMax Scaling
    out_unscaled = (out_scaled - S_MIN) / S_SCALE
    return [float(round(val, 3)) for val in out_unscaled[0]]

# Schema Pydantic Permintaan & Respons
class PredictRequest(BaseModel):
    eto_history: List[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="Daftar 7 nilai ETo harian historis (t-7 s.d. t-1) dalam mm/hari",
        example=[4.5, 4.8, 5.1, 4.9, 5.2, 5.0, 4.7]
    )

class PredictResponse(BaseModel):
    status: str
    prediction_eto: List[float]
    horizon_days: int
    execution_time_ms: float

class RecommendRequest(BaseModel):
    eto_history: List[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="Daftar 7 nilai ETo harian historis (t-7 s.d. t-1) dalam mm/hari",
        example=[4.5, 4.8, 5.1, 4.9, 5.2, 5.0, 4.7]
    )
    current_depletion_mm: float = Field(
        ...,
        ge=0.0,
        le=120.83,
        description="Nilai deplesi air tanah saat ini Dr,t (mm)",
        example=25.4
    )
    lookahead_days: Optional[int] = Field(
        default=3,
        ge=1,
        le=7,
        description="Horizon ramalan yang digunakan untuk keputusan (H hari)",
        example=3
    )
    crop_kc: Optional[float] = Field(
        default=1.2,
        gt=0.0,
        description="Koefisien tanaman tomat Kc",
        example=1.2
    )

class RecommendResponse(BaseModel):
    status: str
    action: str
    irrigation_mm: float
    irrigation_mL: float
    current_depletion_mm: float
    projected_depletion_mm: float
    threshold_raw_mm: float
    lookahead_days: int
    forecasted_eto_H: List[float]
    execution_time_ms: float

# Endpoint 1: Root /
@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Sistem Rekomendasi Irigasi Presisi Berbasis SimpleRNN Cloud aktif.",
        "docs_url": "/docs",
        "health_check": "/health"
    }

# Endpoint 2: GET /health
@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "service": "precision-irrigation-simplernn-api",
        "model_architecture": "SimpleRNN [32, 16], Dense(7)",
        "trainable_parameters": 1991,
        "selected_seed": 42,
        "validation_rmse_mm_day": 0.5726,
        "inference_engine": "Pure-NumPy",
        "container_ready": True
    }

# Endpoint 3: POST /predict
@app.post("/predict", response_model=PredictResponse, tags=["Peramalan"])
def predict(payload: PredictRequest):
    t_start = time.perf_counter()
    try:
        prediction = run_pure_numpy_inference(payload.eto_history)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        return PredictResponse(
            status="success",
            prediction_eto=prediction,
            horizon_days=len(prediction),
            execution_time_ms=round(t_elapsed, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan inferensi peramalan: {str(e)}")

# Endpoint 4: POST /recommend
@app.post("/recommend", response_model=RecommendResponse, tags=["Rekomendasi"])
def recommend(payload: RecommendRequest):
    t_start = time.perf_counter()
    try:
        # 1. Jalankan peramalan ETo 7 hari ke depan
        pred_7d = run_pure_numpy_inference(payload.eto_history)
        
        # 2. Ambil subset horizon H
        H = payload.lookahead_days
        pred_H = pred_7d[:H]
        
        # 3. Proyeksikan deplesi Dr,proj = Dr,t + Kc * sum(ETo_hat)
        D_r_proj = payload.current_depletion_mm + payload.crop_kc * sum(pred_H)
        
        # 4. Logika Keputusan FAO-56
        if D_r_proj > RAW:
            action = "IRRIGATE"
            irrigation_mm = round(payload.current_depletion_mm, 2)
            irrigation_mL = round(payload.current_depletion_mm * ML_PER_MM, 1)
        else:
            action = "IDLE"
            irrigation_mm = 0.0
            irrigation_mL = 0.0
            
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        
        return RecommendResponse(
            status="success",
            action=action,
            irrigation_mm=irrigation_mm,
            irrigation_mL=irrigation_mL,
            current_depletion_mm=round(payload.current_depletion_mm, 2),
            projected_depletion_mm=round(D_r_proj, 2),
            threshold_raw_mm=RAW,
            lookahead_days=H,
            forecasted_eto_H=pred_H,
            execution_time_ms=round(t_elapsed, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses rekomendasi: {str(e)}")
