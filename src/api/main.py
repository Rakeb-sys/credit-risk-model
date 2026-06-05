"""
main.py
-------
FastAPI application for real-time credit risk scoring.
Endpoints:
  GET  /health          - Health check
  POST /predict         - Single applicant prediction
  POST /predict/batch   - Batch prediction (up to 500)
  GET  /model/info      - Model metadata
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.pydantic_models import (
    ApplicantFeatures,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
)
from src.predict import CreditRiskPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/credit_risk_xgb.pkl")

# ─── App State ────────────────────────────────────────────────────────────────

predictor: CreditRiskPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global predictor
    logger.info("🚀 Starting Credit Risk API...")
    try:
        predictor = CreditRiskPredictor(model_path=MODEL_PATH)
        logger.info(f"✅ Model loaded: {predictor.model_name}")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not found: {e}. /predict endpoints will be unavailable.")
    yield
    logger.info("🛑 Shutting down Credit Risk API.")


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Production-ready API for real-time credit risk assessment. "
        "Predicts Probability of Default (PD) and assigns a risk tier.\n\n"
        "Built on the German Credit Dataset with XGBoost / Random Forest / Logistic Regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Middleware: Request Timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - start) * 1000, 2))
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness and readiness health check."""
    return HealthResponse(
        status="ok" if predictor else "degraded",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor else None,
    )


@app.get("/model/info", tags=["Monitoring"])
async def model_info() -> Dict:
    """Return metadata about the currently loaded model."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return {
        "model_name": predictor.model_name,
        "features": predictor.feature_cols,
        "n_features": len(predictor.feature_cols),
        "model_path": MODEL_PATH,
        "api_version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(applicant: ApplicantFeatures):
    """
    Predict credit risk for a single applicant.

    Returns:
    - **default_probability**: Likelihood of default (0.0 – 1.0)
    - **risk_tier**: LOW / MEDIUM-LOW / MEDIUM-HIGH / HIGH
    - **prediction**: 0 (Good Credit) or 1 (Default)
    - **recommendation**: APPROVE or DECLINE
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python src/train.py` first.")

    try:
        results = predictor.predict(applicant.model_dump())
        result = results[0]
        return PredictionResponse(
            default_probability=result["default_probability"],
            risk_tier=result["risk_tier"],
            prediction=result["prediction"],
            recommendation=result["recommendation"],
            model_version=predictor.model_name,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit risk for a batch of applicants (up to 500).
    Returns aggregate statistics alongside individual predictions.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        import pandas as pd
        data = pd.DataFrame([a.model_dump() for a in request.applicants])
        raw_results = predictor.predict(data)

        predictions = [
            PredictionResponse(
                default_probability=r["default_probability"],
                risk_tier=r["risk_tier"],
                prediction=r["prediction"],
                recommendation=r["recommendation"],
                model_version=predictor.model_name,
            )
            for r in raw_results
        ]

        high_risk = sum(1 for r in raw_results if r["risk_tier"] in ("MEDIUM-HIGH", "HIGH"))
        approvals = sum(1 for r in raw_results if r["recommendation"] == "APPROVE")

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            high_risk_count=high_risk,
            approval_rate=round(approvals / len(predictions), 4),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
