"""
A4: FastAPI Prediction App.

Endpoints:
    GET  /health  → Systemstatus
    POST /predict → Benzinpreis-Vorhersage
    GET  /        → HTML-Frontend

Start (JupyterHub – Port aus config.py):
    uvicorn app.main:app --reload --host 0.0.0.0 --port $(python -c "from config import API_PORT; print(API_PORT)")
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import GROUP_ID, STATION_ID, API_PORT

logger = logging.getLogger(__name__)


# ── Lifespan: Modell beim Start laden ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modell beim Start der Anwendung laden."""
    from src.predict import get_predictor
    predictor = get_predictor()
    try:
        predictor.load_model()
        logger.info("Modell erfolgreich geladen")
    except Exception as e:
        logger.error(f"Modell konnte nicht geladen werden: {e}")
    app.state.predictor = predictor
    yield


# ── FastAPI App ────────────────────────────────────────────────────────────────

# Hinter dem JupyterHub-Proxy ist die App unter /user/<gruppe>/proxy/8080/ erreichbar.
# root_path teilt FastAPI diesen Basispfad mit, damit Swagger UI (/docs) korrekt funktioniert.
_jhub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").rstrip("/")
_root_path = f"{_jhub_prefix}/proxy/{API_PORT}" if _jhub_prefix else ""

app = FastAPI(
    title=f"Benzinpreis-Vorhersage – {GROUP_ID}",
    description=(
        "A4: MLOps-Projekt – Vorhersage des nächststündigen E5-Benzinpreises.\n\n"
        f"Gruppe: {GROUP_ID} | Kurs: BIS2242 SS 2026"
    ),
    version="1.0.0",
    lifespan=lifespan,
    root_path=_root_path,
)


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    predict_for: str = datetime.now().strftime("%Y-%m-%dT%H:00:00")
    """ISO-8601 Zeitstempel, z.B. '2026-06-01T14:00:00'"""
    station_id: str = STATION_ID


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str
    prediction_for: str
    station_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Systemstatus und Modell-Verfügbarkeit."""
    predictor = getattr(app.state, "predictor", None)
    return {
        "status": "ok",
        "group_id": GROUP_ID,
        "model_loaded": predictor.is_loaded() if predictor else False,
        "timestamp": datetime.now().isoformat(),
    }


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", response_class=FileResponse)
def index():
    """HTML-Frontend ausliefern."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Benzinpreis-Vorhersage für einen Zeitstempel berechnen."""
    predictor = getattr(app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Modell nicht verfügbar")

    try:
        predict_for = datetime.fromisoformat(request.predict_for)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Ungültiger Zeitstempel: '{request.predict_for}'. Format: YYYY-MM-DDTHH:MM:SS"
        )

    try:
        predicted_price, model_version, confidence = predictor.predict(
            predict_for=predict_for,
            station_id=request.station_id,
        )
    except Exception as e:
        logger.error(f"Vorhersage fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        predicted_price=round(predicted_price, 4),
        model_version=model_version,
        prediction_for=request.predict_for,
        station_id=request.station_id,
    )
