"""
A4 (Teil 1): Inferenz-Modul – Modell laden und Vorhersagen durchführen.

Lädt das trainierte Modell aus der MLflow Model Registry und
führt Vorhersagen durch.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import GROUP_ID, MLFLOW_TRACKING_URI, DB_CONFIG, DEFAULT_PARAMS, GASPRICES_SCHEMA, STATION_ID, FUEL_TYPE
from db.connection import get_connection

logger = logging.getLogger(__name__)


def load_params(params_path: str = None) -> dict:
    """Lädt Parameter aus params.yaml, Fallback auf config.py."""
    if params_path is None:
        params_path = os.path.join(PROJECT_ROOT, "params.yaml")
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        if params and isinstance(params, dict):
            return params
    except Exception:
        pass
    return DEFAULT_PARAMS


def _fetch_hourly_prices(station_id: str, end_dt: datetime, hours: int = 175) -> pd.Series:
    """
    [VORGEGEBEN] Lädt stündliche Kraftstoffpreise aus gasprices-Schema.
    Kraftstoffsorte wird über FUEL_TYPE in config.py gesteuert.
    """
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)

    fuel_col = FUEL_TYPE
    price_col = f"{fuel_col}_price"

    conn = get_connection()
    try:
        query = f"""
            SELECT date, {fuel_col}
            FROM {GASPRICES_SCHEMA}.gas_station_information_history
            WHERE stid = %(station_id)s
              AND {fuel_col} IS NOT NULL
              AND {fuel_col} > 0
              AND date >= %(start)s
              AND date <  %(end)s
            ORDER BY date ASC
        """
        df = pd.read_sql(query, conn, params={"station_id": station_id, "start": start_dt, "end": end_dt})
    finally:
        conn.close()

    if df.empty:
        raise ValueError(
            f"Keine Preisdaten für Station {station_id} im Zeitraum "
            f"{start_dt} – {end_dt} gefunden."
        )

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df[price_col] = df[fuel_col] / 1000.0
    df = df.set_index("date").sort_index()
    hourly = df[price_col].resample("h").median().ffill()
    return hourly


def _get_price_at(hourly: pd.Series, dt: datetime) -> Optional[float]:
    """
    [VORGEGEBEN] Gibt den Preis zur vollen Stunde zurück.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(dt).floor("h")
    if ts in hourly.index:
        return float(hourly[ts])
    subset = hourly.loc[:ts]
    return float(subset.iloc[-1]) if not subset.empty else None


def _features_from_csv(predict_for: datetime) -> Optional[pd.DataFrame]:
    """
    Fallback: Feature-Vektor aus features.csv lesen.
    Wird verwendet wenn gasprices DB nicht erreichbar ist.
    """
    params = load_params()
    features_path = os.path.join(PROJECT_ROOT, params["paths"]["processed"])
    if not os.path.exists(features_path):
        return None

    df = pd.read_csv(features_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if predict_for.tzinfo is None:
        predict_for = predict_for.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(predict_for).floor("h")

    # Genaue Übereinstimmung suchen
    match = df[df["datetime"] == ts]
    if not match.empty:
        return match.iloc[[0]]

    # Nächstgelegene Zeile vor dem gesuchten Zeitpunkt
    before = df[df["datetime"] <= ts]
    if not before.empty:
        return before.iloc[[-1]]

    # Nächstgelegene Zeile nach dem Zeitpunkt
    after = df[df["datetime"] > ts]
    if not after.empty:
        return after.iloc[[0]]

    return None


# Spalten, die nicht als Features dienen (FUEL_TYPE-dynamisch)
_EXCLUDE_COLS = {"datetime", f"{FUEL_TYPE}_price", f"{FUEL_TYPE}_price_next_hour"}


class FuelPricePredictor:
    """
    Singleton-Klasse für Benzinpreis-Inferenz.

    Lädt das Production-Modell aus der MLflow Model Registry und
    stellt Vorhersagen bereit.
    """

    _instance: Optional["FuelPricePredictor"] = None
    _model = None
    _model_version: Optional[str] = None
    _feature_cols: Optional[list] = None
    _params: Optional[dict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._params is None:
            self._params = load_params()

    def load_model(self) -> None:
        """Production-Modell aus MLflow Model Registry laden."""
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(tracking_uri)

        model_name = self._params["mlflow"]["model_name"]
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            raise ValueError(
                f"Kein Modell '{model_name}' in der MLflow Registry.\n"
                "→ Zuerst A3_Training.ipynb ausführen"
            )

        # TODO (1/2): Neueste Modell-Version bestimmen und laden
        # Schritt 1: Neueste Version aus 'versions' ermitteln (höchste Versionsnummer)
        # Schritt 2: model_uri zusammensetzen: "models:/<name>/<version>"
        # Schritt 3: Modell laden und in self._model speichern
        raise NotImplementedError("TODO: load_model – Modell aus MLflow Registry laden")

        # TODO (2/2): Feature-Reihenfolge aus features.csv ermitteln
        # Lesen Sie die Spalten aus features.csv (nrows=0 genügt)
        # Filtern Sie alle Nicht-Feature-Spalten heraus (_EXCLUDE_COLS + target_column)
        # Speichern Sie die Feature-Spalten in self._feature_cols
        raise NotImplementedError("TODO: load_model – Feature-Spalten ermitteln")

        logger.info(f"Modell geladen: {model_name} v{self._model_version}")

    def is_loaded(self) -> bool:
        """Gibt True zurück, wenn das Modell geladen ist."""
        return self._model is not None

    def predict(self, predict_for: datetime, station_id: str = None) -> Tuple[float, str, float]:
        """
        Feature-Vektor aufbauen und Vorhersage durchführen.

        Returns:
            (predicted_price, model_version, confidence)
        """
        if station_id is None:
            station_id = STATION_ID

        if predict_for.tzinfo is None:
            predict_for = predict_for.replace(tzinfo=timezone.utc)

        try:
            # [VORGEGEBEN] Stündliche Preise aus der Datenbank laden
            # hourly: pd.Series – Index=UTC-Timestamps (volle Stunden), Werte=Preis in EUR/L
            # Preis zu einem beliebigen Zeitpunkt: _get_price_at(hourly, datetime_obj)
            hourly = _fetch_hourly_prices(station_id, predict_for, hours=172)

            # TODO: Feature-Vektor aufbauen – dieselben Features wie in A2 create_features()!
            #
            # Erstellen Sie ein dict `features_dict` mit genau den Spalten, die Ihr Modell
            # in A2 trainiert wurde. Die Spaltennamen müssen exakt mit Ihrer features.csv
            # übereinstimmen.
            #
            # Zeitliche Features aus `predict_for` (datetime):
            #   hour, day_of_week, month, is_weekend
            #
            # Historische Preise aus `hourly` (via _get_price_at):
            #   price_lag_1  – Preis 1 Stunde vor predict_for
            #   price_lag_24 – Preis 24 Stunden vor predict_for
            #   price_ma_24  – Mittelwert der letzten 24 Stunden
            #   price_diff_1 – Differenz: Preis(jetzt) – Preis(vor 1h)
            #   ... (weitere Features, die Sie in A2 implementiert haben)
            #
            raise NotImplementedError("TODO: Feature-Vektor aufbauen – wie in A2 create_features()")

        except NotImplementedError:
            raise
        except Exception as db_err:
            logger.warning(f"DB-Zugriff fehlgeschlagen, verwende CSV-Fallback: {db_err}")
            row_df = _features_from_csv(predict_for)
            if row_df is None:
                raise RuntimeError(
                    f"Keine Daten für {predict_for} in DB oder features.csv verfügbar."
                )
            target = self._params["features"]["target_column"]
            cols = [c for c in row_df.columns if c not in _EXCLUDE_COLS and c != target]
            features_dict = row_df[cols].iloc[0].to_dict()

        # Feature-Vektor in der Trainingsreihenfolge aufbauen
        if self._feature_cols:
            feature_values = [features_dict.get(c, 0.0) for c in self._feature_cols]
        else:
            feature_values = list(features_dict.values())

        # TODO (1/2): Vorhersage berechnen
        # Bauen Sie aus feature_values ein numpy-Array der Form (1, n_features) auf
        # und rufen Sie self._model.predict() auf.
        # Speichern Sie das Ergebnis als float in predicted_price.
        raise NotImplementedError("TODO: predict – Vorhersage berechnen")

        confidence = 0.0  # Optional: Standardabweichung der Bäume als Konfidenz-Maß

        # TODO (2/2): Vorhersage in die predictions-Tabelle schreiben
        # Verwenden Sie get_connection() und führen Sie ein INSERT aus:
        #   Tabelle: "{GROUP_ID}".predictions
        #   Spalten: prediction_ts, predicted_price, model_version
        # Fehler beim Logging sollen das Programm nicht stoppen (try/except).
        raise NotImplementedError("TODO: predict – Vorhersage in DB loggen")

        return predicted_price, str(self._model_version), confidence


# Globale Singleton-Instanz
_predictor: Optional[FuelPricePredictor] = None


def get_predictor() -> FuelPricePredictor:
    """Factory-Funktion für den Singleton-Predictor."""
    global _predictor
    if _predictor is None:
        _predictor = FuelPricePredictor()
    return _predictor
