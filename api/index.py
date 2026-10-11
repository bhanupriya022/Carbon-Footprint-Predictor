"""
Carbon Footprint Predictor — Vercel Serverless Function
Handles /api/predict, /api/vehicle-types, and /api/metrics
"""

import json
import os
import numpy as np
import joblib
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Resolve paths relative to this file ────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load once at module level (cold start only, then cached for the function lifetime)
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

VEHICLE_TYPES = METRICS["vehicle_types"]


def _json_response(handler, status: int, body: dict):
    payload = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path in ("/api/vehicle-types", "/vehicle-types"):
            _json_response(self, 200, {"vehicle_types": VEHICLE_TYPES})

        elif path in ("/api/metrics", "/metrics"):
            _json_response(self, 200, METRICS)

        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")

        if path not in ("/api/predict", "/predict"):
            _json_response(self, 404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length))
        except Exception:
            _json_response(self, 400, {"error": "Invalid JSON"})
            return

        required = [
            "vehicle_type", "distance_km", "fuel_consumption_l",
            "monthly_electricity_kwh", "public_transport_km", "flights_per_year",
            "waste_kg_month", "meat_meals_per_week", "household_size",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            _json_response(self, 400, {"error": f"Missing fields: {missing}"})
            return

        vehicle = data["vehicle_type"]
        if vehicle not in VEHICLE_TYPES:
            _json_response(self, 400, {
                "error": f"Unknown vehicle type '{vehicle}'. Valid: {VEHICLE_TYPES}"
            })
            return

        vehicle_enc = le.transform([vehicle])[0]

        features = np.array([[
            vehicle_enc,
            float(data["distance_km"]),
            float(data["fuel_consumption_l"]),
            float(data["monthly_electricity_kwh"]),
            float(data["public_transport_km"]),
            float(data["flights_per_year"]),
            float(data["waste_kg_month"]),
            float(data["meat_meals_per_week"]),
            float(data["household_size"]),
        ]])

        rf_pred = round(float(rf_model.predict(features)[0]), 2)
        lr_pred = round(float(lr_model.predict(features)[0]), 2)

        avg = (rf_pred + lr_pred) / 2
        if avg < 80:
            category, color = "Low", "#22c55e"
        elif avg < 180:
            category, color = "Moderate", "#f59e0b"
        else:
            category, color = "High", "#ef4444"

        _json_response(self, 200, {
            "random_forest":     rf_pred,
            "linear_regression": lr_pred,
            "average":           round(avg, 2),
            "category":          category,
            "category_color":    color,
            "unit":              "kg CO₂/month",
        })
