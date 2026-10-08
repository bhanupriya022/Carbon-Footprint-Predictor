"""
Carbon Footprint Predictor — Flask API
Serves the frontend and exposes /predict and /metrics endpoints.
"""

import json
import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Load models ────────────────────────────────────────────────────────────────
MODELS_DIR = "models"

rf_model = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
lr_model = joblib.load(f"{MODELS_DIR}/linear_regression.pkl")
le       = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

with open(f"{MODELS_DIR}/metrics.json") as f:
    METRICS = json.load(f)

VEHICLE_TYPES = METRICS["vehicle_types"]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/charts/<path:filename>")
def serve_chart(filename):
    return send_from_directory("charts", filename)

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(METRICS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = [
        "vehicle_type", "distance_km", "fuel_consumption_l",
        "monthly_electricity_kwh", "public_transport_km", "flights_per_year",
        "waste_kg_month", "meat_meals_per_week", "household_size"
    ]

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    vehicle = data["vehicle_type"]
    if vehicle not in VEHICLE_TYPES:
        return jsonify({"error": f"Unknown vehicle type '{vehicle}'. "
                                  f"Valid types: {VEHICLE_TYPES}"}), 400

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

    # Simple category
    avg = (rf_pred + lr_pred) / 2
    if avg < 80:
        category, color = "Low", "#22c55e"
    elif avg < 180:
        category, color = "Moderate", "#f59e0b"
    else:
        category, color = "High", "#ef4444"

    return jsonify({
        "random_forest": rf_pred,
        "linear_regression": lr_pred,
        "average": round(avg, 2),
        "category": category,
        "category_color": color,
        "unit": "kg CO₂/month"
    })

@app.route("/vehicle-types", methods=["GET"])
def vehicle_types():
    return jsonify({"vehicle_types": VEHICLE_TYPES})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Carbon Footprint Predictor API...")
    print(f"Vehicle types available: {VEHICLE_TYPES}")
    app.run(debug=True, port=5000)
