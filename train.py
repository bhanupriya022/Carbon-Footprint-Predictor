"""
Carbon Footprint Predictor — Model Training Script
Trains Random Forest Regression and Linear Regression models,
generates EDA charts, saves models and a feature-importance chart.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "carbon.csv"
MODELS_DIR  = "models"
CHARTS_DIR  = "charts"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nBasic stats:\n", df.describe())

# ── 2. EDA charts ──────────────────────────────────────────────────────────────

# 2a. Distribution of carbon emissions
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["carbon_emission_kg"], kde=True, color="#3b82d4", ax=ax)
ax.set_title("Distribution of Carbon Emissions")
ax.set_xlabel("Carbon Emission (kg)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/emission_distribution.png", dpi=100)
plt.close()

# 2b. Emissions by vehicle type
fig, ax = plt.subplots(figsize=(8, 4))
order = df.groupby("vehicle_type")["carbon_emission_kg"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="vehicle_type", y="carbon_emission_kg", order=order,
            palette="muted", ax=ax)
ax.set_title("Carbon Emissions by Vehicle Type")
ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Carbon Emission (kg)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/emissions_by_vehicle.png", dpi=100)
plt.close()

# 2c. Correlation heatmap (numeric columns only)
fig, ax = plt.subplots(figsize=(8, 6))
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/correlation_heatmap.png", dpi=100)
plt.close()

# 2d. Scatter: distance vs emission coloured by vehicle
fig, ax = plt.subplots(figsize=(7, 4))
for vtype in df["vehicle_type"].unique():
    sub = df[df["vehicle_type"] == vtype]
    ax.scatter(sub["distance_km"], sub["carbon_emission_kg"], label=vtype, alpha=0.7, s=50)
ax.set_title("Distance vs Carbon Emission")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Carbon Emission (kg)")
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/distance_vs_emission.png", dpi=100)
plt.close()

print("EDA charts saved.")

# ── 3. Preprocessing ───────────────────────────────────────────────────────────
le = LabelEncoder()
df["vehicle_type_enc"] = le.fit_transform(df["vehicle_type"])
joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")

FEATURES = [
    "vehicle_type_enc", "distance_km", "fuel_consumption_l",
    "monthly_electricity_kwh", "public_transport_km", "flights_per_year",
    "waste_kg_month", "meat_meals_per_week", "household_size"
]
TARGET = "carbon_emission_kg"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train models ────────────────────────────────────────────────────────────

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                           min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)
rf_cv   = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()

print(f"\nRandom Forest  — MAE: {rf_mae:.2f}  RMSE: {rf_rmse:.2f}  R²: {rf_r2:.4f}  CV-R²: {rf_cv:.4f}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)
lr_cv   = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()

print(f"Linear Regression — MAE: {lr_mae:.2f}  RMSE: {lr_rmse:.2f}  R²: {lr_r2:.4f}  CV-R²: {lr_cv:.4f}")

# ── 5. Save models ─────────────────────────────────────────────────────────────
joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
joblib.dump(lr, f"{MODELS_DIR}/linear_regression.pkl")
print("Models saved.")

# ── 6. Feature importance chart (RF) ──────────────────────────────────────────
feat_labels = [
    "Vehicle Type", "Distance (km)", "Fuel Consumption (L)",
    "Electricity (kWh)", "Public Transport (km)", "Flights/yr",
    "Waste (kg/mo)", "Meat Meals/wk", "Household Size"
]
importances = rf.feature_importances_
idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(7, 5))
colors = sns.color_palette("Blues_d", len(feat_labels))
ax.barh([feat_labels[i] for i in idx], importances[idx], color=[colors[i] for i in range(len(idx))])
ax.set_title("Random Forest — Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/feature_importance.png", dpi=100)
plt.close()

# ── 7. Actual vs Predicted chart ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, pred, title, color in zip(
    axes,
    [rf_pred, lr_pred],
    ["Random Forest", "Linear Regression"],
    ["#3b82d4", "#7c5cd8"]
):
    ax.scatter(y_test, pred, alpha=0.75, color=color, edgecolors="white", linewidths=0.4)
    mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax.set_title(f"{title}\nR² = {r2_score(y_test, pred):.4f}")
    ax.set_xlabel("Actual Emission (kg)")
    ax.set_ylabel("Predicted Emission (kg)")

plt.suptitle("Actual vs Predicted Carbon Emissions", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/actual_vs_predicted.png", dpi=100, bbox_inches="tight")
plt.close()

print("All charts saved.")
print("\nTraining complete - Done!")

# ── 8. Save metrics for the API ────────────────────────────────────────────────
import json
metrics = {
    "random_forest": {
        "mae": round(rf_mae, 2), "rmse": round(rf_rmse, 2),
        "r2": round(rf_r2, 4),  "cv_r2": round(rf_cv, 4)
    },
    "linear_regression": {
        "mae": round(lr_mae, 2), "rmse": round(lr_rmse, 2),
        "r2": round(lr_r2, 4),  "cv_r2": round(lr_cv, 4)
    },
    "vehicle_types": list(le.classes_)
}
with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to models/metrics.json")
