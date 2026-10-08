/* ── Carbon Footprint Predictor — main.js ─────────────────────────────────── */

const API = "https://carbon-footprint-predictor.onrender.com";

// ── Load vehicle types into <select> ─────────────────────────────────────────
async function loadVehicleTypes() {
  try {
    const res  = await fetch(`${API}/vehicle-types`);
    const data = await res.json();
    const sel  = document.getElementById("vehicle_type");
    sel.innerHTML = '<option value="" disabled selected>Select…</option>';
    data.vehicle_types.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error("Could not load vehicle types", e);
  }
}

// ── Form validation ───────────────────────────────────────────────────────────
function validateForm(formData) {
  const errors = [];
  const fields = [
    "vehicle_type", "distance_km", "fuel_consumption_l",
    "monthly_electricity_kwh", "public_transport_km", "flights_per_year",
    "waste_kg_month", "meat_meals_per_week", "household_size"
  ];

  fields.forEach(f => {
    const el = document.getElementById(f);
    el.classList.remove("invalid");
    if (!formData[f] && formData[f] !== 0) {
      el.classList.add("invalid");
      errors.push(f);
    }
  });
  return errors;
}

// ── Tips based on category ────────────────────────────────────────────────────
function getTip(category, avg) {
  if (category === "Low") {
    return "🌿 Great job! Your footprint is below average. Keep it up by maintaining your current habits and encouraging others.";
  } else if (category === "Moderate") {
    return "⚡ Your footprint is moderate. Consider reducing meat consumption, switching to public transport, or upgrading to an electric vehicle to lower your impact.";
  } else {
    return "🔴 Your footprint is high. Key actions: reduce flights, switch to an EV or public transport, lower home energy use, and cut meat meals per week.";
  }
}

// ── Form submit ───────────────────────────────────────────────────────────────
async function handleSubmit(e) {
  e.preventDefault();

  const errorEl  = document.getElementById("formError");
  const resultEl = document.getElementById("resultCard");
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");

  // Build payload
  const formData = {
    vehicle_type:            document.getElementById("vehicle_type").value,
    distance_km:             parseFloat(document.getElementById("distance_km").value),
    fuel_consumption_l:      parseFloat(document.getElementById("fuel_consumption_l").value),
    monthly_electricity_kwh: parseFloat(document.getElementById("monthly_electricity_kwh").value),
    public_transport_km:     parseFloat(document.getElementById("public_transport_km").value),
    flights_per_year:        parseFloat(document.getElementById("flights_per_year").value),
    waste_kg_month:          parseFloat(document.getElementById("waste_kg_month").value),
    meat_meals_per_week:     parseFloat(document.getElementById("meat_meals_per_week").value),
    household_size:          parseFloat(document.getElementById("household_size").value),
  };

  const errors = validateForm(formData);
  if (errors.length) {
    errorEl.textContent = `Please fill in all required fields.`;
    errorEl.classList.remove("hidden");
    return;
  }

  // Loading state
  const submitBtn = document.getElementById("submitBtn");
  document.getElementById("btnText").classList.add("hidden");
  document.getElementById("btnSpinner").classList.remove("hidden");
  submitBtn.disabled = true;

  try {
    const res  = await fetch(`${API}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(formData),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const lr = data.linear_regression;

    // ── Result card (inline) ──────────────────────────────────────────────────
    document.getElementById("lrValue").textContent = lr;

    const badge = document.getElementById("resultCategory");
    badge.textContent = data.category;
    badge.style.background = data.category_color;
    setResultGlow(data.category_color);

    // Gauge — max display 500 kg
    const pct = Math.min((lr / 500) * 100, 100);
    const gaugeFill = document.getElementById("gaugeFill");
    gaugeFill.style.width = `${pct}%`;
    gaugeFill.style.background = data.category_color;

    // Tip
    document.getElementById("resultTip").textContent = getTip(data.category, lr);

    // ── Dashboard section ─────────────────────────────────────────────────────
    document.getElementById("dashboardEmpty").classList.add("hidden");
    document.getElementById("dashboardContent").classList.remove("hidden");

    document.getElementById("dashValue").textContent  = lr;
    document.getElementById("dashAnnual").textContent = (lr * 12).toFixed(1);
    // 1 tree absorbs ~21 kg CO₂/year
    document.getElementById("dashTrees").textContent  = Math.ceil((lr * 12) / 21);

    const avg = 152;
    const diff = (lr - avg).toFixed(1);
    const vsEl = document.getElementById("dashVsAvg");
    vsEl.textContent = (diff > 0 ? "+" : "") + diff + " kg";
    vsEl.style.color = diff > 0 ? "var(--danger)" : "var(--accent-dark)";
    document.getElementById("dashVsAvgLabel").textContent =
      diff > 0 ? "above dataset average" : "below dataset average";

    const dashGauge = document.getElementById("dashGaugeFill");
    dashGauge.style.width = `${pct}%`;
    dashGauge.style.background = data.category_color;

    const dashBadge = document.getElementById("dashBadge");
    dashBadge.textContent = data.category;
    dashBadge.style.background = data.category_color;

    resultEl.classList.remove("hidden");
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  } finally {
    document.getElementById("btnText").classList.remove("hidden");
    document.getElementById("btnSpinner").classList.add("hidden");
    submitBtn.disabled = false;
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function handleReset() {
  document.getElementById("predictForm").reset();
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("formError").classList.add("hidden");
  document.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
}

// ── Navbar scroll effect ──────────────────────────────────────────────────────
window.addEventListener("scroll", () => {
  document.getElementById("navbar").classList.toggle("scrolled", window.scrollY > 10);
});

// ── Result glow colour ────────────────────────────────────────────────────────
function setResultGlow(color) {
  const glow = document.getElementById("resultGlow");
  if (glow) glow.style.background = color;
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadVehicleTypes();
  document.getElementById("predictForm").addEventListener("submit", handleSubmit);
  document.getElementById("resetBtn").addEventListener("click", handleReset);
});
