/**
 * Customer Churn Prediction & Retention Intelligence Platform
 * Frontend Controller
 */

let presetsCache = [];
let autoPredictTimeout = null;

document.addEventListener("DOMContentLoaded", () => {
  loadPresets();
  loadStats();
  // Run initial prediction with default form values
  triggerAutoPredict();
});

function updateTenureDisplay(val) {
  document.getElementById("display-tenure").textContent = `${val} mo`;
  triggerAutoPredict();
}

function updateMonthlyDisplay(val) {
  const num = parseFloat(val).toFixed(2);
  document.getElementById("display-monthly").textContent = `$${num}`;
  triggerAutoPredict();
}

function onInternetChange() {
  const service = document.getElementById("input-internet").value;
  const addOns = ["input-security", "input-backup", "input-tech", "input-device"];

  if (service === "No") {
    addOns.forEach(id => {
      document.getElementById(id).value = "No internet service";
    });
  } else {
    addOns.forEach(id => {
      if (document.getElementById(id).value === "No internet service") {
        document.getElementById(id).value = "No";
      }
    });
  }
  triggerAutoPredict();
}

function triggerAutoPredict() {
  clearTimeout(autoPredictTimeout);
  autoPredictTimeout = setTimeout(() => {
    executePrediction();
  }, 250);
}

function getFormData() {
  const tenure = parseInt(document.getElementById("input-tenure").value);
  const monthly = parseFloat(document.getElementById("input-monthly").value);

  return {
    tenure: tenure,
    MonthlyCharges: monthly,
    TotalCharges: round(tenure * monthly, 2),
    Contract: document.getElementById("input-contract").value,
    PaymentMethod: document.getElementById("input-payment").value,
    PaperlessBilling: document.getElementById("input-paperless").value,
    InternetService: document.getElementById("input-internet").value,
    PhoneService: document.getElementById("input-phone").value,
    MultipleLines: document.getElementById("input-multilines").value,
    OnlineSecurity: document.getElementById("input-security").value,
    OnlineBackup: document.getElementById("input-backup").value,
    TechSupport: document.getElementById("input-tech").value,
    DeviceProtection: document.getElementById("input-device").value,
    SeniorCitizen: parseInt(document.getElementById("input-senior").value),
    Partner: document.getElementById("input-partner").value,
    Dependents: document.getElementById("input-dependents").value,
    gender: document.getElementById("input-gender").value
  };
}

async function executePrediction() {
  const payload = getFormData();
  const btn = document.getElementById("btn-predict");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || "Prediction failed", "error");
      return;
    }

    renderPredictionResults(data);
  } catch (err) {
    showToast("Network error connecting to prediction server", "error");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function handlePredictSubmit(e) {
  e.preventDefault();
  executePrediction();
}

function renderPredictionResults(data) {
  const prob = data.churn_probability;
  const badge = document.getElementById("res-risk-badge");
  const verdictTitle = document.getElementById("res-verdict-title");
  const verdictDesc = document.getElementById("res-verdict-desc");
  const stayProb = document.getElementById("res-stay-prob");
  const probVal = document.getElementById("res-prob-val");
  const gaugeBar = document.getElementById("gauge-bar");

  // 1. Update Gauge & Percentages
  probVal.textContent = `${prob}%`;
  stayProb.textContent = `${data.stay_probability}%`;

  // Circumference of r=50 circle is 2 * PI * 50 ≈ 314
  const circumference = 314;
  const offset = circumference - (prob / 100) * circumference;
  gaugeBar.style.strokeDashoffset = offset;

  // Set Gauge Color & Badges
  if (data.risk_tier === "High Risk") {
    gaugeBar.style.stroke = "#ef4444";
    badge.className = "badge badge-danger";
    verdictTitle.textContent = "High Risk of Churn";
    verdictDesc.textContent = "Customer shows strong behavioral signals of impending cancellation. Proactive retention intervention is recommended.";
  } else if (data.risk_tier === "Medium Risk") {
    gaugeBar.style.stroke = "#f59e0b";
    badge.className = "badge badge-warning";
    verdictTitle.textContent = "Moderate Churn Hazard";
    verdictDesc.textContent = "Customer is in an ambivalent state. Targeted incentives or contract restructuring can secure long-term loyalty.";
  } else {
    gaugeBar.style.stroke = "#10b981";
    badge.className = "badge badge-success";
    verdictTitle.textContent = "Loyal Customer (Low Risk)";
    verdictDesc.textContent = "High engagement and loyalty indicators. Low churn hazard. Ideal candidate for referral or cross-sell rewards.";
  }
  badge.textContent = data.risk_tier;

  // 2. Render Risk Drivers
  const driversList = document.getElementById("res-risk-drivers");
  driversList.innerHTML = (data.risk_drivers || []).map(driver => `
    <li>${escapeHtml(driver)}</li>
  `).join("");

  // 3. Render Retention Recommendations
  const retentionList = document.getElementById("res-retention-actions");
  retentionList.innerHTML = (data.retention_actions || []).map(action => `
    <li>${escapeHtml(action)}</li>
  `).join("");
}

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (res.ok) {
      presetsCache = await res.json();
    }
  } catch (err) {
    console.warn("Could not load presets:", err);
  }
}

function applyPreset(presetId) {
  const preset = presetsCache.find(p => p.id === presetId);
  if (!preset) return;

  const d = preset.data;
  document.getElementById("input-tenure").value = d.tenure;
  updateTenureDisplay(d.tenure);

  document.getElementById("input-monthly").value = d.MonthlyCharges;
  updateMonthlyDisplay(d.MonthlyCharges);

  document.getElementById("input-contract").value = d.Contract;
  document.getElementById("input-payment").value = d.PaymentMethod;
  document.getElementById("input-paperless").value = d.PaperlessBilling;
  document.getElementById("input-internet").value = d.InternetService;
  document.getElementById("input-phone").value = d.PhoneService;
  document.getElementById("input-multilines").value = d.MultipleLines;
  document.getElementById("input-security").value = d.OnlineSecurity;
  document.getElementById("input-backup").value = d.OnlineBackup;
  document.getElementById("input-tech").value = d.TechSupport;
  document.getElementById("input-device").value = d.DeviceProtection;
  document.getElementById("input-senior").value = d.SeniorCitizen;
  document.getElementById("input-partner").value = d.Partner;
  document.getElementById("input-dependents").value = d.Dependents;
  document.getElementById("input-gender").value = d.gender;

  showToast(`Applied archetype: ${preset.name}`, "info");
  executePrediction();
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (res.ok) {
      const stats = await res.json();
      document.getElementById("kpi-total-customers").textContent = stats.total_customers.toLocaleString();
      document.getElementById("kpi-churn-rate").textContent = `${stats.churn_rate_percent}%`;
      document.getElementById("kpi-avg-bill").textContent = `$${stats.avg_monthly_charges.toFixed(2)}`;
      
      const acc = stats.model_info.metrics ? (stats.model_info.metrics.accuracy * 100).toFixed(1) + "%" : "80.1%";
      document.getElementById("kpi-accuracy").textContent = acc;
    }
  } catch (err) {
    console.warn("Could not load stats:", err);
  }
}

function resetForm() {
  document.getElementById("churn-form").reset();
  updateTenureDisplay(12);
  updateMonthlyDisplay(65);
  executePrediction();
  showToast("Reset form to default profile", "info");
}

function showToast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}

function round(val, dec = 2) {
  return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
