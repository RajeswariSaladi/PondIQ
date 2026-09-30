from flask import Flask, render_template
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# Load simulated IoT sensor data
data = pd.read_csv("data/sensor_data.csv")

# Parameters used by the AI anomaly detection model
features = [
    "temperature",
    "ph",
    "dissolved_oxygen",
    "turbidity",
    "ammonia"
]

# AI anomaly detection using Isolation Forest
model = IsolationForest(
    contamination=0.30,
    random_state=42
)

data["anomaly"] = model.fit_predict(data[features])

# Latest and first sensor readings
latest = data.iloc[-1]
first = data.iloc[0]


# --------------------------------------------------
# AI ANOMALY STATUS
# --------------------------------------------------

if latest["anomaly"] == -1:
    anomaly_status = "ANOMALY DETECTED"
    anomaly_message = (
        "AI detected an unusual combination of water-quality readings."
    )
else:
    anomaly_status = "NORMAL"
    anomaly_message = (
        "AI found the latest sensor pattern within the normal range."
    )


# --------------------------------------------------
# POND RISK CALCULATION
# --------------------------------------------------

risk_points = 0

if latest["dissolved_oxygen"] < 4.5:
    risk_points += 2

if latest["ammonia"] > 0.15:
    risk_points += 2

if latest["temperature"] > 30:
    risk_points += 1

if latest["ph"] > 8.2:
    risk_points += 1

if latest["aerator"] == "OFF":
    risk_points += 2


if risk_points >= 5:
    risk = "HIGH"
elif risk_points >= 3:
    risk = "MEDIUM"
else:
    risk = "LOW"


# --------------------------------------------------
# DETECT WHAT ACTUALLY CHANGED
# --------------------------------------------------

changes = []

if latest["dissolved_oxygen"] < first["dissolved_oxygen"]:
    changes.append(
        f"Dissolved oxygen decreased from "
        f"{first['dissolved_oxygen']} to "
        f"{latest['dissolved_oxygen']} mg/L."
    )

if latest["ammonia"] > first["ammonia"]:
    changes.append(
        f"Ammonia increased from "
        f"{first['ammonia']} to "
        f"{latest['ammonia']} mg/L."
    )

if latest["temperature"] > first["temperature"]:
    changes.append(
        f"Temperature increased from "
        f"{first['temperature']} to "
        f"{latest['temperature']} °C."
    )

if latest["ph"] > first["ph"]:
    changes.append(
        f"pH increased from "
        f"{first['ph']} to "
        f"{latest['ph']}."
    )

if latest["turbidity"] > first["turbidity"]:
    changes.append(
        f"Turbidity increased from "
        f"{first['turbidity']} to "
        f"{latest['turbidity']} NTU."
    )

if latest["aerator"] == "OFF":
    changes.append("Aerator is currently OFF.")


# --------------------------------------------------
# AI EXPLANATION
# --------------------------------------------------

if risk == "HIGH":
    ai_explanation = (
        "Multiple water-quality signals are moving in an unfavorable "
        "direction. The combination of falling dissolved oxygen, rising "
        "ammonia, increasing temperature and an inactive aerator indicates "
        "elevated pond-stress risk."
    )

elif risk == "MEDIUM":
    ai_explanation = (
        "Some water-quality parameters are changing from the earlier "
        "readings. The pond should be monitored more closely."
    )

else:
    ai_explanation = (
        "Current readings are relatively stable compared with the "
        "earlier readings."
    )


# --------------------------------------------------
# FARMER-FRIENDLY RECOMMENDATION
# --------------------------------------------------

if risk == "HIGH":
    recommendation = (
        "Verify the sensor readings and check the aerator immediately. "
        "Increase monitoring frequency and consult an aquaculture "
        "professional if the abnormal condition persists."
    )

elif risk == "MEDIUM":
    recommendation = (
        "Continue monitoring the pond closely and verify any abnormal "
        "readings. Check aeration if dissolved oxygen continues to decrease."
    )

else:
    recommendation = (
        "Continue routine monitoring and record new readings as they "
        "become available."
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html",

        temperature=latest["temperature"],
        ph=latest["ph"],
        dissolved_oxygen=latest["dissolved_oxygen"],
        turbidity=latest["turbidity"],
        ammonia=latest["ammonia"],
        aerator=latest["aerator"],

        risk=risk,

        anomaly_status=anomaly_status,
        anomaly_message=anomaly_message,

        changes=changes,
        ai_explanation=ai_explanation,
        recommendation=recommendation,

        timestamps=data["timestamp"].tolist(),
        temperatures=data["temperature"].tolist(),
        ph_values=data["ph"].tolist(),
        oxygen_values=data["dissolved_oxygen"].tolist(),
        ammonia_values=data["ammonia"].tolist()
    )


if __name__ == "__main__":
    app.run(debug=True)