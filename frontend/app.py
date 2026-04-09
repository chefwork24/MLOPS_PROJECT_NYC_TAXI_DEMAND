import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NYC Taxi Demand Dashboard",
    page_icon="🚕",
    layout="wide"
)

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def get_prediction(payload):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_recommendation(payload):
    try:
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_batch_prediction(zones_payload):
    try:
        r = requests.post(f"{API_URL}/predict/batch", json=zones_payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ── Status colour ─────────────────────────────────────────────────────────────
def status_color(status):
    if status == "Surge":
        return "🔴"
    elif status == "Dead Zone":
        return "🔵"
    return "🟢"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚕 NYC Taxi Demand")
st.sidebar.markdown("---")

health = get_health()
if health.get("status") == "ok":
    st.sidebar.success("API: Online ✅")
else:
    st.sidebar.error("API: Offline ❌")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 Single Zone Prediction", "🗺️ Zone Heatmap", "🚗 Driver Recommendations"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 NYC Taxi Demand — Operations Dashboard")
    st.markdown("Real-time demand forecasting across 263 NYC taxi zones.")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",        "LightGBM")
    col2.metric("Zones Covered", "263")
    col3.metric("Forecast Horizon", "6 Hours")
    col4.metric("API Status", "Online" if health.get("status") == "ok" else "Offline")

    st.markdown("---")

    # Sample batch prediction for top zones
    st.subheader("Top Zone Demand Forecast")

    top_zones = [161, 162, 237, 236, 230, 186, 48, 170, 234, 142]

    batch_payload = {
        "zones": [
            {
                "zone_id": z,
                "hour_of_day": 18,
                "day_of_week": 2,
                "month": 4,
                "is_weekend": 0,
                "is_rush_hour": 1,
                "demand_lag_1h": 400.0,
                "demand_lag_24h": 380.0,
                "demand_lag_168h": 370.0,
                "rolling_mean_24h": 350.0,
                "rolling_mean_7d": 360.0,
                "is_airport_zone": 1 if z in [1, 132, 138] else 0
            }
            for z in top_zones
        ]
    }

    result = get_batch_prediction(batch_payload)

    if "predictions" in result:
        df = pd.DataFrame(result["predictions"])
        df["status_icon"] = df["demand_status"].apply(status_color)
        df["display"] = df["status_icon"] + " " + df["demand_status"]

        # Bar chart
        fig = px.bar(
            df,
            x="zone_id",
            y="predicted_demand",
            color="demand_status",
            color_discrete_map={"Surge": "red", "Normal": "green", "Dead Zone": "blue"},
            title="Predicted Hourly Demand — Top 10 Zones",
            labels={"zone_id": "Zone ID", "predicted_demand": "Predicted Pickups"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(
            df[["zone_id", "predicted_demand", "display"]].rename(columns={
                "zone_id": "Zone",
                "predicted_demand": "Predicted Demand",
                "display": "Status"
            }),
            use_container_width=True
        )
    else:
        st.error("Could not fetch predictions. Is the API running?")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Zone Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Zone Prediction":
    st.title("🔍 Single Zone Demand Prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        zone_id      = st.selectbox("Zone ID", options=list(range(1, 264)), index=160)
        hour_of_day  = st.slider("Hour of Day", 0, 23, 18)
        day_of_week  = st.selectbox("Day of Week", [1,2,3,4,5,6,7],
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])
        month        = st.selectbox("Month", list(range(1, 13)),
                                     format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                             "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

    with col2:
        is_weekend    = st.checkbox("Weekend", value=False)
        is_rush_hour  = st.checkbox("Rush Hour", value=True)
        is_airport    = st.checkbox("Airport Zone", value=False)
        demand_lag_1h    = st.number_input("Demand Lag 1h",   value=400.0)
        demand_lag_24h   = st.number_input("Demand Lag 24h",  value=380.0)
        demand_lag_168h  = st.number_input("Demand Lag 168h", value=370.0)
        rolling_mean_24h = st.number_input("Rolling Mean 24h", value=350.0)
        rolling_mean_7d  = st.number_input("Rolling Mean 7d",  value=360.0)

    if st.button("Predict Demand", type="primary"):
        payload = {
            "zone_id":          zone_id,
            "hour_of_day":      hour_of_day,
            "day_of_week":      day_of_week,
            "month":            month,
            "is_weekend":       int(is_weekend),
            "is_rush_hour":     int(is_rush_hour),
            "demand_lag_1h":    demand_lag_1h,
            "demand_lag_24h":   demand_lag_24h,
            "demand_lag_168h":  demand_lag_168h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_mean_7d":  rolling_mean_7d,
            "is_airport_zone":  int(is_airport)
        }

        result = get_prediction(payload)

        if "predicted_demand" in result:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Demand",   f"{result['predicted_demand']:.0f} pickups")
            col2.metric("Demand Status",      f"{status_color(result['demand_status'])} {result['demand_status']}")
            col3.metric("Zone",               f"Zone {result['zone_id']}")
        else:
            st.error(f"Error: {result}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Zone Heatmap
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Zone Heatmap":
    st.title("🗺️ NYC Zone Demand Heatmap")
    st.markdown("Demand forecast visualised across NYC zones.")
    st.markdown("---")

    # Sample zones with approximate NYC coordinates
    zone_coords = {
        161: (40.7549, -73.9840),  # Midtown
        162: (40.7489, -73.9680),  # Midtown East
        237: (40.7736, -73.9566),  # Upper East Side
        236: (40.7794, -73.9632),  # Upper East Side N
        230: (40.7648, -73.9808),  # Times Square
        186: (40.7282, -74.0776),  # Newark Airport area
        48:  (40.6501, -73.9496),  # Crown Heights
        170: (40.7580, -73.9855),  # Midtown West
        234: (40.7580, -73.9697),  # Upper East Side S
        142: (40.7484, -73.9967),  # Lincoln Tunnel
        132: (40.6413, -73.7781),  # JFK Airport
        138: (40.7769, -73.8740),  # LaGuardia Airport
        1:   (40.6895, -74.1745),  # Newark Airport
    }

    batch_payload = {
        "zones": [
            {
                "zone_id": z,
                "hour_of_day": 18,
                "day_of_week": 2,
                "month": 4,
                "is_weekend": 0,
                "is_rush_hour": 1,
                "demand_lag_1h": 400.0,
                "demand_lag_24h": 380.0,
                "demand_lag_168h": 370.0,
                "rolling_mean_24h": 350.0,
                "rolling_mean_7d": 360.0,
                "is_airport_zone": 1 if z in [1, 132, 138] else 0
            }
            for z in zone_coords.keys()
        ]
    }

    result = get_batch_prediction(batch_payload)

    if "predictions" in result:
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

        for pred in result["predictions"]:
            zid    = pred["zone_id"]
            demand = pred["predicted_demand"]
            status = pred["demand_status"]

            if zid not in zone_coords:
                continue

            lat, lon = zone_coords[zid]

            color = "red" if status == "Surge" else "blue" if status == "Dead Zone" else "green"

            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(30, demand / 20)),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Zone {zid}<br>Demand: {demand:.0f}<br>Status: {status}"
            ).add_to(m)

        st_folium(m, width=900, height=500)

        # Legend
        st.markdown("🔴 Surge &nbsp;&nbsp; 🟢 Normal &nbsp;&nbsp; 🔵 Dead Zone")
    else:
        st.error("Could not fetch predictions. Is the API running?")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Driver Recommendations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚗 Driver Recommendations":
    st.title("🚗 Driver Pre-positioning Recommendations")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        zone_id         = st.selectbox("Zone ID", options=list(range(1, 264)), index=160)
        predicted_demand = st.number_input("Predicted Demand", value=480.0)
    with col2:
        current_drivers = st.number_input("Current Drivers in Zone", value=20, step=1)

    if st.button("Get Recommendation", type="primary"):
        payload = {
            "zone_id":          zone_id,
            "predicted_demand": predicted_demand,
            "current_drivers":  int(current_drivers)
        }

        result = get_recommendation(payload)

        if "recommended_drivers" in result:
            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Recommended Drivers", result["recommended_drivers"])
            col2.metric("Current Drivers",     current_drivers)
            st.info(f"💡 Action: {result['action']}")
        else:
            st.error(f"Error: {result}")