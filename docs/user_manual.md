# User Manual — NYC Taxi Demand Dashboard

## What This Application Does
This dashboard helps taxi fleet dispatchers predict demand across
NYC zones and decide where to position drivers.

## Starting the Application

### Step 1 — Start all services
Open 4 terminal windows and run one command in each:

Terminal 1 — MLflow:
  cd MLOPS_PROJECT && ./start_mlflow.sh

Terminal 2 — Backend API:
  cd MLOPS_PROJECT/src/api && uvicorn main:app --host 0.0.0.0 --port 8000

Terminal 3 — Frontend:
  cd MLOPS_PROJECT/frontend && streamlit run app.py --server.port 8502

Terminal 4 — Monitoring (optional):
  cd MLOPS_PROJECT && docker compose up

### Step 2 — Open the Dashboard
Open your browser and go to: http://localhost:8502

---

## Using the Dashboard

### Page 1 — Dashboard (Home)
Shows the top 10 busiest NYC zones with predicted demand.
- Green bars = Normal demand
- Red bars = Surge (high demand)
- Blue bars = Dead Zone (low demand)

### Page 2 — Single Zone Prediction
Predict demand for any specific zone.

How to use:
1. Select a Zone ID from the dropdown (e.g. 161 = Midtown)
2. Set the hour of day (0–23)
3. Set the day of week (1=Monday to 7=Sunday)
4. Set the month (1–12)
5. Check "Rush Hour" if between 7–9am or 5–7pm
6. Enter the last known demand values in the lag fields
7. Click "Predict Demand"

Result shows:
- Predicted pickups for that zone and hour
- Demand status: Normal, Surge, or Dead Zone

### Page 3 — Zone Heatmap
Shows a map of NYC with demand circles per zone.
- Circle size = predicted demand volume
- Red = Surge zones needing more drivers
- Green = Normal zones
- Blue = Dead zones with excess drivers

### Page 4 — Driver Recommendations
Get a recommendation for how many drivers a zone needs.

How to use:
1. Select Zone ID
2. Enter predicted demand (from Page 2)
3. Enter how many drivers are currently in that zone
4. Click "Get Recommendation"

Result shows recommended driver count and action to take.

---

## Monitoring (For Technical Users)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (login: admin/admin)
- API docs: http://localhost:8000/docs

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Dashboard shows "API: Offline" | Start the backend API on port 8000 |
| Predictions show error | Check MLflow server is running on port 5000 |
| No data in Grafana | Send a few requests to generate metrics |
| Port already in use | Run: lsof -ti:PORT followed by xargs kill -9 |