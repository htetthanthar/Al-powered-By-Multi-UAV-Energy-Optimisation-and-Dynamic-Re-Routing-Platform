# Multi-UAV Energy Optimisation and Dynamic Re-Routing Platform

**Author:** Than Thar Han  
**Institution:** Coventry University / PSB Academy (`6002CEM` Individual Project)

An end-to-end simulation and command platform for multi-UAV last-mile parcel delivery in Singapore's low-altitude urban airspace (`< 60 m` AGL, `103.70°E – 104.00°E`, `1.275°N – 1.450°N`), combining:

- **Deep Energy Surrogate (MLP)**: Predicts segment electrical power consumption ($P_e$ in Watts) and leg energy ($E_{ij}$ in Wh) as a function of airspeed, payload mass, headwind/tailwind vector, climb rate, and ambient temperature (held-out validation **$R^2 = 0.9942$**).
- **Attention-Based Graph Neural Network (GNN) + Visibility-Graph Planner**: Computes collision-free routes around Singapore's active No-Fly Zones (`NFZ-A` Central Catchment, `NFZ-B` Paya Lebar Air Base Buffer, `NFZ-C` Changi Approach Buffer + `500 m` safety buffer) while minimising total fleet energy.
- **16-UAV Continuous Hub-and-Spoke Loop**: All 16 drones (`UAV-ALPHA-01` through `UAV-PAPA-16`) depart from **Headquarters (`Queenstown Central Depot`, `1.3000°N, 103.8000°E`)**, fly along corridors `RT-01` to `RT-16` to 16 distinct Singapore destinations, drop their parcel payloads, return to Headquarters, and repeat in a continuous loop.

---

## Interactive Dashboards (Streamlit Cloud & Local)

When deployed on **Streamlit Community Cloud** (via [`streamlit_app.py`](streamlit_app.py) or [`All sensor/dashboard.py`](All%20sensor/dashboard.py)), the app provides three live tabs:

1. **`📡 2D TACTICAL SENSOR RADAR & BMS TELEMETRY`** — Interactive 2D Leaflet satellite radar with live NFZ creation, BMS/ESC thermal profiling, LiDAR range rings, optical flow lock, and individual/global emergency recall (`RTL`).
2. **`🌐 STRATUM 3D MULTI-UAV COMMAND & NEURAL SURROGATE STUDIO`** — Interactive 3D CesiumJS Singapore globe, real-time MLP + GNN telemetry charts, wind vector slider (`0–10 m/s`), and the **Learning Lab** model observability dialog.
3. **`⚡ INTEGRATED WEATHER, FLIR THERMAL & VSLAM COMMAND`** — Multi-sensor weather, thermal IR, and VIO keypoint HUD.

---

## Quick Start (Local Execution)

### 1. Run the 3D Command Server (`http://127.0.0.1:4174`)
```bash
cd uav-command
npm install
npm test
npm start
```

### 2. Run the Streamlit Dashboard (`http://localhost:8501`)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
