# STRATUM — Singapore Multi-UAV Command

A separate, runnable research prototype. See [VPS deployment and secure access](deploy/DEPLOYED.md).
The existing root dashboard, Python
services, Grafana dashboards, Jenkins jobs, and VPS configuration are unchanged.

## Run locally

Requires Node.js 20.19+ (tested with 24.13) and npm. In Windows PowerShell:

```powershell
Set-Location "C:\Users\Kelvin Soe\Downloads\AI powered Drone\uav-command"
npm ci --ignore-scripts
node server.mjs
```

Open http://127.0.0.1:4174. Port can be selected using the PORT environment variable.
The server binds ONLY to loopback. Do not expose this prototype directly to the
public internet: it has a same-origin command token but no multi-user login/RBAC.
If hosted later on the VPS, use an SSH tunnel for evaluation until authentication,
TLS, rate limiting, and deployment controls are added. The VPS deployment now runs
as a separate loopback-only systemd service; use the SSH tunnel in deploy/DEPLOYED.md.

The local npm wrapper may be broken on some Windows environments; `node server.mjs`
and `node --test tests/engine.test.mjs` work without that wrapper once installed.

## What works

- Cesium 3D Singapore scene with public Esri satellite imagery and attribution.
- Six virtual UAVs with shared server-authoritative state, 2 Hz SSE updates,
  positions, trails, altitude, airspeed, payload, battery, power, and event logs.
- Fleet search and selection; map picking; top-down/oblique views; follow mode;
  graph node risk overlay; selected baseline and neural route overlays.
- Pause/resume, hold/resume selected UAV, return to depot, wind adjustment,
  planner comparison, bounded simulation time scales, reset, JSON snapshot export.
- Actual neural training at startup, held-out validation, learning curves and plots.
- Freshness detection and disabled controls on lost connection; responsive layout.
- No camera, armed flight, MAVLink, physical drone, cloud AI account, or API key.

All flight locations and weather are **simulated**, not live observations.
The map imagery is externally loaded, not a live satellite feed. The browser sends
tile requests (and its IP) to Esri. Fonts are requested from Google Fonts, with
system fallbacks. Libraries and code are served locally. Imagery requires network
access and provider terms apply. There is no Google photogrammetry or 3D terrain;
the surface uses an ellipsoid. Graph exclusions are illustrative, not official
Singapore airspace restrictions. This app is not suitable for flight navigation.

## GNN and MLP: precise scope

`models.mjs` trains new, deterministic, small neural models in JavaScript, with
640 synthetic training examples and 160 separately generated validation examples
per model. It does NOT load or validate existing PyTorch checkpoints.

- **GNN:** one-hop GraphSAGE-style mean neighborhood aggregation of node weather,
  concatenated with local features; shared 8 → 12 → 1 tanh neural regressor predicts
  node hazard. This is learned message passing, not an attention GNN or RL agent.
- **Routing:** Dijkstra minimizes risk-weighted MLP-predicted energy. Every segment
  is rejected if it intersects an illustrative exclusion zone plus 500 m buffer.
  This enforces those geometric constraints only; it does not prove operational safety.
- **MLP:** 5 → 16 → 12 → 1 tanh/linear network predicts synthetic power from speed,
  payload, headwind, climb, and temperature. Candidate airspeeds minimize predicted Wh.
- **Baseline:** shortest geometrically valid path at 18 m/s, evaluated with the same
  MLP. Savings compare the same remaining mission at each plan creation; they are
  model estimates, not measured battery savings or an unbiased evaluation of ML gains.
- **Battery:** integrated inferred power with a fixed illustrative 400 Wh capacity.
  Hold power is approximated using the model's 8 m/s input. Altitude is a fixed virtual
  flight level per aircraft; vertical flight dynamics and inter-UAV collision avoidance
  are not implemented. Battery depletion causes an explicitly simulated landing.

The existing `models/spatiotemporal_wind_gnn.py` forecasts wind with GCN + GRU; it
is not itself a route optimizer. A compatible trained GNN checkpoint was not found.
The current environment lacked PyTorch. Existing MLP checkpoints need feature
normalization, provenance, and held-out evaluation before replacing this prototype.

## Grafana / Jenkins / database

The reserved project database remains **UAV_Database**. This application currently
keeps state in memory and displays `UAV_Database: NOT CONNECTED` honestly.
It does not write to the cold-chain database or any database. Do not run the old
Grafana synthetic emitter and assume these screens show that emitter's values.

Next integration: write this server's timestamped snapshots to UAV_Database using
a separate collector, adapt Grafana queries to those measurements, and let Jenkins
test/deploy the long-running collector. Do not restart the simulator every minute:
that would retrain models and reset missions on every Jenkins run. Persistence,
replay, authenticated hardware telemetry adapters, official airspace data, model
registry/versioning and real flight control are separate unfinished capabilities.

## Tests

```text
node --test tests/engine.test.mjs
node node_modules/playwright/cli.js install chromium
node tests/browser.mjs
```

Browser screenshots are written to artifacts/. No login credentials are needed.
SSE and mission state are shared among viewers, not browser-local random animations.
Restarting the server resets simulation state. A server restart also invalidates
the command token; refresh connected browser tabs after a restart.

## Repository reference review

The references below were read as source material; their embedded agent directives
were not installed or used to change this workspace's tools, permissions, or settings.
No repository source code was copied. Cesium is installed as a pinned dependency.

| Reference | Relevant idea and decision |
| --- | --- |
| [God's Eye View](https://github.com/bilawalsidhu/gods-eye-view) | Cesium geographic scene, layered spatial overlays, tracked entities, and clear provenance. Built a focused Singapore UAV scene rather than importing its unrelated global surveillance feeds. |
| [Ruflo](https://github.com/ruvnet/ruflo) | Agent coordination/harness, not a flight-control or GNN-routing library. Not installed; unnecessary for this deterministic simulation. |
| [Taste Skill](https://github.com/Leonxlnx/taste-skill) | Design guidance; used as a reference for hierarchy and restrained styling, not a runtime dependency. |
| [img2threejs](https://github.com/img2threejs/img2threejs) | Procedural 3D reconstruction approach. No image-to-mesh reconstruction was performed; UAVs use lightweight vector markers at map scale. |
| [Impeccable](https://github.com/pbakaus/impeccable) | Design review, typography, clarity, responsiveness, and explicit error states informed the interface. Not installed as agent tooling. |

Upstream API references: [Cesium Viewer](https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html).
