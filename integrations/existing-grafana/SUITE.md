# Seven UAV Grafana dashboards

Prepared locally; not published to Grafana and not scheduled in Jenkins.
UAV_Database was created and verified on the VPS on 2026-09-10 at the user's request.
Existing service configuration, jobs, dashboards, and datasource settings are unchanged.
This guide supersedes earlier instructions that reused the cold-chain database.

## Included views

| File under dashboards/ | View |
| --- | --- |
| uav-pre-mlp-live.json | Baseline energy: airspeed, headwind, temperature, SoC, route power |
| uav-post-mlp-live.json | Illustrative optimized energy and savings |
| uav-energy-surrogate-live.json | Energy, power, climb, battery voltage, distribution |
| uav-gnn-routing-live.json | Waypoint positions, clearance, wind components, route cost |
| uav-training-live.json | Epoch loss and route-cost curves |
| uav-accuracy-live.json | Prediction scatter, ideal line, residuals, calculated R-squared/RMSE/MAE |
| uav-comparison-live.json | Framework ETA and 55-minute violation rate |

The 53 panels use native Grafana visualizations, dark styling, colored statistics,
XY charts, bar charts and histograms. Navigation links connect the seven views.
Refresh is 10 seconds. Queries require samples within the last 30 seconds, so
stopping the writer produces empty charts instead of displaying stale values as current.

This is a reference-inspired layout, not a pixel-exact copy. In particular the
routing view plots waypoint coordinates and wind components; it does not draw
validated flight paths, mapped NFZ polygons, or geographic wind streamlines.

## Data honesty

The bundled writer is a SIMULATION for testing the dashboard integration.
It does not connect to a physical drone or execute trained MLP/GNN/RL models.
Savings are an explicitly illustrative counterfactual, not measured model gains.
Training curves are synthetic snapshots, not training history. Benchmark samples
are synthetic durations; accuracy scores are calculated from synthetic point pairs.
Example NFZ circles are not authoritative restrictions and are not safe flight guidance.

Every dashboard is marked DEMO and every query selects source=simulation.
To display real data, a real producer must supply the equivalent measurements
and fields, use a different source tag, and the dashboard filters and descriptions
must be updated. Do not merely remove the DEMO label from the fixture.

## Dedicated UAV connection

Use InfluxDB v1 at http://127.0.0.1:8086, database UAV_Database (case-sensitive).
Create a separate Grafana InfluxDB datasource named UAV_Database, using InfluxQL.
Leave the existing cold-chain datasource and the global default unchanged.
These JSON files use an import-time datasource selector, ${DS_UAV_DATABASE},
instead of a hardcoded datasource UID. During UI import select only the new
UAV_Database datasource. Direct HTTP dashboard creation must resolve this
placeholder to that datasource's actual UID before posting the dashboard.
They use only six NEW measurement names: uav_suite_energy, uav_suite_routing,
uav_suite_fit, uav_suite_accuracy, uav_suite_training, uav_suite_comparison.
They neither query nor overwrite the existing sensor_telemetry measurement.

## Publish without changing existing service settings

1. First create the separate datasource as described below. In Grafana, use
   Dashboards > New > Import, and import the seven JSON files
   into a new UAV folder. Each file has a separate stable UID. If a UID already
   exists, inspect it before replacing anything. Select UAV_Database when
   prompted for the datasource; do not select the cold-chain connection.
2. Place suite_emitter.py at /opt/uav-grafana-demo/suite_emitter.py on the VPS
   with read permission for the Jenkins execution user. No root permission is
   required to run the writer; no packages need installing.
3. Create a separate Jenkins Pipeline job named UAV-Grafana-Suite. Paste
   Jenkinsfile.suite into its Pipeline script. Run it on the same VPS as InfluxDB
   because the emitter uses the loopback address. Run the job once to register
   its timer. Do not replace any existing job.
4. Verify the Jenkins build log and all seven Grafana dashboards. Each run writes
   six batches of 294 synthetic points about 10 seconds apart, ending after
   approximately 50 seconds. Jenkins schedules a new run each minute. Queues or
   slow requests can create gaps; this is periodic polling, not Grafana Live streaming.

The new UAV database and a new datasource are required for this isolated setup.
No Grafana restart, plugin installation, existing datasource edit, firewall
change or Jenkins global setting change is required. Publishing creates new
dashboard/job records and writing creates measurement data; it is not a read-only action.
Do not run the older Docker/provisioning installation scripts for this approach.

Grafana application authorization is required to publish via its API. SSH root
access is not used to bypass Grafana authentication. Jenkins job creation likewise
requires authorized Jenkins access. No credentials are included in this bundle.

## Updated setup steps

1. The database already exists on 64.118.128.181. Verify in the VPS terminal:

   ```bash
   curl -sS -G http://127.0.0.1:8086/query --data-urlencode 'q=SHOW DATABASES'
   ```

   For a fresh installation only, create it with:

   ```bash
   curl -sS -X POST http://127.0.0.1:8086/query --data-urlencode 'q=CREATE DATABASE "UAV_Database"'
   ```

2. In Grafana: Connections > Data sources > Add new data source > InfluxDB.
   Set name UAV_Database; query language InfluxQL; URL http://127.0.0.1:8086;
   database UAV_Database; leave Default off. Use existing authorized database
   credentials if required; do not disable authentication. Click Save & test.
   This is a new connection, not a change to the cold-chain connection.
3. Upload the updated suite_emitter.py to /opt/uav-grafana-demo/suite_emitter.py.
   If a previous UAV writer/job is running, disable only that UAV job and wait
   for its current build to finish before replacing the writer. Do not touch
   other jobs. Both supplied writers now reject any database except UAV_Database.
4. Use the updated Jenkinsfile.suite in the UAV-Grafana-Suite job. It explicitly
   passes --database UAV_Database. Preserve its same-VPS execution-node selection.
5. Import the updated dashboard JSONs and select the new UAV_Database datasource.
   If older UAV dashboards are already imported, back them up and deliberately
   update only those UAV dashboards; never repoint the shared cold-chain datasource.
6. Run Build Now and check new points using:

   ```bash
   curl -sS -G http://127.0.0.1:8086/query --data-urlencode 'db=UAV_Database' --data-urlencode 'q=SELECT * FROM "uav_suite_energy" WHERE time > now() - 5m LIMIT 3'
   ```

Any earlier UAV samples in another database are left intact; no data was moved
or deleted. Do not use old copies of the bundle or previous cold-chain commands.

## Offline checks and bounded preview

From this directory:

```text
python3 -B test_suite.py
python3 -B suite_emitter.py --summary
python3 -B suite_emitter.py
```

The default is dry-run with no network writes. Explicit opt-in for one batch:

```text
python3 -B suite_emitter.py --database UAV_Database --write
```

Do not leave the synthetic writer running indefinitely. At six batches per minute,
it generates about 2.54 million points daily. The new database currently has the
default autogen retention policy with unlimited duration. No retention policy was
changed. Disable only the new job when the preview is finished;
this stops new data and leaves stored samples intact.

Validation: all 43 distinct dashboard SELECT queries were accepted by the existing
InfluxDB server in read-only checks. This checks syntax, not populated chart rendering.
Browser rendering and scheduled delivery remain to be verified after authorized import.

References: [Grafana XY charts](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/xy-chart/),
[InfluxDB v1 write API](https://docs.influxdata.com/influxdb/v1/api/write/),
[Jenkins Pipeline syntax](https://www.jenkins.io/doc/book/pipeline/syntax/).
