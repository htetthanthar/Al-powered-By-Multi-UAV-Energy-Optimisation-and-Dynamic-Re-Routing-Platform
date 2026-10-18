# UAV dashboards with a dedicated database

Use **UAV_Database** for this project. Do not use or repoint the existing
cold-chain database or Grafana datasource. The dedicated database was created
and verified on 64.118.128.181. Grafana datasource/dashboard import
and Jenkins scheduling still need authorized application access.

See [SUITE.md](SUITE.md) for the current setup instructions. This supersedes
earlier instructions that reused the cold-chain database.

## Main seven-dashboard suite

- suite_emitter.py: bounded simulated samples, default UAV_Database.
- Jenkinsfile.suite: explicit --database UAV_Database; run on the InfluxDB VPS.
- dashboards/: seven JSON files that prompt for the dedicated datasource on import.
- test_suite.py: offline contract, data-integrity, and database-target tests.

Create a NEW Grafana InfluxDB datasource named UAV_Database, with InfluxQL,
URL http://127.0.0.1:8086 and database UAV_Database. Do not make it the default.
Select it when importing the JSON files. No hardcoded cold-chain UID remains.

The older two-drone demo (emit_demo.py, Jenkinsfile.demo and
uav-existing-influx-dashboard.json) has also been updated to the dedicated
database and import-time datasource selection. Use one demo pipeline, not both.

Both writers reject other database names. Neither creates databases, changes
authentication, or edits Grafana/Jenkins configuration. Dry-run is the default;
--write explicitly appends synthetic data to UAV_Database.

No physical drone or trained model is connected. All supplied data is simulated.
Publishing these dashboards does not turn synthetic data into real telemetry.
Do not run older Docker or provisioning installers for this setup.
