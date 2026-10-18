#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root so it can provision the existing Grafana instance." >&2
  exit 1
fi

PROJECT_DIR="${1:?Usage: sudo $0 /opt/uav-live-ops}"
SOURCE_DIR="${PROJECT_DIR}/Config/observability/grafana"
DATASOURCE_DIR="/etc/grafana/provisioning/datasources"
DASHBOARD_DIR="/etc/grafana/provisioning/dashboards/uav-operations"

install -d -m 0755 "${DATASOURCE_DIR}" "${DASHBOARD_DIR}"
install -m 0644 "${SOURCE_DIR}/vps-provisioning/datasources/uav-prometheus.yml" "${DATASOURCE_DIR}/uav-prometheus.yml"
install -m 0644 "${SOURCE_DIR}/vps-provisioning/dashboards/dashboards.yml" "/etc/grafana/provisioning/dashboards/uav-operations.yml"
install -m 0644 "${SOURCE_DIR}/provisioning/dashboards/uav-live-operations.json" "${DASHBOARD_DIR}/uav-live-operations.json"

systemctl restart grafana-server
echo "Grafana provisioning installed. Open the UAV Operations / UAV Live Operations dashboard."
