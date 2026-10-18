# Live Grafana fleet dashboard

The dashboard is provisioned automatically by Docker Compose. It displays live
telemetry from the WebSocket broadcaster through Prometheus; no CSV import or
manual Grafana configuration is required.

## Start

For a local Grafana container, from the project root:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "choose-a-strong-password"
docker compose -f Config/docker-compose.yml --profile local-grafana up --build
```

Open `http://localhost:3000`, sign in as `admin` using the password above, and
open **UAV Operations / UAV Live Operations**. Grafana refreshes every five
seconds. Prometheus is available at `http://localhost:9090` for metric queries.

The telemetry endpoint is `http://localhost:8765/metrics`. Replace the
simulated telemetry producer with real flight messages while preserving the
published metric names to keep the dashboard compatible.

## Debian VPS with existing Grafana and Jenkins

Use the repository [Jenkinsfile](../Jenkinsfile) as a Pipeline-from-SCM job.
Create these Jenkins credentials before the first build:

- `uav-vps-ssh`: SSH private key for a non-root VPS deploy user in the `docker`
  group.
- `uav-vps-known-hosts`: secret file containing the VPS SSH host key. Do not use
  `StrictHostKeyChecking=no`.

Set the `VPS_HOST` build parameter to the VPS DNS name or address. The pipeline
uploads the project, builds it with the VPS override, starts the telemetry and
Prometheus services, provisions the existing Debian Grafana, then verifies the
live metric endpoint. Jenkins polls source control every five minutes; use your
Git provider webhook for immediate deployments.

The VPS must accept the SSH key that Jenkins uses. For a root deployment, add
Jenkins' **public** deploy key to `/root/.ssh/authorized_keys` over your VPS
provider console or an already-authenticated terminal. Do not share the root
password in chat or place it in a Jenkinsfile. The current workstation has no
SSH key loaded, so its connection attempt is correctly rejected until a key is
authorized.

For Grafana provisioning, permit the deploy user to run only the installer
without a password by adding this line through `visudo` (replace `jenkinsdeploy`
with the deploy user):

```sudoers
jenkinsdeploy ALL=(root) NOPASSWD: /bin/bash /opt/uav-live-ops/Config/vps/install-grafana-provisioning.sh *
```

All UAV service ports bind to `127.0.0.1` by default, including Prometheus.
Keep Grafana behind your existing TLS reverse proxy; Grafana reads Prometheus at
`http://127.0.0.1:9090` on the VPS. Set `UAV_BIND_ADDRESS=0.0.0.0` only when a
separate host genuinely needs direct access, and restrict it with a firewall.
