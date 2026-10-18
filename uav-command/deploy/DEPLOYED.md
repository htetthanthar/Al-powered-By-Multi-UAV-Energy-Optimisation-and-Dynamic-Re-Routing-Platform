# VPS deployment record

STRATUM was deployed to 64.118.128.181 as its own systemd service.

- Application: /opt/stratum/app
- Application-local Node.js: /opt/stratum/runtime/bin/node (24.21.0)
- Unit: /etc/systemd/system/stratum.service
- Listener: 127.0.0.1:4174 (not public)
- DynamicUser, read-only filesystem sandbox, 384 MB memory limit, 75% CPU quota.
- Enabled at boot; restart on failure. No other service was restarted or reconfigured.
- All five unit/API tests passed on Debian. Page, JS, CSS, Cesium and SSE returned successfully.
- Grafana, Jenkins, InfluxDB and Nginx remained active. No firewall or proxy changes.

## Access from Windows

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:14174:127.0.0.1:4174 root@64.118.128.181
```

Keep this window open and browse http://127.0.0.1:14174. The service is running on
the VPS; the local port is only an encrypted forwarding connection. If that port
is already occupied by an existing tunnel, use it or select a different local port.
Do not expose port 4174 publicly. A public HTTPS domain with authentication needs
separate setup; this prototype does not implement user accounts or RBAC.

## Management on the VPS

```bash
systemctl status stratum --no-pager
journalctl -u stratum -n 50 --no-pager
systemctl restart stratum
```

Restarting clears virtual missions and in-memory history and retrains the small
synthetic models. Browser tabs need refreshing because the command token changes.
To stop and disable this service only:

```bash
systemctl disable --now stratum
```

Files are retained; no data is deleted by that command. The installer is for the
first install and intentionally refuses an existing /opt/stratum deployment.
Future upgrades should back up the current app, replace only its files, run tests,
then restart only stratum. Do not rerun the installer blindly.

## Scope

Still simulation only. No physical UAV connection and no database persistence.
UAV_Database remains the designated future database, but this service does not
write to it or to the cold-chain database. Jenkins scheduling is not needed to keep
this simulator alive: systemd manages the long-running process. Jenkins can later
run tests and deploy updates. Do not create a minute-by-minute simulator restart job.

Deployment retains the uploaded source archive in /tmp/stratum-deploy-20260915.tar.gz
and verified Node archive/checksums under /opt/stratum for provenance.
