# STRATUM IP-only HTTPS

Deployed 2026-09-15 at https://64.118.128.181/ with a publicly trusted Let's Encrypt IP certificate and a separate dashboard login. No domain or SSH tunnel is needed.

## Get your login

Run from your own terminal (do not paste the resulting password into chat):

```powershell
ssh root@64.118.128.181 "cat /opt/stratum/private/login.txt"
```

The login username is `stratum`. Its random password is different from the SSH password and is stored only in a root-readable file on the VPS. Nginx stores its salted hash.

## Services and renewal

- New Nginx site: `/etc/nginx/sites-available/stratum-ip` (ports 80 and 443).
- Port 80 serves ACME challenges and redirects other requests to HTTPS.
- All application pages, APIs and live events on 443 require authentication.
- STRATUM remains bound to `127.0.0.1:4174`; the proxy preserves browser Origin checks.
- STRATUM-only override: `/etc/systemd/system/stratum.service.d/https-origin.conf`.
- Dedicated Certbot runtime: `/opt/stratum/acme` (5.8.0 at deployment).
- Certificate files: `/etc/letsencrypt-stratum/live/stratum-ip/`.
- `stratum-cert-renew.timer` checks every six hours with up to 30 minutes jitter. Successful renewal validates and gracefully reloads Nginx.
- UFW rules allow TCP 80/443 only to this VPS IPv4 address. Keep port 80 accessible for renewal.

```bash
systemctl list-timers stratum-cert-renew.timer
journalctl -u stratum-cert-renew.service --no-pager -n 30
python3 /opt/stratum/app/deploy/verify-https.py
/opt/stratum/acme/bin/certbot renew --dry-run --config-dir /etc/letsencrypt-stratum --work-dir /var/lib/letsencrypt-stratum --logs-dir /var/log/letsencrypt-stratum --deploy-hook /opt/stratum/reload-nginx.sh --run-deploy-hooks
```

The setup script is a **first-install script**, not an update script; it refuses an existing Nginx site. A failed certificate request leaves the HTTP challenge site in place for troubleshooting. Do not rerun the whole script on an installed server.

Existing Jenkins port 8090 configuration was preserved byte-for-byte. Grafana, Jenkins, InfluxDB and the cold-chain database were not reconfigured. STRATUM restarted once, resetting its in-memory simulation. No database persistence or real aircraft connection has been added by HTTPS setup; data remains clearly labeled simulation.

IP certificates are short-lived; see the [official Certbot IP certificate guidance](https://letsencrypt.org/2026/03/11/shorter-certs-certbot). Protect VPS access and monitor renewal failures. Shared Basic authentication is suitable for this protected prototype; separate operator accounts and audit logging are needed before multi-user production control.
