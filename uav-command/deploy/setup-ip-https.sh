#!/bin/bash
set -euo pipefail
cd /opt/stratum/app
test ! -e /etc/nginx/sites-available/stratum-ip
install -d -m 755 /var/www/stratum-acme
python3 -m venv /opt/stratum/acme
/opt/stratum/acme/bin/pip install 'certbot>=5.4,<6'
install -m 644 deploy/stratum-ip-http.conf /etc/nginx/sites-available/stratum-ip
ln -s /etc/nginx/sites-available/stratum-ip /etc/nginx/sites-enabled/stratum-ip
nginx -t
systemctl reload nginx
/opt/stratum/acme/bin/certbot certonly --non-interactive --agree-tos --register-unsafely-without-email --preferred-profile shortlived --webroot --webroot-path /var/www/stratum-acme --ip-address 64.118.128.181 --cert-name stratum-ip --config-dir /etc/letsencrypt-stratum --work-dir /var/lib/letsencrypt-stratum --logs-dir /var/log/letsencrypt-stratum
python3 deploy/create-login.py
install -d /etc/systemd/system/stratum.service.d
install -m 644 deploy/https-origin.conf /etc/systemd/system/stratum.service.d/https-origin.conf
install -m 644 deploy/stratum-ip-https.conf /etc/nginx/sites-available/stratum-ip
nginx -t
install -m 755 deploy/reload-nginx.sh /opt/stratum/reload-nginx.sh
install -m 644 deploy/stratum-cert-renew.service /etc/systemd/system/stratum-cert-renew.service
install -m 644 deploy/stratum-cert-renew.timer /etc/systemd/system/stratum-cert-renew.timer
systemctl daemon-reload
systemctl restart stratum
systemctl reload nginx
systemctl enable --now stratum-cert-renew.timer
systemctl is-active stratum nginx stratum-cert-renew.timer
