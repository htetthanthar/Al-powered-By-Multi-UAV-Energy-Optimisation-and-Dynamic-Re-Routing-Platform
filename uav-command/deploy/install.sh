#!/usr/bin/env bash
# First installation only. Does not alter other services, proxies, or databases.
set -euo pipefail
test "$(id -u)" = 0
test "$(uname -m)" = x86_64
test ! -e /opt/stratum
test ! -e /etc/systemd/system/stratum.service
test -z "$(ss -H -ltn 'sport = :4174')"
install -d -m 0755 /opt/stratum/app /opt/stratum/runtime
tar -xzf "$1" -C /opt/stratum/app
cd /opt/stratum
# Obtain the current Node 24 LTS build from the official release directory,
# and verify the archive against the checksums served by that same origin.
curl --fail --silent --show-error --location --max-time 180 https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt -o SHASUMS256.txt
archive=$(awk '$2 ~ /^node-v24\.[0-9]+\.[0-9]+-linux-x64\.tar\.xz$/ {print $2}' SHASUMS256.txt)
test "$(printf '%s\n' "$archive" | wc -l)" = 1
test -n "$archive"
curl --fail --silent --show-error --location --max-time 300 "https://nodejs.org/dist/latest-v24.x/$archive" -o "$archive"
awk -v target="$archive" '$2 == target' SHASUMS256.txt | sha256sum --check -
tar -xJf "$archive" -C runtime --strip-components=1
export PATH="/opt/stratum/runtime/bin:$PATH"
cd /opt/stratum/app
npm ci --omit=dev --ignore-scripts --no-audit --no-fund
node --test tests/engine.test.mjs
install -m 0644 deploy/stratum.service /etc/systemd/system/stratum.service
systemctl daemon-reload
systemctl enable --now stratum.service
systemctl is-active stratum.service
echo 'Installed STRATUM; HTTP verification is still required.'
