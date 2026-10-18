"""Verify Prometheus ingested fresh telemetry, with bounded startup retries."""
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen


def verify(base_url="http://127.0.0.1:9090", timeout_seconds=90):
    deadline = time.monotonic() + timeout_seconds
    last_error = "No scrape received"
    query = ('uav_battery_soc_percent{job="uav-telemetry"} '
             'and (time() - timestamp(uav_battery_soc_percent{job="uav-telemetry"}) < 30) '
             'and on(job, instance) (up{job="uav-telemetry"} == 1)')
    while time.monotonic() < deadline:
        try:
            url = base_url + "/api/v1/query?" + urlencode({"query": query})
            with urlopen(url, timeout=5) as response:
                result = json.load(response)
            if result.get("status") == "success" and result["data"]["result"]:
                print("Prometheus is scraping fresh UAV telemetry. Current producer: simulation.")
                return
            last_error = "Prometheus has no fresh UAV battery series from an available target"
        except (OSError, ValueError, KeyError) as exc:
            last_error = str(exc)
        time.sleep(3)
    raise SystemExit("Live metrics verification failed: " + last_error)


if __name__ == "__main__":
    verify()
