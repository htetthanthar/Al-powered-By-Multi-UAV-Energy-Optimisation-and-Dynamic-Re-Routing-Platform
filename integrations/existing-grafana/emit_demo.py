"""Bounded, explicitly simulated UAV emitter for the existing InfluxDB v1 service."""
import argparse
import math
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def sample(timestamp):
    lines = []
    phase = (timestamp % 1800) / 1800
    for index, drone in enumerate(("UAV-ALPHA", "UAV-BRAVO")):
        wave = math.sin(timestamp / 30 + index)
        fields = {
            "battery_soc_pct": 95 - 60 * phase - index * 3,
            "power_watts": 280 + 20 * wave + index * 15,
            "ground_speed_kmh": 62 + 3 * wave,
            "wind_speed_kmh": 14 + 2 * math.sin(timestamp / 50),
            "altitude_m": 30 + index * 5 + wave,
        }
        values = ",".join(f"{key}={value:.3f}" for key, value in fields.items())
        lines.append(f"uav_demo_telemetry,drone_id={drone},source=simulation {values} {timestamp}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Actually send samples; otherwise print a preview only")
    parser.add_argument("--url", default="http://127.0.0.1:8086")
    parser.add_argument("--database", default="UAV_Database", choices=["UAV_Database"])
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.samples <= 6:
        parser.error("--samples must be 1..6 (a maximum of about 50 seconds)")
    for index in range(args.samples):
        payload = sample(int(time.time()))
        if args.write:
            endpoint = args.url.rstrip("/") + "/write?" + urlencode({"db": args.database, "precision": "s"})
            request = Request(endpoint, data=payload.encode(), headers={"Content-Type": "text/plain"}, method="POST")
            with urlopen(request, timeout=5) as response:
                if response.status != 204:
                    raise RuntimeError(f"InfluxDB write returned {response.status}")
            print("Wrote two SIMULATED UAV samples.", flush=True)
        else:
            print(payload)
        if index + 1 < args.samples:
            time.sleep(10)

if __name__ == "__main__":
    main()
