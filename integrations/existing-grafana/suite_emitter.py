"""Synthetic dashboard fixture, not trained-model output or physical flight data.

Default invocation previews one batch. --write explicitly appends to the existing
UAV_Database database. No schema, service, or datasource settings are modified.
"""
import argparse
import json
import math
import random
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def batch(timestamp):
    rng = random.Random(timestamp // 10)
    points = []

    def add(measurement, tags, fields):
        points.append((measurement, {"source": "simulation", **tags}, fields, timestamp))

    actual, predicted = [], []
    for i in range(15):
        route = f"RT-{i + 1:02d}"
        speed = 10 + i + rng.uniform(-0.25, 0.25)
        wind = -6 + i + rng.uniform(-0.3, 0.3)
        temp = 27 + i * 0.75 + rng.uniform(-0.1, 0.1)
        soc = 98 - i * 3.25
        payload = 0.3 + (i % 3) * 0.7
        climb = rng.uniform(-3, 3)
        power = 250 + 30 * payload + 1.15 * speed ** 2 + max(0, wind) * 30
        energy = power / (3.6 * max(2, speed - wind))
        # Illustrative counterfactual only: this multiplier is NOT a measured MLP improvement.
        reduction = 0.15 + rng.random() * 0.15
        common = {"airspeed_ms": speed, "headwind_ms": wind, "temperature_c": temp,
                  "battery_soc_pct": soc, "payload_kg": payload, "climb_ms": climb,
                  "voltage_v": 21.6 + 3.6 * soc / 100}
        for mode, factor in (("baseline", 1.0), ("optimized", 1.0 - reduction)):
            add("uav_suite_energy", {"route_id": route, "mode": mode}, {
                **common, "power_w": power * factor, "energy_whkm": energy * factor,
                "current_a": power * factor / common["voltage_v"],
                "savings_pct": (1 - factor) * 100})
        for j in range(4):
            node = i * 4 + j
            lon = 103.70 + rng.random() * 0.31
            lat = 1.28 + rng.random() * 0.16
            zones = [(103.82, 1.355, 1200), (103.90, 1.37, 1500), (104.0, 1.365, 1800)]
            clearance = min(math.hypot((lon - x) * 111000, (lat - y) * 111000) - radius
                            for x, y, radius in zones)
            add("uav_suite_routing", {"node_id": f"N-{node:02d}", "route_id": route}, {
                "lon": lon, "lat": lat, "clearance_m": clearance,
                "breach": float(clearance < 500), "gust_ms": rng.uniform(2, 8),
                "edge_cost": 8 + rng.random() * 12 + max(0, 500 - clearance) / 100,
                "wind_u_ms": rng.uniform(-6, 6), "wind_v_ms": rng.uniform(-6, 6)})
        for k in range(10):
            idx = i * 10 + k
            truth = 180 + idx * 1.6
            estimate = truth + rng.gauss(0, 6)
            actual.append(truth)
            predicted.append(estimate)
            add("uav_suite_fit", {"sample_id": f"S-{idx:03d}"}, {
                "physical_w": truth, "predicted_w": estimate, "ideal_w": truth,
                "residual_w": estimate - truth})
    mean = sum(actual) / len(actual)
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    spread = sum((a - mean) ** 2 for a in actual)
    add("uav_suite_accuracy", {}, {"r_squared": 1 - residual / spread,
        "rmse_w": math.sqrt(residual / len(actual)),
        "mae_w": sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual),
        "sample_count": float(len(actual))})
    # Entire curve is a synthetic fixture regenerated as a snapshot, not a training run.
    for epoch in range(1, 51):
        add("uav_suite_training", {"epoch_id": f"E-{epoch:02d}"}, {
            "epoch": float(epoch), "loss": 1 + 12 * math.exp(-epoch / 8) + rng.random() * 0.3,
            "tour_cost": 19 + 45 * math.exp(-epoch / 11) + rng.random()})
    for label, center in (("Greedy", 58), ("TwoOpt", 49), ("Hybrid", 42)):
        durations = [center + rng.gauss(0, 9) for _ in range(100)]
        add("uav_suite_comparison", {"framework": label}, {
            "mean_eta_min": sum(durations) / len(durations),
            "violations_pct": 100 * sum(d > 55 for d in durations) / len(durations),
            "sample_count": float(len(durations)), "limit_min": 55.0})
    return points


def line_protocol(points):
    lines = []
    for name, tags, fields, timestamp in points:
        # All identifiers here are fixed fixture identifiers (not untrusted input).
        tag_text = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        field_text = ",".join(f"{k}={float(v):.6f}" for k, v in fields.items())
        lines.append(f"{name},{tag_text} {field_text} {timestamp}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8086")
    parser.add_argument("--database", default="UAV_Database", choices=["UAV_Database"])
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batches <= 6:
        parser.error("--batches must be 1..6")
    for i in range(args.batches):
        points = batch(int(time.time()))
        if args.write:
            url = args.url.rstrip("/") + "/write?" + urlencode({"db": args.database, "precision": "s"})
            with urlopen(Request(url, data=line_protocol(points).encode(), method="POST",
                                 headers={"Content-Type": "text/plain"}), timeout=5) as response:
                if response.status != 204:
                    raise RuntimeError(f"Write failed: HTTP {response.status}")
            print(f"Wrote {len(points)} SIMULATED dashboard points", flush=True)
        elif args.summary:
            counts = {}
            for name, *_ in points:
                counts[name] = counts.get(name, 0) + 1
            print(json.dumps(counts, indent=2))
        else:
            print(line_protocol(points))
        if i + 1 < args.batches:
            time.sleep(10)


if __name__ == "__main__":
    main()
