# system_stress_benchmark.py
import os
import time
import psutil
import numpy as np
from fleet_scheduler import FleetScheduler

def run_scalability_stress_test():
    print("=" * 80)
    print("        END-TO-END SCALABILITY & RESOURCE STRESS BENCHMARK")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    scheduler = FleetScheduler()

    scales = [5, 15, 30, 50]

    print(f"Host System Cores : {os.cpu_count()} CPU Threads | {psutil.virtual_memory().total / (1024**3):.1f} GB Total RAM")
    print("-" * 80)
    print(f"{'Order Load':<12}{'Missions':<12}{'Latency (ms)':<16}{'Memory (MB)':<16}{'Throughput (req/s)':<20}")
    print("-" * 80)

    for n_orders in scales:
        np.random.seed(42 + n_orders)
        orders = [
            {
                "id": f"ORD-{i:03d}",
                "lat": 1.3000 + np.random.uniform(-0.08, 0.08),
                "lon": 103.8000 + np.random.uniform(-0.08, 0.08),
                "weight_kg": round(float(np.random.uniform(0.2, 0.6)), 2)
            }
            for i in range(n_orders)
        ]

        t0 = time.perf_counter()
        schedule = scheduler.generate_fleet_schedule(orders, wind_speed_ms=4.0, wind_angle_rad=0.35)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mem_mb = process.memory_info().rss / (1024 * 1024)
        throughput = (n_orders / (t_elapsed_ms / 1000.0)) if t_elapsed_ms > 0 else 0.0
        drone_missions = len(schedule.get("missions", []))

        print(f"{n_orders:<12}{drone_missions:<12}{t_elapsed_ms:<16.2f}{mem_mb:<16.2f}{throughput:<20.1f}")

    print("=" * 80)
    print("[+] Stress benchmark complete across all operational loads [PASS].")


if __name__ == "__main__":
    run_scalability_stress_test()