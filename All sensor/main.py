import argparse
import sys
import os
import asyncio

def run_dispatch_cli(args):
    print("[*] Engaging Autonomous Flight Dispatch Pipeline...")
    from dynamic_planner import EnergySurrogateMLP, optimize_stop_sequence
    import torch

    depot = {"lat": 1.3000, "lon": 103.8000, "name": "Central Depot"}
    orders = [
        {"id": "ORD-01", "lat": 1.3840, "lon": 103.7470, "weight_kg": 0.65},
        {"id": "ORD-02", "lat": 1.4304, "lon": 103.8354, "weight_kg": 0.70},
        {"id": "ORD-03", "lat": 1.3532, "lon": 103.9447, "weight_kg": 0.50}
    ]

    model = EnergySurrogateMLP()
    model.eval()
    route, eta, energy = optimize_stop_sequence(orders, depot, model, wind_speed_ms=args.wind, wind_angle_rad=0.0)

    print("==========================================================================")
    print("                   DISPATCHED FLIGHT MANIFEST & TOUR                      ")
    print("==========================================================================")
    for idx, stop in enumerate(route):
        print(f"  Leg {idx:02d}: {stop.get('name', stop.get('id'))} (Payload: {stop.get('weight_kg', 0.0)} kg)")
    print("--------------------------------------------------------------------------")
    print(f"Projected Mission Duration : {eta:.2f} minutes (Envelop Limit: <= 55.0 min)")
    print(f"Projected Energy Draw      : {energy:.2f} Wh (Battery Capacity: 488.4 Wh)")
    print("==========================================================================")

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Micro-Climate UAV Route Optimization & Scheduling Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: Dispatch
    parser_dispatch = subparsers.add_parser("dispatch", help="Execute single-mission optimal dispatch")
    parser_dispatch.add_argument("--wind", type=float, default=3.5, help="Wind speed magnitude (m/s)")

    # Subcommand: Train
    parser_train = subparsers.add_parser("train", help="Train deep learning surrogate models")
    parser_train.add_argument("--target", choices=["energy", "weather", "router", "all"], default="all")

    # Subcommand: Benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run 100-trial statistical ablation study")

    # Subcommand: Verify
    parser_verify = subparsers.add_parser("verify", help="Run full-stack smoke & invariant verification")

    # Subcommand: Sim
    parser_sim = subparsers.add_parser("simulate", help="Run multi-UAV virtual airspace simulator")

    # Subcommand: Export
    parser_export = subparsers.add_parser("export-onnx", help="Export PyTorch models to FP16 ONNX runtime")

    args = parser.parse_args()

    if args.command == "dispatch":
        run_dispatch_cli(args)
    elif args.command == "train":
        if args.target in ["energy", "all"]:
            import train_energy
            train_energy.train_energy_surrogate()
        if args.target in ["weather", "all"]:
            import train_weather
            train_weather.train_weather_forecaster()
        if args.target in ["router", "all"]:
            import train_rl_router
            train_rl_router.train_attention_router()
    elif args.command == "benchmark":
        import ablation_benchmark
        ablation_benchmark.run_ablation_study()
    elif args.command == "verify":
        import full_stack_verification_audit
        full_stack_verification_audit.run_system_audit()
    elif args.command == "simulate":
        from simulate_fleet_mission import run_simulation_cluster
        asyncio.run(run_simulation_cluster())
    elif args.command == "export-onnx":
        import export_onnx_tensorrt
        export_onnx_tensorrt.export_energy_mlp()
        export_onnx_tensorrt.export_weather_lstm()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()