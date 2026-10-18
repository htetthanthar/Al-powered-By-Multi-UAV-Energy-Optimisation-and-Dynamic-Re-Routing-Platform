# tests/test_end_to_end_pipeline.py
import pytest
import math
import torch
from dynamic_planner import optimize_stop_sequence
from formal_invariants_verifier import RouteSafetySMTVerifier

try:
    from models.energy_mlp import EnergySurrogateMLP
except ModuleNotFoundError:
    from dynamic_planner import EnergySurrogateMLP


def test_energy_surrogate_inference():
    model = EnergySurrogateMLP()
    model.eval()
    dummy_input = torch.tensor([[18.05, 1.2, 3.5, 0.0, 5.0]], dtype=torch.float32)
    with torch.no_grad():
        power_pred = model(dummy_input)
    assert power_pred is not None
    assert power_pred.shape[-1] == 1


def test_dynamic_vrp_route_optimization():
    depot = {"lat": 1.3000, "lon": 103.8000, "name": "Central Hub"}
    orders = [
        {"id": "ORD-01", "lat": 1.3500, "lon": 103.8200, "weight_kg": 0.8},
        {"id": "ORD-02", "lat": 1.3800, "lon": 103.7900, "weight_kg": 0.5}
    ]
    model = EnergySurrogateMLP()
    if hasattr(model, "eval"):
        model.eval()

    res = optimize_stop_sequence(
        orders=orders,
        depot=depot,
        model=model,
        wind_speed_ms=3.5,
        wind_angle_rad=0.0,
        cruise_speed_ms=18.05
    )
    route, eta, energy = res[0], res[1], res[2]

    assert len(route) == len(orders) + 2
    assert eta > 0.0
    assert energy > 0.0


def test_smt_safety_invariants_verification():
    verifier = RouteSafetySMTVerifier()
    orders = [{"id": "ORD-01", "weight_kg": 1.2}]
    
    valid_res = verifier.verify_mission_invariants(orders, total_distance_km=15.0)
    assert valid_res["mathematical_safety_clearance"] is True

    overload_orders = [{"id": "ORD-01", "weight_kg": 3.5}]
    invalid_res = verifier.verify_mission_invariants(overload_orders, total_distance_km=15.0)
    assert invalid_res["mathematical_safety_clearance"] is False