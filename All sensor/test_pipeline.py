import pytest

from dynamic_planner import (
    evaluate_route_feasibility,
    optimize_stop_sequence,
    process_order_batch,
)
from spatial_geometry import AirspaceGeometryResolver


@pytest.fixture
def central_depot():
    return {"name": "Central Depot", "lat": 1.3000, "lon": 103.8000}


def test_no_fly_zone_detection():
    resolver = AirspaceGeometryResolver()
    in_nfz, zone_name = resolver.check_trajectory_conflict(
        (103.89, 1.35), (103.92, 1.38)
    )
    assert in_nfz is True
    assert "Restricted" in zone_name


def test_safe_airspace_acceptance():
    resolver = AirspaceGeometryResolver()
    in_nfz, zone_name = resolver.check_trajectory_conflict(
        (103.75, 1.28), (103.80, 1.30)
    )
    assert in_nfz is False
    assert zone_name is None


def test_order_batching_respects_capacity():
    orders = [
        {"id": "ORD-01", "weight_kg": 1.2},
        {"id": "ORD-02", "weight_kg": 0.9},
    ]
    batches = process_order_batch(orders)
    assert [[item["id"] for item in batch] for batch in batches] == [["ORD-01"], ["ORD-02"]]


def test_order_batching_rejects_unsupported_payload():
    with pytest.raises(ValueError, match="unsupported payload"):
        process_order_batch([{"id": "ORD-OVERWEIGHT", "weight_kg": 2.5}])


def test_payload_capacity_enforcement(central_depot):
    stops = [
        {"id": "P1", "lat": 1.3100, "lon": 103.8100, "weight_kg": 1.2},
        {"id": "P2", "lat": 1.3200, "lon": 103.8200, "weight_kg": 0.9},
    ]
    result = evaluate_route_feasibility([central_depot, *stops, central_depot])
    assert result["feasible"] is False
    assert result["payload_valid"] is False


def test_battery_duration_cutoff(central_depot):
    stops = [
        {"id": "DIST-1", "lat": 1.8000, "lon": 104.2000, "weight_kg": 0.5},
        {"id": "DIST-2", "lat": 1.1000, "lon": 103.2000, "weight_kg": 0.5},
    ]
    result = evaluate_route_feasibility([central_depot, *stops, central_depot], wind_speed_ms=2.0)
    assert result["feasible"] is False
    assert result["duration_valid"] is False


def test_optimization_does_not_increase_energy(central_depot):
    stops = [
        {"id": "STOP-A", "lat": 1.3840, "lon": 103.7470, "weight_kg": 0.5},
        {"id": "STOP-B", "lat": 1.4304, "lon": 103.8354, "weight_kg": 0.5},
        {"id": "STOP-C", "lat": 1.3532, "lon": 103.9447, "weight_kg": 0.5},
    ]
    baseline = evaluate_route_feasibility([central_depot, *stops, central_depot], wind_speed_ms=3.0, wind_angle_rad=0.5)
    _, _, optimized_energy = optimize_stop_sequence(
        stops, central_depot, wind_speed_ms=3.0, wind_angle_rad=0.5
    )
    assert optimized_energy <= baseline["energy_consumed_wh"]
