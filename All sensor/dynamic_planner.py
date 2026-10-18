# dynamic_planner.py
import math
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

try:
    if not _TORCH_AVAILABLE:
        raise ModuleNotFoundError("Torch is not installed")
    from models.energy_mlp import EnergySurrogateMLP
except ModuleNotFoundError:
    if _TORCH_AVAILABLE:
        class EnergySurrogateMLP(nn.Module):
            def __init__(self, in_features: int = 5, hidden_dim: int = 64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 32),
                    nn.Mish(),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                return self.net(x)
    else:
        class EnergySurrogateMLP:
            def __init__(self, in_features: int = 5, hidden_dim: int = 64):
                pass
            def eval(self):
                pass
            def __call__(self, x):
                return x


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def evaluate_route_feasibility(
    route: List[Dict],
    battery_capacity_wh: float = 488.4,
    max_duration_min: float = 55.0,
    max_payload_kg: float = 2.0,
    wind_speed_ms: float = 3.5,
    wind_angle_rad: float = 0.0,
    cruise_speed_ms: float = 18.05
) -> Dict[str, Any]:
    """Evaluates mission feasibility against dynamic aerodynamic and battery envelope constraints."""
    total_payload = sum(stop.get("weight_kg", 0.0) for stop in route)
    payload_valid = total_payload <= max_payload_kg

    total_dist_m = 0.0
    total_time_min = 0.0
    total_energy_wh = 0.0
    curr_payload = total_payload

    for i in range(len(route) - 1):
        p1, p2 = route[i], route[i+1]
        lat1, lon1 = p1.get("lat", 1.3), p1.get("lon", 103.8)
        lat2, lon2 = p2.get("lat", 1.3), p2.get("lon", 103.8)

        dx = (lon2 - lon1) * 111319.5 * math.cos(math.radians(lat1))
        dy = (lat2 - lat1) * 110574.0
        dist_m = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) if dist_m > 0 else 0.0

        v_rel = cruise_speed_ms - wind_speed_ms * math.cos(heading - wind_angle_rad)
        ground_speed = max(2.0, v_rel)
        flight_time_s = dist_m / ground_speed

        power_w = 220.0 + 4.2 * (ground_speed ** 1.5) + 38.0 * curr_payload + 12.0 * wind_speed_ms

        total_dist_m += dist_m
        total_time_min += flight_time_s / 60.0
        total_energy_wh += power_w * (flight_time_s / 3600.0)
        curr_payload -= p2.get("weight_kg", 0.0)

    duration_valid = total_time_min <= max_duration_min
    energy_valid = total_energy_wh <= (battery_capacity_wh * 0.80)
    feasible = payload_valid and duration_valid and energy_valid

    return {
        "feasible": feasible,
        "total_distance_km": round(total_dist_m / 1000.0, 2),
        "flight_time_min": round(total_time_min, 2),
        "energy_consumed_wh": round(total_energy_wh, 2),
        "battery_soc_remaining_pct": round(max(0.0, (1.0 - (total_energy_wh / battery_capacity_wh)) * 100.0), 1),
        "payload_valid": payload_valid,
        "duration_valid": duration_valid,
        "energy_valid": energy_valid
    }


def process_order_batch(orders: List[Dict], max_batch_size: int = 5) -> List[List[Dict]]:
    """Partitions incoming delivery orders into capacity-compliant batches."""
    batches = []
    curr_batch = []
    curr_weight = 0.0

    for order in orders:
        w = order.get("weight_kg", 0.5)
        if w <= 0.0 or w > 2.0:
            raise ValueError(f"Order {order.get('id', '<unknown>')} has an unsupported payload: {w} kg")
        if len(curr_batch) < max_batch_size and (curr_weight + w) <= 2.0:
            curr_batch.append(order)
            curr_weight += w
        else:
            if curr_batch:
                batches.append(curr_batch)
            curr_batch = [order]
            curr_weight = w
    if curr_batch:
        batches.append(curr_batch)
    return batches


def optimize_stop_sequence(
    orders: List[Dict],
    depot: Dict,
    model: Any = None,
    wind_speed_ms: float = 3.5,
    wind_angle_rad: float = 0.0,
    cruise_speed_ms: float = 18.05,
    **kwargs
) -> Tuple[List[Dict], float, float]:
    route = [depot] + list(orders) + [depot]
    n = len(route)
    if n <= 3:
        feas = evaluate_route_feasibility(route, wind_speed_ms=wind_speed_ms, wind_angle_rad=wind_angle_rad, cruise_speed_ms=cruise_speed_ms)
        return route, feas["flight_time_min"], feas["energy_consumed_wh"]

    def calc_metrics(tour):
        f = evaluate_route_feasibility(tour, wind_speed_ms=wind_speed_ms, wind_angle_rad=wind_angle_rad, cruise_speed_ms=cruise_speed_ms)
        return f["flight_time_min"], f["energy_consumed_wh"]

    best_time, best_energy = calc_metrics(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                t_time, t_energy = calc_metrics(new_route)
                if t_energy < best_energy and t_time <= 55.0:
                    route = new_route
                    best_time = t_time
                    best_energy = t_energy
                    improved = True
                    break
            if improved:
                break

    return route, best_time, best_energy
