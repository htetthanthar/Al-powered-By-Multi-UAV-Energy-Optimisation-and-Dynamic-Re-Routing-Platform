import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dynamic_planner import EnergySurrogateMLP

class FederatedFleetTrainer:
    def __init__(
        self,
        num_drones: int = 4,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05
    ):
        self.num_drones = num_drones
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        
        # Central Global Model
        self.global_model = EnergySurrogateMLP()
        self.global_model.eval()

    def simulate_local_drone_training(
        self,
        drone_id: int,
        local_data_size: int = 250,
        local_epochs: int = 3,
        lr: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        """
        Simulates on-device local training on an individual drone's companion computer.
        """
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Generate synthetic localized telemetry with distinct motor friction biases
        np.random.seed(drone_id * 101)
        speed = np.random.uniform(12.0, 22.0, local_data_size)
        payload = np.random.uniform(0.0, 2.0, local_data_size)
        wind = np.random.uniform(0.0, 10.0, local_data_size)
        angle = np.random.uniform(0.0, 2 * np.pi, local_data_size)
        alt_delta = np.random.uniform(-5.0, 5.0, local_data_size)

        X = torch.tensor(np.column_stack([speed, payload, wind, angle, alt_delta]), dtype=torch.float32)
        # Power calculation with drone-specific hardware drift
        motor_degradation_offset = (drone_id * 8.5)
        Y = (180.0 + 4.5 * speed + 35.0 * payload + 12.0 * wind + motor_degradation_offset + np.random.normal(0, 3.0, local_data_size))
        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for _ in range(local_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(local_model(bx), by)
                loss.backward()
                optimizer.step()

        # Compute parameter update delta: Delta_W = W_local - W_global
        update_delta = {}
        global_state = self.global_model.state_dict()
        local_state = local_model.state_dict()

        for key in global_state:
            diff = local_state[key] - global_state[key]
            # Differential Privacy: Gradient Norm Clipping
            norm = torch.norm(diff)
            if norm > self.clip_norm:
                diff = diff * (self.clip_norm / (norm + 1e-6))
            
            # Differential Privacy: Gaussian Noise Addition
            noise = torch.randn_like(diff) * (self.noise_multiplier * self.clip_norm)
            update_delta[key] = diff + noise

        return update_delta

    def aggregate_fleet_updates(self, fleet_deltas: List[Dict[str, torch.Tensor]]):
        """
        Executes FedAvg parameter aggregation onto the central global surrogate model.
        """
        global_state = self.global_model.state_dict()
        num_clients = len(fleet_deltas)

        for key in global_state:
            # Average deltas across participating fleet edge nodes
            avg_delta = torch.stack([client_delta[key] for client_delta in fleet_deltas]).mean(dim=0)
            global_state[key] += avg_delta

        self.global_model.load_state_dict(global_state)

    def evaluate_global_model_loss(self) -> float:
        """Evaluates MSE loss of central model against a standardized validation set."""
        self.global_model.eval()
        np.random.seed(999)
        X_val = torch.tensor(np.random.uniform(0.0, 15.0, (200, 5)), dtype=torch.float32)
        Y_val = torch.tensor(np.random.uniform(200.0, 380.0, (200, 1)), dtype=torch.float32)
        
        with torch.no_grad():
            preds = self.global_model(X_val)
            loss = nn.MSELoss()(preds, Y_val).item()
        return loss

if __name__ == "__main__":
    federated_server = FederatedFleetTrainer(num_drones=4)
    print("==========================================================================")
    print("      FEDERATED FLEET LEARNING (FEDAVG + DP-SGD) ORCHESTRATION            ")
    print("==========================================================================")
    
    initial_loss = federated_server.evaluate_global_model_loss()
    print(f"Initial Central Model Validation Loss : {initial_loss:.4f}")

    # Execute 3 Federated Communication Rounds
    for round_num in range(1, 4):
        fleet_updates = []
        for d_id in range(federated_server.num_drones):
            delta = federated_server.simulate_local_drone_training(drone_id=d_id, local_epochs=2)
            fleet_updates.append(delta)

        federated_server.aggregate_fleet_updates(fleet_updates)
        round_loss = federated_server.evaluate_global_model_loss()
        print(f"  • Federated Round {round_num:02d} Completed | Aggregated {len(fleet_updates)} Drones | Loss: {round_loss:.4f}")

    print("==========================================================================\n")