import math
import numpy as np
from scipy.sparse import lil_matrix
from typing import Dict, List, Tuple

class UrbanWindFieldTomographySolver:
    def __init__(
        self,
        grid_bounds_x_m: Tuple[float, float] = (0.0, 5000.0),
        grid_bounds_y_m: Tuple[float, float] = (0.0, 5000.0),
        grid_resolution_m: float = 500.0,
        regularization_lambda: float = 0.15
    ):
        self.x_min, self.x_max = grid_bounds_x_m
        self.y_min, self.y_max = grid_bounds_y_m
        self.res = grid_resolution_m
        self.reg_lambda = regularization_lambda

        self.nx = int((self.x_max - self.x_min) / self.res) + 1
        self.ny = int((self.y_max - self.y_min) / self.res) + 1
        self.num_nodes = self.nx * self.ny

        self.laplacian = self._build_discrete_laplacian_2d()

    def _node_index(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def _build_discrete_laplacian_2d(self) -> np.ndarray:
        """
        Constructs the 2D finite-difference Laplace operator L (N x N) for spatial smoothness.
        """
        L = np.zeros((self.num_nodes, self.num_nodes))
        for iy in range(self.ny):
            for ix in range(self.nx):
                idx = self._node_index(ix, iy)
                neighbors = []
                if ix > 0: neighbors.append(self._node_index(ix - 1, iy))
                if ix < self.nx - 1: neighbors.append(self._node_index(ix + 1, iy))
                if iy > 0: neighbors.append(self._node_index(ix, iy - 1))
                if iy < self.ny - 1: neighbors.append(self._node_index(ix, iy + 1))

                L[idx, idx] = len(neighbors)
                for n_idx in neighbors:
                    L[idx, n_idx] = -1.0
        return L

    def reconstruct_wind_field(
        self,
        fleet_observations: List[Dict]
    ) -> Dict:
        """
        fleet_observations contains dicts with:
          - x_m, y_m: UAV spatial coordinate
          - wind_u_obs, wind_v_obs: In-flight EKF estimated wind vector components
        """
        m = len(fleet_observations)
        if m < 4:
            return {"status": "INSUFFICIENT_OBSERVATION_POINTS"}

        # Observation matrix G (m x N) using bilinear interpolation weights
        G = np.zeros((m, self.num_nodes))
        y_u = np.zeros(m)
        y_v = np.zeros(m)

        for obs_idx, obs in enumerate(fleet_observations):
            px, py = obs["x_m"], obs["y_m"]
            y_u[obs_idx] = obs["wind_u_obs"]
            y_v[obs_idx] = obs["wind_v_obs"]

            # Bilinear interpolation weights
            ix = int(np.clip((px - self.x_min) / self.res, 0, self.nx - 2))
            iy = int(np.clip((py - self.y_min) / self.res, 0, self.ny - 2))

            rx = (px - (self.x_min + ix * self.res)) / self.res
            ry = (py - (self.y_min + iy * self.res)) / self.res

            G[obs_idx, self._node_index(ix, iy)]         += (1.0 - rx) * (1.0 - ry)
            G[obs_idx, self._node_index(ix + 1, iy)]     += rx * (1.0 - ry)
            G[obs_idx, self._node_index(ix, iy + 1)]     += (1.0 - rx) * ry
            G[obs_idx, self._node_index(ix + 1, iy + 1)] += rx * ry

        # Solve Regularized Normal Equations for u and v components independently:
        # (G^T G + lambda * L^T L) * w = G^T * y
        a_matrix = G.T @ G + self.reg_lambda * (self.laplacian.T @ self.laplacian)
        w_u = np.linalg.solve(a_matrix, G.T @ y_u)
        w_v = np.linalg.solve(a_matrix, G.T @ y_v)

        # Compute reconstruction residual RMSE
        rmse_u = float(np.sqrt(np.mean((G @ w_u - y_u) ** 2)))
        rmse_v = float(np.sqrt(np.mean((G @ w_v - y_v) ** 2)))

        return {
            "status": "TOMOGRAPHY_RECONSTRUCTION_CONVERGED",
            "grid_nodes_count": self.num_nodes,
            "fleet_measurements_ingested": m,
            "reconstruction_rmse_u_ms": round(rmse_u, 3),
            "reconstruction_rmse_v_ms": round(rmse_v, 3),
            "mean_field_wind_speed_ms": round(float(np.mean(np.hypot(w_u, w_v))), 2),
            "max_field_wind_speed_ms": round(float(np.max(np.hypot(w_u, w_v))), 2),
            "wind_grid_shape": (self.ny, self.nx)
        }

if __name__ == "__main__":
    tomo = UrbanWindFieldTomographySolver(
        grid_bounds_x_m=(0.0, 3000.0), grid_bounds_y_m=(0.0, 3000.0), grid_resolution_m=500.0
    )

    # Simulate 8 fleet drones distributed across urban corridors reporting EKF wind vectors
    np.random.seed(42)
    obs_data = []
    for d_id in range(12):
        x = np.random.uniform(200.0, 2800.0)
        y = np.random.uniform(200.0, 2800.0)
        # Synthetic vortex wind field centered at (1500, 1500)
        dx, dy = x - 1500.0, y - 1500.0
        u_true = -0.002 * dy + 3.0 + np.random.normal(0, 0.2)
        v_true = 0.002 * dx - 1.0 + np.random.normal(0, 0.2)
        obs_data.append({"x_m": x, "y_m": y, "wind_u_obs": u_true, "wind_v_obs": v_true})

    recon = tomo.reconstruct_wind_field(obs_data)

    print("==========================================================================")
    print("      MULTI-UAV FLEET 2D WIND TOMOGRAPHY INVERSION AUDIT                  ")
    print("==========================================================================")
    for k, v in recon.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")