import math
import numpy as np
from typing import Dict, List, Tuple

class SparseGaussianProcessWindMapper:
    def __init__(
        self,
        length_scale_m: float = 750.0,
        signal_variance_sigma_f: float = 2.8,
        noise_variance_sigma_n: float = 0.35,
        num_inducing_points: int = 16
    ):
        self.l = length_scale_m
        self.sigma_f2 = signal_variance_sigma_f ** 2
        self.sigma_n2 = noise_variance_sigma_n ** 2
        self.m_inducing = num_inducing_points

        # Inducing point locations X_u (M x 2)
        self.X_u: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_u_train: Optional[np.ndarray] = None # Wind U component
        self.y_v_train: Optional[np.ndarray] = None # Wind V component

    def _matern_32_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes Matérn 3/2 covariance matrix between point sets X1 and X2."""
        dist = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)
        sqrt3_d_over_l = (math.sqrt(3.0) * dist) / self.l
        k_mat = self.sigma_f2 * (1.0 + sqrt3_d_over_l) * np.exp(-sqrt3_d_over_l)
        return k_mat

    def fit_sparse_observations(
        self,
        observation_coords: np.ndarray, # Shape (N, 2): [x_m, y_m]
        wind_vectors_ms: np.ndarray     # Shape (N, 2): [u_ms, v_ms]
    ):
        """
        Initializes inducing points via k-means clustering and precomputes FITC inversion matrices.
        """
        self.X_train = observation_coords.copy()
        self.y_u_train = wind_vectors_ms[:, 0]
        self.y_v_train = wind_vectors_ms[:, 1]
        n_obs = len(observation_coords)

        # Select inducing points evenly distributed across observation domain
        idx_step = max(1, n_obs // self.m_inducing)
        self.X_u = self.X_train[::idx_step][:self.m_inducing].copy()
        m = len(self.X_u)

        # Kernel matrices
        K_uu = self._matern_32_kernel(self.X_u, self.X_u) + 1e-5 * np.eye(m)
        K_uf = self._matern_32_kernel(self.X_u, self.X_train)
        K_fu = K_uf.T

        # Diagonal matrix Lambda = diag(K_ff - Q_ff + sigma_n^2 * I)
        K_uu_inv = np.linalg.inv(K_uu)
        Q_ff_diag = np.sum(K_fu * (K_fu @ K_uu_inv), axis=1)
        Lambda_diag = self.sigma_f2 - Q_ff_diag + self.sigma_n2
        Lambda_inv_diag = 1.0 / np.maximum(Lambda_diag, 1e-4)

        # Common inversion term: Sigma_M = (K_uu + K_uf * Lambda^-1 * K_fu)^-1
        middle_mat = K_uu + (K_uf * Lambda_inv_diag) @ K_fu
        self.Sigma_M = np.linalg.inv(middle_mat)
        self.K_uu_inv = K_uu_inv

        # Precompute weights for U and V components: alpha = Sigma_M * K_uf * Lambda^-1 * y
        self.alpha_u = self.Sigma_M @ (K_uf @ (Lambda_inv_diag * self.y_u_train))
        self.alpha_v = self.Sigma_M @ (K_uf @ (Lambda_inv_diag * self.y_v_train))

    def predict_wind_at_corridor(
        self,
        query_coords: np.ndarray # Shape (P, 2)
    ) -> Dict:
        """
        Predicts mean wind vector [u, v] and epistemic standard deviation sigma at query coordinates.
        """
        K_star_u = self._matern_32_kernel(query_coords, self.X_u)
        
        # Predicted Means
        mu_u = K_star_u @ self.alpha_u
        mu_v = K_star_u @ self.alpha_v
        mu_speed = np.hypot(mu_u, mu_v)

        # Predicted Variances: Var = K_** - K_*u * (K_uu^-1 - Sigma_M) * K_u*
        k_star_star_diag = np.full(len(query_coords), self.sigma_f2)
        diff_mat = self.K_uu_inv - self.Sigma_M
        var_reduction = np.sum(K_star_u * (K_star_u @ diff_mat), axis=1)
        sigma_epistemic = np.sqrt(np.maximum(0.0, k_star_star_diag - var_reduction))

        return {
            "query_points_evaluated": len(query_coords),
            "mean_wind_u_ms": [round(float(u), 2) for u in mu_u],
            "mean_wind_v_ms": [round(float(v), 2) for v in mu_v],
            "mean_wind_speed_ms": [round(float(s), 2) for s in mu_speed],
            "epistemic_uncertainty_std_ms": [round(float(sig), 3) for sig in sigma_epistemic],
            "high_uncertainty_zone_flag": [bool(sig > 1.8) for sig in sigma_epistemic]
        }

if __name__ == "__main__":
    gp_mapper = SparseGaussianProcessWindMapper(length_scale_m=800.0, num_inducing_points=8)

    # Simulate 30 distributed UAV wind observations across a 3km x 3km urban sector
    np.random.seed(42)
    obs_x = np.random.uniform(0.0, 3000.0, (30, 2))
    # True underlying advective field: 4.5 m/s North-East + micro-canyon vortex
    u_true = 3.2 + 0.001 * obs_x[:, 1] + np.random.normal(0, 0.2, 30)
    v_true = 2.1 - 0.001 * obs_x[:, 0] + np.random.normal(0, 0.2, 30)
    obs_wind = np.column_stack([u_true, v_true])

    gp_mapper.fit_sparse_observations(obs_x, obs_wind)

    # Query flight path waypoints: from dense observation area into unvisited sector (x=4500m)
    test_wps = np.array([
        [1500.0, 1500.0], # Center of visited area (low uncertainty)
        [2800.0, 2800.0], # Boundary of sector
        [4200.0, 4200.0]  # Unvisited remote corridor (high epistemic uncertainty)
    ])

    predictions = gp_mapper.predict_wind_at_corridor(test_wps)

    print("==========================================================================")
    print("      SPARSE GAUSSIAN PROCESS SPATIAL WIND & UNCERTAINTY MAPPER AUDIT     ")
    print("==========================================================================")
    for i in range(len(test_wps)):
        print(f"Waypoint #{i+1} [X: {test_wps[i, 0]:4.0f}m, Y: {test_wps[i, 1]:4.0f}m] -> "
              f"Wind: [U: {predictions['mean_wind_u_ms'][i]:+4.2f}, V: {predictions['mean_wind_v_ms'][i]:+4.2f}] m/s "
              f"({predictions['mean_wind_speed_ms'][i]:4.2f} m/s) | Uncertainty (1-sigma): {predictions['epistemic_uncertainty_std_ms'][i]:4.2f} m/s | "
              f"Status: {'HIGH_UNCERTAINTY_EXPLORE' if predictions['high_uncertainty_zone_flag'][i] else 'CONFIDENT_CORRIDOR'}")
    print("==========================================================================\n")