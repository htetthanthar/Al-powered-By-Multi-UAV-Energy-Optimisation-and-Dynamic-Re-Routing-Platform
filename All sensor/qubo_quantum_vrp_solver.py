import math
import numpy as np
from typing import Dict, List, Tuple

class QUBOQuantumVRPSolver:
    def __init__(
        self,
        num_customers: int = 5,
        penalty_unvisited: float = 500.0,
        penalty_capacity: float = 350.0,
        max_payload_kg: float = 2.0
    ):
        self.n = num_customers
        self.num_nodes = num_customers + 1  # Depot is node 0
        self.lambda_visit = penalty_unvisited
        self.lambda_cap = penalty_capacity
        self.q_max = max_payload_kg

    def build_qubo_matrix(
        self,
        cost_matrix: np.ndarray,
        demands_kg: np.ndarray
    ) -> np.ndarray:
        """
        Constructs upper-triangular QUBO matrix Q of size (N*T, N*T)
        where binary variable x_{i, t} indicates visiting node i at time-step t.
        """
        N = self.num_nodes
        T = self.num_nodes # T time steps
        dim = N * T
        Q = np.zeros((dim, dim))

        def var_idx(node, step):
            return node * T + step

        # 1. Routing Cost Objective: sum_{i, j, t} C_{ij} * x_{i, t} * x_{j, t+1}
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    u = var_idx(i, t)
                    v = var_idx(j, t + 1)
                    idx1, idx2 = min(u, v), max(u, v)
                    Q[idx1, idx2] += cost_matrix[i, j]

        # 2. Constraint: Every customer node visited exactly once
        # lambda_1 * sum_{i=1}^n (1 - sum_t x_{i,t})^2 = lambda_1 * sum_i [ 1 - 2*sum_t x_{i,t} + sum_{t1, t2} x_{i,t1}*x_{i,t2} ]
        for i in range(1, N):
            for t1 in range(T):
                u = var_idx(i, t1)
                Q[u, u] -= 2.0 * self.lambda_visit
                for t2 in range(t1 + 1, T):
                    v = var_idx(i, t2)
                    Q[u, v] += 2.0 * self.lambda_visit

        # 3. Constraint: Exactly one node visited per time step t
        # lambda_2 * sum_t (1 - sum_i x_{i,t})^2
        for t in range(T):
            for i1 in range(N):
                u = var_idx(i1, t)
                Q[u, u] -= 2.0 * self.lambda_visit
                for i2 in range(i1 + 1, N):
                    v = var_idx(i2, t)
                    Q[u, v] += 2.0 * self.lambda_visit

        # 4. Depot Departure and Return Fixation
        # Force x_{0, 0} = 1 and x_{0, T-1} = 1
        u_start = var_idx(0, 0)
        u_end = var_idx(0, T - 1)
        Q[u_start, u_start] -= 1000.0
        Q[u_end, u_end] -= 1000.0

        return Q

    def solve_simulated_bifurcation_annealing(
        self,
        Q: np.ndarray,
        num_sweeps: int = 1500,
        beta_init: float = 0.01,
        beta_final: float = 10.0
    ) -> Tuple[np.ndarray, float]:
        """
        Executes Simulated Annealing on the QUBO objective: E(x) = x^T * Q * x.
        """
        dim = Q.shape[0]
        state = np.random.randint(0, 2, dim)
        
        # Initialize depot variables
        state[0] = 1 # Depot start
        state[dim - 1] = 1 # Depot end

        current_energy = float(state.T @ Q @ state)
        best_state = state.copy()
        best_energy = current_energy

        betas = np.linspace(beta_init, beta_final, num_sweeps)

        for sweep, beta in enumerate(betas):
            for i in range(dim):
                # Propose bit flip
                state[i] = 1 - state[i]
                new_energy = float(state.T @ Q @ state)
                delta_e = new_energy - current_energy

                # Metropolis acceptance criterion
                if delta_e < 0.0 or np.random.rand() < math.exp(-beta * delta_e):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state.copy()
                else:
                    # Revert flip
                    state[i] = 1 - state[i]

        return best_state, best_energy

    def decode_tour_permutation(self, binary_state: np.ndarray) -> List[int]:
        """Extracts ordered node sequence from binary QUBO state vector."""
        N = self.num_nodes
        T = self.num_nodes
        tour = []

        for t in range(T):
            active_node = 0
            for i in range(N):
                if binary_state[i * T + t] == 1:
                    active_node = i
                    break
            tour.append(active_node)

        return tour

if __name__ == "__main__":
    qubo_solver = QUBOQuantumVRPSolver(num_customers=4)

    # Coordinates: Depot (0) + 4 Customer nodes
    coords = np.array([
        [1.3000, 103.8000],  # Depot
        [1.3500, 103.8200],  # C1
        [1.3800, 103.7700],  # C2
        [1.4100, 103.8400],  # C3
        [1.3400, 103.8800]   # C4
    ])
    
    # Pairwise distance matrix with synthetic wind penalty
    N = len(coords)
    cost_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.linalg.norm(coords[i] - coords[j]) * 111.0
                cost_mat[i, j] = d * 1.15  # Energy equivalent cost

    demands = np.array([0.0, 0.45, 0.60, 0.50, 0.40])

    Q = qubo_solver.build_qubo_matrix(cost_mat, demands)
    opt_state, min_energy = qubo_solver.solve_simulated_bifurcation_annealing(Q)
    tour = qubo_solver.decode_tour_permutation(opt_state)

    print("==========================================================================")
    print("      QUBO / QUANTUM ISING HAMILTONIAN VRP COMBINATORIAL SOLVER          ")
    print("==========================================================================")
    print(f"QUBO Hamiltonian Dimensions : {Q.shape[0]} x {Q.shape[1]} Binary Variables")
    print(f"Ground State Energy E_min   : {min_energy:.2f}")
    print(f"Decoded Optimal Node Tour   : {' -> '.join([f'Node_{idx}' for idx in tour])}")
    print("==========================================================================\n")