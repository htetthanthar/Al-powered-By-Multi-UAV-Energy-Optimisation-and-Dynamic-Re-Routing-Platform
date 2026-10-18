import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, List

class ActorCriticCruisePolicy(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 2):
        """
        State:  [v_air, ground_dist_to_goal, wind_head_tail, wind_vertical, battery_soc, battery_temp]
        Action: [target_airspeed_command_ms, pitch_trim_rad]
        """
        super(ActorCriticCruisePolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared Feature Extractor
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # Policy Head (Actor: Gaussian Mean and Log-Std)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Value Head (Critic: State-Value V(s))
        self.critic_value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        feat = self.shared_backbone(state)
        
        # Mean action with bounded ranges:
        # Airspeed centered at 18.05 m/s, pitch centered at 0.0 rad
        mu_raw = self.actor_mean(feat)
        mu_airspeed = 18.05 + 6.0 * torch.tanh(mu_raw[..., 0:1]) # [12.05, 24.05] m/s
        mu_pitch = 0.08 * torch.tanh(mu_raw[..., 1:2])          # [-0.08, +0.08] rad
        mu = torch.cat([mu_airspeed, mu_pitch], dim=-1)

        std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = Normal(mu, std)
        value = self.critic_value(feat)
        return dist, value

class PPOEnergyOptimalCruiseAgent:
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_lr: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.clip_eps = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.v_coef = value_coef
        self.ent_coef = entropy_coef

        self.policy = ActorCriticCruisePolicy(state_dim=6, action_dim=2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Draws action from current policy distribution."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).numpy(), float(log_prob.item()), float(value.item())

    def compute_energy_reward(
        self,
        airspeed_ms: float,
        wind_head_tail_ms: float,
        wind_vertical_ms: float,
        payload_kg: float = 1.5,
        battery_soc_pct: float = 80.0
    ) -> float:
        """
        Calculates specific range reward (distance covered per unit energy).
        v_ground = v_air + wind_tailwind
        P_aero = P_induced(v_air) + P_parasite(v_air) + P_profile
        """
        v_ground = max(0.5, airspeed_ms + wind_head_tail_ms)
        mass_tot = 4.5 + payload_kg

        # Aerodynamic power surrogate model
        rho = 1.225
        disk_area = 4.0 * math.pi * (0.1905 ** 2)
        weight_n = mass_tot * 9.80665

        # Glauert induced power
        p_induced = 1.15 * (weight_n ** 2) / (2.0 * rho * disk_area * max(airspeed_ms, 1.0))
        # Parasite drag power
        p_parasite = 0.5 * rho * (airspeed_ms ** 3) * 0.042
        # Blade profile power
        p_profile = 75.0 * (1.0 + 3.0 * ((airspeed_ms / 45.0) ** 2))
        # Vertical climb/descent power adjustment
        p_climb = weight_n * (-wind_vertical_ms * 0.5)

        total_power_w = p_induced + p_parasite + p_profile + p_climb + 25.0

        # Specific range: meters traveled per Joule (v_ground / Power)
        specific_range_m_per_j = v_ground / max(total_power_w, 50.0)

        # Reward formulation scaled to order unity
        reward = specific_range_m_per_j * 1000.0 # Reward in mm/J

        # Penalty for low battery SOC depletion stress
        if battery_soc_pct < 25.0:
            reward -= 2.5 * ((25.0 - battery_soc_pct) / 25.0)

        return float(reward)

    def optimize_policy_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """Executes one PPO mini-batch policy gradient update."""
        dist, values = self.policy(states)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        # Ratio r_t(theta)
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (MSE)
        critic_loss = F.mse_loss(values.squeeze(-1), returns)

        total_loss = actor_loss + self.v_coef * critic_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "actor_loss": round(float(actor_loss.item()), 4),
            "critic_loss": round(float(critic_loss.item()), 4),
            "entropy": round(float(entropy.item()), 4),
            "total_loss": round(float(total_loss.item()), 4)
        }

if __name__ == "__main__":
    agent = PPOEnergyOptimalCruiseAgent()

    print("==========================================================================")
    print("      PPO ENERGY-OPTIMAL AIRSPEED & CRUISE REGULATION AUDIT               ")
    print("==========================================================================")

    # 1. Action sampling under 5.0 m/s headwind vs 5.0 m/s tailwind
    # State: [v_air=18.0, dist_goal=5000m, w_headtail, w_vert=0.0, soc=75%, temp=32°C]
    s_headwind = np.array([18.05, 5000.0, -5.0, 0.0, 75.0, 32.0])
    s_tailwind = np.array([18.05, 5000.0, +5.0, 0.0, 75.0, 32.0])

    a_head, logp_h, _ = agent.select_action(s_headwind)
    a_tail, logp_t, _ = agent.select_action(s_tailwind)

    r_head = agent.compute_energy_reward(a_head[0], wind_head_tail_ms=-5.0, wind_vertical_ms=0.0)
    r_tail = agent.compute_energy_reward(a_tail[0], wind_head_tail_ms=+5.0, wind_vertical_ms=0.0)

    print("Policy Cruise Dispatch Evaluation:")
    print(f"  • Headwind (-5 m/s) -> Commanded Airspeed: {a_head[0]:5.2f} m/s | Pitch Trim: {math.degrees(a_head[1]):+4.2f}° | Specific Range Reward: {r_head:.3f} mm/J")
    print(f"  • Tailwind (+5 m/s) -> Commanded Airspeed: {a_tail[0]:5.2f} m/s | Pitch Trim: {math.degrees(a_tail[1]):+4.2f}° | Specific Range Reward: {r_tail:.3f} mm/J")

    # 2. Synthetic Mini-Batch Optimization Step
    dummy_states = torch.randn(32, 6)
    dummy_actions = torch.randn(32, 2)
    dummy_old_logp = torch.zeros(32)
    dummy_returns = torch.ones(32) * 2.5
    dummy_adv = torch.randn(32)

    loss_metrics = agent.optimize_policy_batch(dummy_states, dummy_actions, dummy_old_logp, dummy_returns, dummy_adv)
    print("\nPPO Policy Gradient Optimization Diagnostics:")
    for k, v in loss_metrics.items():
        print(f"  • {k:20s}: {v}")
    print("==========================================================================\n")