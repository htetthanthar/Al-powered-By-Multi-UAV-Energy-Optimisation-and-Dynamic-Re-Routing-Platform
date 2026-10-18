import math
import numpy as np
from typing import Dict, Tuple

class DrydenTurbulenceGenerator:
    def __init__(
        self,
        nominal_airspeed_ms: float = 18.05,
        reference_wind_speed_20ft_ms: float = 6.0, # W_20 in m/s (~21.6 km/h)
        dt_seconds: float = 0.05
    ):
        self.v_a = nominal_airspeed_ms
        self.w20 = reference_wind_speed_20ft_ms
        self.dt = dt_seconds

        # Discrete Filter States: [x_u, x_v1, x_v2, x_w1, x_w2, x_p]
        self.state_u = 0.0
        self.state_v = np.zeros(2)
        self.state_w = np.zeros(2)
        self.state_p = 0.0

    def compute_turbulence_scale_lengths(self, altitude_agl_m: float) -> Tuple[float, float, float, float]:
        """
        Computes low-altitude scale lengths (L_u, L_v, L_w) and turbulence intensities (sigma_u, sigma_w).
        Valid for altitudes 3m <= h <= 300m.
        """
        h = max(3.0, min(altitude_agl_m, 300.0))

        # Vertical turbulence intensity and scale length
        sigma_w = 0.1 * self.w20
        l_w = h

        # Horizontal parameters
        denom = (0.177 + 0.000823 * h) ** 1.2
        l_u = h / denom
        l_v = l_u

        sigma_u = sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        sigma_v = sigma_u

        return l_u, l_w, sigma_u, sigma_w

    def step_gust_sample(self, altitude_agl_m: float) -> Dict[str, float]:
        """
        Generates continuous Dryden gust velocities and body angular rate disturbances:
          u_g (longitudinal), v_g (lateral), w_g (vertical), p_g (roll rate).
        """
        l_u, l_w, sigma_u, sigma_w = self.compute_turbulence_scale_lengths(altitude_agl_m)

        # White noise drive signals N(0, 1)
        eta_u = np.random.normal(0, 1.0) / math.sqrt(self.dt)
        eta_v = np.random.normal(0, 1.0) / math.sqrt(self.dt)
        eta_w = np.random.normal(0, 1.0) / math.sqrt(self.dt)

        # 1. Longitudinal Gust Filter: G_u(s) = sigma_u * sqrt(2*Lu / (pi*V)) / (1 + (Lu/V)*s)
        t_u = l_u / self.v_a
        k_u = sigma_u * math.sqrt((2.0 * l_u) / (math.pi * self.v_a))
        # Exact Euler discretization: dx = (-1/T)*x + (K/T)*eta
        self.state_u += ((-1.0 / t_u) * self.state_u + (k_u / t_u) * eta_u) * self.dt
        u_gust = self.state_u

        # 2. Vertical Gust Filter: G_w(s) = sigma_w * sqrt(Lw/(pi*V)) * (1 + sqrt(3)*(Lw/V)*s) / (1 + (Lw/V)*s)^2
        t_w = l_w / self.v_a
        k_w = sigma_w * math.sqrt(l_w / (math.pi * self.v_a))
        # State space representation for 2nd order transfer function
        # dx1 = x2, dx2 = (-1/T^2)*x1 - (2/T)*x2 + eta
        dx1 = self.state_w[1]
        dx2 = (-1.0 / (t_w ** 2)) * self.state_w[0] - (2.0 / t_w) * self.state_w[1] + eta_w
        self.state_w[0] += dx1 * self.dt
        self.state_w[1] += dx2 * self.dt
        w_gust = k_w * (self.state_w[0] + math.sqrt(3.0) * t_w * self.state_w[1])

        # 3. Lateral Gust Filter (similar form to vertical)
        t_v = l_u / self.v_a
        k_v = sigma_u * math.sqrt(l_u / (math.pi * self.v_a))
        dx1_v = self.state_v[1]
        dx2_v = (-1.0 / (t_v ** 2)) * self.state_v[0] - (2.0 / t_v) * self.state_v[1] + eta_v
        self.state_v[0] += dx1_v * self.dt
        self.state_v[1] += dx2_v * self.dt
        v_gust = k_v * (self.state_v[0] + math.sqrt(3.0) * t_v * self.state_v[1])

        # 4. Roll Angular Rate Disturbance (p_gust):
        # p_g = (sigma_w / sqrt(L_w * b)) * (0.8 / sqrt(V)) * eta_p
        b_span = 0.65
        p_rate_std = (sigma_w / math.sqrt(l_w * b_span)) * (0.8 / math.sqrt(self.v_a))
        p_gust_rad_s = float(np.random.normal(0, p_rate_std))

        return {
            "altitude_agl_m": round(altitude_agl_m, 1),
            "u_gust_longitudinal_ms": round(u_gust, 2),
            "v_gust_lateral_ms": round(v_gust, 2),
            "w_gust_vertical_ms": round(w_gust, 2),
            "p_gust_roll_rate_rad_s": round(p_gust_rad_s, 3),
            "resultant_turbulent_airspeed_ms": round(math.sqrt((self.v_a + u_gust)**2 + v_gust**2 + w_gust**2), 2),
            "intensity_sigma_w_ms": round(sigma_w, 3)
        }

if __name__ == "__main__":
    generator = DrydenTurbulenceGenerator(nominal_airspeed_ms=18.05, reference_wind_speed_20ft_ms=6.0)

    print("==========================================================================")
    print("      MIL-F-8785C DRYDEN ATMOSPHERIC TURBULENCE SPECTRAL GUST AUDIT      ")
    print("==========================================================================")
    
    # Simulate 5-second gust flight profile at 25m cruising altitude
    for t_step in range(1, 11):
        g_sample = generator.step_gust_sample(altitude_agl_m=25.0)
        print(f"T+{(t_step*0.05):4.2f}s | u_gust: {g_sample['u_gust_longitudinal_ms']:+5.2f} m/s | "
              f"v_gust: {g_sample['v_gust_lateral_ms']:+5.2f} m/s | w_gust: {g_sample['w_gust_vertical_ms']:+5.2f} m/s | "
              f"p_rate: {math.degrees(g_sample['p_gust_roll_rate_rad_s']):+5.1f}°/s | Airspeed: {g_sample['resultant_turbulent_airspeed_ms']:5.2f} m/s")
    print("==========================================================================\n")