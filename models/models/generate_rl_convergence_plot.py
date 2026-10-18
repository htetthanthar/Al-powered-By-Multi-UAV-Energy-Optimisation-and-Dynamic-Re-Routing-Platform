import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------------------
# 1. RECONSTRUCTED CURVE DATA FROM FIG. 3
# ------------------------------------------------------------------------------
epochs = np.arange(1, 51)

# Reconstructed Policy Gradient Loss (smooth exponential decay with realistic stochastic variance)
loss_rl = np.array([
    11.95, 10.21, 8.42, 7.55, 6.68, 6.15, 5.48, 5.12, 4.91, 4.22,
    3.95, 3.71, 3.48, 3.22, 3.05, 2.89, 2.75, 2.32, 2.28, 2.22,
    2.27, 2.09, 2.19, 1.67, 1.82, 1.67, 1.91, 1.41, 1.38, 1.35,
    1.13, 1.31, 1.25, 1.39, 1.40, 1.61, 1.44, 1.21, 1.15, 1.35,
    1.07, 1.53, 1.38, 1.13, 0.98, 1.42, 1.19, 1.41, 0.98, 1.12
])

# Reconstructed Mean Route Distance / Tour Cost Convergence Metric
route_dist = np.array([
    60.0, 55.4, 51.8, 47.9, 45.2, 43.0, 40.8, 38.9, 37.5, 35.8,
    35.5, 33.5, 32.2, 32.1, 30.3, 29.9, 28.6, 28.5, 27.5, 26.2,
    26.1, 25.0, 25.2, 23.9, 23.8, 23.6, 22.1, 22.2, 22.1, 21.4,
    22.1, 22.0, 21.9, 21.9, 21.0, 21.0, 20.1, 20.3, 19.9, 20.5,
    20.2, 19.8, 19.1, 19.8, 19.2, 19.9, 19.7, 18.9, 18.9, 19.6
])

# ------------------------------------------------------------------------------
# 2. PLOTTING SETUP (16:9 PRESENTATION READY)
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300, facecolor='#ffffff')

# Left Plot: Policy Gradient Loss L_RL
ax1.plot(epochs, loss_rl, color='#0d9488', linewidth=2.5, label=r'$\mathcal{L}_{RL}$ Policy Loss')
ax1.set_title(r'Policy Gradient Loss $\mathcal{L}_{RL}$', fontsize=16, fontweight='bold', pad=12)
ax1.set_xlabel('Training Epochs', fontsize=14)
ax1.set_ylabel('Expected Advantage Cost', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
ax1.tick_params(labelsize=12)

# Right Plot: Mean Route Distance Convergence
ax2.plot(epochs, route_dist, color='#c2410c', linewidth=2.5, label='Mean Route Distance')
ax2.set_title('Mean Route Distance Convergence', fontsize=16, fontweight='bold', pad=12)
ax2.set_xlabel('Training Epochs', fontsize=14)
ax2.set_ylabel('Total Tour Cost Metric', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
ax2.tick_params(labelsize=12)

plt.tight_layout()
output_filename = 'fig3_rl_convergence_reproduced.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] Reproduced convergence plot saved as: {output_filename}")
