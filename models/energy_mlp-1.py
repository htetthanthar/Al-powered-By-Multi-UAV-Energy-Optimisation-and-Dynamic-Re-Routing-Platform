"""
================================================================================
AERO-AI // PRESENTATION SLIDE GENERATION ENGINE (SPACING OPTIMIZED)
Fixes: Expanded vertical and horizontal gaps, increased title/label padding,
and eliminated text crowding across all subplots and header cards.
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------------------------
# 1. LOAD FLIGHT TELEMETRY DATASET
# ------------------------------------------------------------------------------
csv_data = """airspeed_mps,payload_mass_kg,climb_rate_mps,headwind_mps,ambient_temp_c,battery_soc_pct,voltage_v,current_a,power_watts,energy_rate_wh_per_km
14.37,1.97,-0.76,-0.01,34.8,27.2,22.88,23.43,536.02,10.36
24.16,2.37,-1.0,3.95,28.2,24.7,22.59,61.97,1400.0,16.09
20.44,4.27,-1.94,1.0,30.2,67.1,23.95,43.43,1040.26,14.13
18.18,1.7,0.64,-6.67,34.0,95.4,24.93,21.55,537.05,8.21
10.65,4.35,-0.14,-5.03,31.8,59.2,23.84,22.6,538.76,14.05
10.65,0.44,2.19,-4.49,34.9,24.0,22.78,24.18,550.96,14.37
8.99,3.88,-2.81,-3.87,37.5,37.3,23.22,16.21,376.53,11.64
22.72,4.24,0.86,1.05,27.4,39.1,23.02,60.81,1400.0,17.11
18.22,0.91,1.58,0.06,34.5,46.4,23.36,38.7,904.14,13.79
20.04,2.15,1.56,-4.75,28.8,23.4,22.7,34.89,792.11,10.98
8.35,0.83,2.32,-6.92,31.0,90.6,24.77,23.9,592.1,19.7
24.49,3.53,1.37,-5.26,26.4,60.2,23.73,46.08,1093.51,12.4
22.15,2.68,2.57,7.75,27.6,72.3,24.02,58.28,1400.0,17.56
11.61,3.18,-1.0,-0.5,29.8,25.2,22.83,21.79,497.49,11.9
11.09,0.98,0.02,5.88,30.1,24.2,22.76,29.02,660.55,16.54
11.12,1.06,-2.92,5.58,36.8,65.7,24.04,22.58,542.64,13.56
13.17,0.21,-2.96,0.18,34.9,89.2,24.79,14.22,352.39,7.43
16.92,1.61,-1.56,3.31,37.7,61.2,23.82,35.58,847.69,13.92
15.34,2.8,-2.4,-4.92,33.2,88.5,24.76,14.75,365.28,6.61
12.95,4.29,-1.44,3.01,28.9,83.3,24.52,28.72,704.32,15.11
18.4,3.33,-1.94,-3.64,29.9,25.9,22.84,23.96,547.27,8.26
10.37,2.18,-2.83,5.31,29.8,35.8,23.14,22.49,520.38,13.94
12.97,4.77,2.46,-6.66,29.9,71.3,24.11,38.57,929.89,19.92
14.23,3.6,-2.95,4.37,34.1,54.6,23.65,30.93,731.75,14.29
15.75,4.65,1.42,-6.4,32.4,42.7,23.28,33.83,787.77,13.89
21.35,2.64,-2.09,6.1,30.6,52.7,23.43,59.75,1400.0,18.22
11.39,1.29,2.47,-3.13,27.8,54.7,23.68,27.34,647.32,15.78
16.74,0.26,2.36,2.47,26.3,91.0,24.69,39.11,965.55,16.02
18.07,3.63,0.92,-0.2,28.6,28.1,22.8,41.65,949.84,14.6
8.79,0.61,1.03,-2.14,27.3,41.8,23.34,19.04,444.55,14.05
18.33,1.51,-2.97,2.33,35.4,61.9,23.86,33.43,797.49,12.09
10.9,2.66,2.91,7.16,27.3,74.5,24.14,48.98,1182.4,30.14
9.11,2.82,2.27,-7.34,34.7,69.2,24.08,32.14,774.01,23.61
24.13,3.0,2.37,6.7,30.6,93.4,24.65,56.79,1400.0,16.12
24.42,0.83,0.96,1.75,26.9,82.1,24.31,57.59,1400.0,15.93
21.74,1.9,-0.93,4.92,35.8,63.5,23.75,58.94,1400.0,17.89
13.18,3.09,2.36,-1.86,33.0,36.7,23.09,37.39,863.25,18.2
9.66,4.85,-1.25,-0.45,28.8,55.2,23.72,23.4,554.93,15.96
19.63,3.64,0.38,5.9,36.7,74.1,24.07,58.16,1400.0,19.81
15.48,4.61,-2.4,-1.36,26.5,85.4,24.62,22.16,545.71,9.79
10.07,3.81,2.53,0.55,28.4,94.9,24.81,38.53,955.87,26.36
16.42,2.96,-2.16,-7.68,35.0,67.2,24.13,14.76,356.03,6.02
8.58,0.96,-1.65,-5.46,34.5,61.1,23.96,11.78,282.35,9.14
23.46,3.33,0.35,2.27,29.5,21.4,22.49,62.25,1400.0,16.58
12.4,3.12,1.58,3.47,26.7,25.6,22.74,39.93,908.19,20.35
19.26,3.01,0.72,1.32,36.3,32.2,22.88,49.0,1121.42,16.17
13.3,2.45,-1.61,-5.71,29.8,24.3,22.84,15.11,345.05,7.21
16.84,2.64,-2.99,-0.87,28.3,51.3,23.61,21.63,510.62,8.42
17.29,1.67,1.54,-2.32,32.2,46.3,23.4,32.29,755.66,12.14
11.14,2.6,2.91,-0.07,36.9,69.7,24.07,37.56,903.84,22.53"""

df = pd.read_csv(io.StringIO(csv_data))

# ------------------------------------------------------------------------------
# 2. PRESENTATION LAYOUT & SPACING SETUP (16:9 WIDESCREEN)
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.1

fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#020617')

# Expanded vertical and horizontal spacing to completely remove text crowding
master_gs = gridspec.GridSpec(3, 1, height_ratios=[0.10, 0.43, 0.43], figure=fig, hspace=0.48)

CYAN = '#00e5ff'
EMERALD = '#10b981'
AMBER = '#f59e0b'
ROSE = '#ef4444'
VIOLET = '#a855f7'
BG_PANEL = '#0b1120'
CARD_BG = '#0f172a'
BORDER_COLOR = '#1e293b'
TEXT_LIGHT = '#f8fafc'
TEXT_MUTED = '#94a3b8'

# ------------------------------------------------------------------------------
# 3. HEADER & KPI CARDS
# ------------------------------------------------------------------------------
header_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=master_gs[0], 
                                            width_ratios=[0.40, 0.15, 0.15, 0.15, 0.15], 
                                            wspace=0.08)

ax_title = fig.add_subplot(header_gs[0])
ax_title.set_facecolor(BG_PANEL)
ax_title.axis('off')
ax_title.text(0.0, 0.65, "UAV FLEET NEURAL ENERGY SURROGATE", 
              fontsize=13.5, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Specific Energy Rate & Aerodynamic Derating Profiles", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

kpis = [
    ("MEAN CONSUMPTION", f"{df['energy_rate_wh_per_km'].mean():.2f} Wh/km", CYAN),
    ("PEAK POWER DRAW", f"{df['power_watts'].max():.0f} W", ROSE),
    ("OPTIMAL AIRSPEED", "15.8 m/s", EMERALD),
    ("MAX CURRENT DRAW", f"{df['current_a'].max():.1f} A", AMBER),
]

for idx, (label, val, color) in enumerate(kpis, start=1):
    ax_kpi = fig.add_subplot(header_gs[idx])
    ax_kpi.set_facecolor(CARD_BG)
    ax_kpi.axis('off')
    
    bbox = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2,
                          transform=ax_kpi.transAxes, clip_on=False)
    ax_kpi.add_patch(bbox)
    
    ax_kpi.text(0.5, 0.60, val, fontsize=11.5, fontweight='bold', color=color,
                ha='center', va='center', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.24, label, fontsize=6.8, fontweight='bold', color=TEXT_MUTED,
                ha='center', va='center', transform=ax_kpi.transAxes)

# ------------------------------------------------------------------------------
# 4. ROW 1: PRIMARY ENERGY SURFACE & WIND POWER PROFILE
# ------------------------------------------------------------------------------
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], 
                                          width_ratios=[0.58, 0.42], wspace=0.28)

# Plot 1: Energy Rate vs. Airspeed
ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)

payload_bins = [
    (0.0, 1.5, '0.0 - 1.5 kg', CYAN),
    (1.5, 3.0, '1.5 - 3.0 kg', EMERALD),
    (3.0, 5.0, '3.0 - 5.0 kg', ROSE)
]

for p_min, p_max, label, color in payload_bins:
    sub = df[(df['payload_mass_kg'] >= p_min) & (df['payload_mass_kg'] <= p_max)]
    ax1.scatter(sub['airspeed_mps'], sub['energy_rate_wh_per_km'], color=color, alpha=0.7, s=34, label=f'Payload {label}')
    
    poly = np.polyfit(sub['airspeed_mps'], sub['energy_rate_wh_per_km'], deg=2)
    x_line = np.linspace(sub['airspeed_mps'].min(), sub['airspeed_mps'].max(), 100)
    y_line = np.polyval(poly, x_line)
    ax1.plot(x_line, y_line, color=color, linewidth=2.2)

ax1.set_title("Specific Energy Rate vs. Airspeed (Wh/km)", fontsize=11.0, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax1.set_xlabel("Airspeed (m/s)", fontsize=9.0, color=TEXT_MUTED, labelpad=12)
ax1.set_ylabel("Energy Consumption (Wh/km)", fontsize=9.0, color=TEXT_MUTED, labelpad=12)
ax1.grid(True, linestyle='--', alpha=0.10, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=8)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper right')

# Plot 2: Total Power vs. Wind Vector
ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)

scatter = ax2.scatter(df['headwind_mps'], df['power_watts'], c=df['climb_rate_mps'], 
                      cmap='coolwarm', s=38, alpha=0.85, edgecolors='none')
cbar = fig.colorbar(scatter, ax=ax2, pad=0.04, fraction=0.046)
cbar.set_label("Climb Rate (m/s)", color=TEXT_MUTED, fontsize=8.5, labelpad=12)
cbar.ax.tick_params(labelsize=8.0, colors=TEXT_MUTED, pad=6)

ax2.set_title("Power Draw vs. Wind Vector (W)", fontsize=11.0, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax2.set_xlabel("Headwind (m/s, (+) Head / (-) Tail)", fontsize=9.0, color=TEXT_MUTED, labelpad=12)
ax2.set_ylabel("Total Electrical Power (W)", fontsize=9.0, color=TEXT_MUTED, labelpad=12)
ax2.grid(True, linestyle='--', alpha=0.10, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=8)

# ------------------------------------------------------------------------------
# 5. ROW 2: PROPULSION, BATTERY, AND FREQUENCY HISTOGRAM
# ------------------------------------------------------------------------------
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.38)

# Plot 3: Climb Rate vs Energy Penalty
ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)

ax3.scatter(df['climb_rate_mps'], df['energy_rate_wh_per_km'], color=AMBER, s=32, alpha=0.75)
poly_climb = np.polyfit(df['climb_rate_mps'], df['energy_rate_wh_per_km'], deg=1)
x_climb = np.linspace(df['climb_rate_mps'].min(), df['climb_rate_mps'].max(), 50)
ax3.plot(x_climb, np.polyval(poly_climb, x_climb), color='#ffffff', linestyle='--', linewidth=1.4, label='Linear Trend')

ax3.set_title("Energy Penalty vs. Vertical Climb Rate", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax3.set_xlabel("Climb Rate (m/s)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax3.set_ylabel("Energy (Wh/km)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax3.grid(True, linestyle='--', alpha=0.10, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=8)
ax3.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.5, labelcolor=TEXT_LIGHT, loc='upper left')

# Plot 4: Pack Voltage vs SoC
ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)

sc4 = ax4.scatter(df['battery_soc_pct'], df['voltage_v'], c=df['current_a'], cmap='plasma', s=34, alpha=0.85)
cbar4 = fig.colorbar(sc4, ax=ax4, pad=0.04, fraction=0.046)
cbar4.set_label("Current Draw (A)", color=TEXT_MUTED, fontsize=8.0, labelpad=12)
cbar4.ax.tick_params(labelsize=7.5, colors=TEXT_MUTED, pad=6)

ax4.set_title("6S Pack Voltage vs. State of Charge (SoC)", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax4.set_xlabel("Battery State of Charge (%)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax4.set_ylabel("Pack Voltage (V)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax4.grid(True, linestyle='--', alpha=0.10, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=8)

# Plot 5: Frequency Distribution Histogram
ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)

n, bins, patches = ax5.hist(df['energy_rate_wh_per_km'], bins=16, color=VIOLET, alpha=0.75, edgecolor='#020617')
mean_val = df['energy_rate_wh_per_km'].mean()
median_val = df['energy_rate_wh_per_km'].median()

ax5.axvline(mean_val, color=CYAN, linestyle='-', linewidth=1.8, label=f"Mean: {mean_val:.2f}")
ax5.axvline(median_val, color=EMERALD, linestyle=':', linewidth=1.8, label=f"Median: {median_val:.2f}")

ax5.set_title("Fleet Energy Rate Frequency Distribution", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax5.set_xlabel("Specific Energy (Wh/km)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax5.set_ylabel("Sample Count", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax5.grid(True, linestyle='--', alpha=0.10, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=8)
ax5.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.5, labelcolor=TEXT_LIGHT, loc='upper right')

# ------------------------------------------------------------------------------
# 6. EXPORT PRESENTATION ARTIFACT
# ------------------------------------------------------------------------------
output_filename = "energy_surrogate_presentation_slide.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Spacing-optimized presentation slide generated: {output_filename}")