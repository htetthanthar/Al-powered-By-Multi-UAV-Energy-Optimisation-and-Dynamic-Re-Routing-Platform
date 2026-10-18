"""
================================================================================
AERO-AI // PRE-MLP BATTERY DRAIN VISUALIZATION ENGINE (FIXED)
Fixes KeyError: 'power_watts' by aligning with column name 'power_draw_watts'
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

csv_data = """route_id,uav_id,airspeed_mps,headwind_mps,ambient_temp_c,battery_soc_pct,payload_mass_kg,power_draw_watts,unoptimized_energy_wh_per_km
RT-01,UAV-ALPHA-01,22.5,4.5,34.5,88.0,2.4,1245.8,15.36
RT-01,UAV-ALPHA-01,20.0,6.2,35.0,82.0,2.4,1310.2,18.21
RT-02,UAV-BRAVO-02,18.0,-5.0,28.5,92.0,0.0,520.4,10.41
RT-02,UAV-BRAVO-02,15.0,-3.5,29.0,86.0,0.0,410.8,9.85
RT-03,UAV-CHARLIE-03,24.0,7.8,37.2,65.0,4.8,1395.0,20.94
RT-03,UAV-CHARLIE-03,21.0,5.1,36.5,59.0,4.8,1280.5,16.95
RT-04,UAV-DELTA-04,16.0,0.0,30.0,95.0,1.2,610.2,13.72
RT-04,UAV-DELTA-04,14.0,1.2,30.5,90.0,1.2,580.0,14.90
RT-05,UAV-ECHO-05,23.0,6.8,36.0,74.0,3.6,1360.5,18.45
RT-05,UAV-ECHO-05,19.0,4.0,34.5,68.0,3.6,1120.3,16.38
RT-06,UAV-FOXTROT-06,12.0,-6.5,27.5,98.0,0.0,350.1,10.51
RT-06,UAV-FOXTROT-06,10.0,-4.8,28.0,94.0,0.0,310.5,11.22
RT-07,UAV-GOLF-07,25.0,8.0,38.0,50.0,5.0,1400.0,21.50
RT-07,UAV-GOLF-07,22.0,5.5,37.0,44.0,5.0,1315.6,17.80
RT-08,UAV-HOTEL-08,17.0,2.1,31.0,80.0,2.0,780.4,15.30"""

df = pd.read_csv(io.StringIO(csv_data))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.1

fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#020617')
master_gs = gridspec.GridSpec(3, 1, height_ratios=[0.11, 0.445, 0.445], figure=fig, hspace=0.52)

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

header_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=master_gs[0], 
                                            width_ratios=[0.42, 0.145, 0.145, 0.145, 0.145], 
                                            wspace=0.08)

ax_title = fig.add_subplot(header_gs[0])
ax_title.set_facecolor(BG_PANEL)
ax_title.axis('off')
ax_title.text(0.0, 0.65, "PRE-MLP METEOROLOGICAL BATTERY DRAIN", 
              fontsize=14.0, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Unoptimized Specific Energy Consumption & Meteorological Impedance Prior to MLP Routing", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

kpis = [
    ("TOTAL SAMPLES", f"{len(df)} Flights", CYAN),
    ("MEAN POWER DRAW", f"{df['power_draw_watts'].mean():.1f} W", ROSE),
    ("MEAN ENERGY RATE", f"{df['unoptimized_energy_wh_per_km'].mean():.2f} Wh/km", AMBER),
    ("MAX WIND HEADWIND", f"{df['headwind_mps'].max():.1f} m/s", EMERALD),
]

for idx, (label, val, color) in enumerate(kpis, start=1):
    ax_kpi = fig.add_subplot(header_gs[idx])
    ax_kpi.set_facecolor(CARD_BG)
    ax_kpi.axis('off')
    
    bbox = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2,
                          transform=ax_kpi.transAxes, clip_on=False)
    ax_kpi.add_patch(bbox)
    
    ax_kpi.text(0.5, 0.60, val, fontsize=12.0, fontweight='bold', color=color,
                ha='center', va='center', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.24, label, fontsize=7.2, fontweight='bold', color=TEXT_MUTED,
                ha='center', va='center', transform=ax_kpi.transAxes)

row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], width_ratios=[0.56, 0.44], wspace=0.25)

ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)
ax1.scatter(df['airspeed_mps'], df['unoptimized_energy_wh_per_km'], c=df['payload_mass_kg'], cmap='viridis', s=60, edgecolors='none')
ax1.set_title("Unoptimized Energy Rate vs Airspeed", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax1.set_xlabel("Airspeed (m/s)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax1.set_ylabel("Specific Energy (Wh/km)", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax1.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)

ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)
ax2.scatter(df['headwind_mps'], df['power_draw_watts'], color=ROSE, s=55, alpha=0.85)
ax2.set_title("Power Draw vs Headwind Component", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax2.set_xlabel("Headwind (m/s)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax2.set_ylabel("Power Draw (W)", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax2.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)

row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.36)

ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)
ax3.scatter(df['ambient_temp_c'], df['power_draw_watts'], color=AMBER, s=45)
ax3.set_title("Thermal Impact on Power Draw", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax3.set_xlabel("Ambient Temperature (°C)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax3.set_ylabel("Power (W)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax3.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)
ax4.scatter(df['battery_soc_pct'], df['unoptimized_energy_wh_per_km'], color=CYAN, s=45)
ax4.set_title("Energy Drain across Battery SoC", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax4.set_xlabel("State of Charge (%)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax4.set_ylabel("Energy Rate (Wh/km)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax4.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)
ax5.bar(df['route_id'], df['power_draw_watts'], color=VIOLET, alpha=0.8, edgecolor=BORDER_COLOR)
ax5.set_title("Baseline Power Draw per Route", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax5.set_xlabel("Route ID", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax5.set_ylabel("Power (W)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax5.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

output_filename = "pre_mlp_battery_drain.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Generated slide: {output_filename}")