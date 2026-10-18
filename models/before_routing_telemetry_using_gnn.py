"""
================================================================================
AERO-AI // METEOROLOGICAL ROUTING PRE-GNN VISUALIZATION ENGINE (CLEAN PADDING)
Fixes:
- Displaced NFZ zone labels to outer perimeters to avoid breach marker overlap
- Expanded vertical row clearance (hspace=0.52) to eliminate inter-plot label collisions
- Increased labelpad & title pad across all axes for crisp visual separation
- Added dedicated margin buffers around colorbars and rotated categorical ticks
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch

# ------------------------------------------------------------------------------
# 1. RAW METEOROLOGICAL ROUTING DATASET (PRE-GNN BASELINE)
# ------------------------------------------------------------------------------
csv_raw = """route_id,uav_id,waypoint_idx,latitude,longitude,alt_m,wind_u_mps,wind_v_mps,wind_gust_mps,rain_intensity_mm_hr,temp_c,nearest_nfz_id,dist_to_nfz_m,nfz_breach_pre_gnn,baseline_edge_cost
RT-01,UAV-ALPHA-01,0,1.3400,103.7050,48.2,3.8,-2.6,6.8,18.4,30.5,NFZ-A,12850.4,0,12.4
RT-01,UAV-ALPHA-01,1,1.3520,103.7650,50.0,4.2,-1.9,7.4,12.2,31.2,NFZ-A,6120.8,0,18.9
RT-01,UAV-ALPHA-01,2,1.3530,103.8180,48.5,2.1,0.8,3.5,0.0,32.4,NFZ-A,220.5,1,94.2
RT-01,UAV-ALPHA-01,3,1.3540,103.8210,48.0,1.8,1.2,3.1,0.0,32.6,NFZ-A,-110.0,1,120.0
RT-01,UAV-ALPHA-01,4,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,24.1
RT-02,UAV-BRAVO-02,0,1.3150,103.6850,55.0,5.1,-3.5,8.2,26.5,29.8,NFZ-A,15640.2,0,14.6
RT-02,UAV-BRAVO-02,1,1.3320,103.7400,55.0,4.6,-2.8,7.0,15.8,30.4,NFZ-A,9120.0,0,19.2
RT-02,UAV-BRAVO-02,2,1.3480,103.8120,52.0,2.4,-0.5,4.0,0.0,32.1,NFZ-A,410.6,1,88.4
RT-02,UAV-BRAVO-02,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,22.8
RT-03,UAV-CHARLIE-03,0,1.3700,103.7250,42.6,3.2,-1.4,5.1,6.4,31.0,NFZ-A,10780.5,0,11.8
RT-03,UAV-CHARLIE-03,1,1.3650,103.7750,45.0,2.9,-0.8,4.6,2.1,31.8,NFZ-A,5140.2,0,14.2
RT-03,UAV-CHARLIE-03,2,1.3580,103.8150,44.0,1.9,0.4,3.2,0.0,32.3,NFZ-A,320.0,1,91.0
RT-03,UAV-CHARLIE-03,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,25.4
RT-04,UAV-DELTA-04,0,1.3500,103.7600,38.0,4.0,-2.1,6.3,8.0,31.4,NFZ-A,6850.1,0,10.2
RT-04,UAV-DELTA-04,1,1.3530,103.8190,40.0,2.0,0.6,3.4,0.0,32.5,NFZ-A,110.4,1,98.5
RT-04,UAV-DELTA-04,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,21.9
RT-05,UAV-ECHO-05,0,1.4350,103.7850,60.0,1.5,1.2,2.8,0.0,32.1,NFZ-A,9820.6,0,16.5
RT-05,UAV-ECHO-05,1,1.3900,103.7950,58.0,2.1,0.8,3.4,0.0,32.3,NFZ-A,4650.0,0,18.1
RT-05,UAV-ECHO-05,2,1.3560,103.8180,55.0,2.0,0.5,3.1,0.0,32.5,NFZ-A,380.2,1,86.2
RT-05,UAV-ECHO-05,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,23.0
RT-06,UAV-FOXTROT-06,0,1.4250,103.8300,50.5,1.8,0.9,3.0,0.0,32.5,NFZ-A,8120.3,0,15.4
RT-06,UAV-FOXTROT-06,1,1.3800,103.8250,48.0,2.0,0.4,3.2,0.0,32.6,NFZ-A,3120.0,0,16.8
RT-06,UAV-FOXTROT-06,2,1.3550,103.8220,46.0,1.9,0.6,3.0,0.0,32.5,NFZ-A,210.0,1,93.5
RT-06,UAV-FOXTROT-06,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,22.1
RT-07,UAV-GOLF-07,0,1.4100,103.8100,58.0,1.9,0.7,3.1,0.0,32.2,NFZ-A,6420.7,0,14.0
RT-07,UAV-GOLF-07,1,1.3650,103.8150,52.0,2.0,0.5,3.3,0.0,32.4,NFZ-A,1380.4,0,21.5
RT-07,UAV-GOLF-07,2,1.3520,103.8210,50.0,1.8,0.7,3.0,0.0,32.6,NFZ-A,-80.0,1,120.0
RT-07,UAV-GOLF-07,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,24.0
RT-08,UAV-HOTEL-08,0,1.4450,103.8150,44.0,1.6,1.0,2.9,0.0,32.0,NFZ-A,10320.1,0,17.2
RT-08,UAV-HOTEL-08,1,1.3850,103.8120,42.0,2.1,0.6,3.4,0.0,32.3,NFZ-A,3640.8,0,18.9
RT-08,UAV-HOTEL-08,2,1.3540,103.8190,40.0,1.9,0.5,3.1,0.0,32.5,NFZ-A,120.0,1,97.0
RT-08,UAV-HOTEL-08,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,23.5
RT-09,UAV-INDIA-09,0,1.2850,103.7850,40.0,4.8,-2.8,7.9,0.0,33.0,NFZ-A,8420.5,0,8.2
RT-09,UAV-INDIA-09,1,1.2950,103.7920,40.0,4.2,-2.4,7.1,0.0,33.2,NFZ-A,7100.2,0,9.6
RT-09,UAV-INDIA-09,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,12.0
RT-10,UAV-JULIETT-10,0,1.2750,103.8250,56.0,4.6,-1.8,7.5,0.0,33.4,NFZ-A,8720.0,0,9.8
RT-10,UAV-JULIETT-10,1,1.2900,103.8120,50.0,3.8,-2.2,6.4,0.0,33.1,NFZ-A,7140.4,0,11.5
RT-10,UAV-JULIETT-10,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.2
RT-11,UAV-KILO-11,0,1.2950,103.8450,46.0,4.2,-1.2,6.8,0.0,33.2,NFZ-A,7040.6,0,11.2
RT-11,UAV-KILO-11,1,1.2980,103.8220,45.0,3.4,-2.0,5.9,0.0,33.0,NFZ-A,6420.0,0,12.8
RT-11,UAV-KILO-11,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,14.5
RT-12,UAV-LIMA-12,0,1.3050,103.7650,52.0,3.9,-2.1,6.0,0.0,32.8,NFZ-A,7950.0,0,10.5
RT-12,UAV-LIMA-12,1,1.3020,103.7820,48.0,3.2,-2.6,5.4,0.0,32.9,NFZ-A,6820.2,0,11.9
RT-12,UAV-LIMA-12,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.1
RT-13,UAV-MIKE-13,0,1.3950,103.8850,43.0,2.8,-1.2,4.5,0.0,31.2,NFZ-B,3120.4,0,13.4
RT-13,UAV-MIKE-13,1,1.3800,103.8920,42.0,2.9,-1.0,4.8,0.0,31.5,NFZ-B,1220.0,0,24.6
RT-13,UAV-MIKE-13,2,1.3720,103.8960,40.0,3.0,-0.8,5.0,0.0,31.8,NFZ-B,240.0,1,92.5
RT-13,UAV-MIKE-13,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,32.1
RT-14,UAV-NOVEMBER-14,0,1.3850,103.9350,39.0,3.1,-1.5,5.2,0.0,31.5,NFZ-B,4850.0,0,14.8
RT-14,UAV-NOVEMBER-14,1,1.3740,103.9020,38.0,3.0,-1.1,5.0,0.0,31.7,NFZ-B,980.5,0,26.4
RT-14,UAV-NOVEMBER-14,2,1.3690,103.8940,36.0,2.8,-0.9,4.7,0.0,31.9,NFZ-B,180.0,1,96.8
RT-14,UAV-NOVEMBER-14,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,31.2
RT-15,UAV-OSCAR-15,0,1.3350,103.9350,47.0,3.4,-1.8,5.8,0.0,32.0,NFZ-C,7120.0,0,15.2
RT-15,UAV-OSCAR-15,1,1.3480,103.9650,45.0,3.2,-1.4,5.4,0.0,31.8,NFZ-C,3420.0,0,19.4
RT-15,UAV-OSCAR-15,2,1.3620,103.9880,44.0,3.4,-1.0,5.6,0.0,31.4,NFZ-C,390.0,1,89.5
RT-15,UAV-OSCAR-15,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,36.4
RT-16,UAV-PAPA-16,0,1.3200,103.9100,53.0,3.6,-2.0,6.0,0.0,32.4,NFZ-C,9450.0,0,13.8
RT-16,UAV-PAPA-16,1,1.3450,103.9550,50.0,3.3,-1.6,5.6,0.0,31.9,NFZ-C,4210.0,0,18.6
RT-16,UAV-PAPA-16,2,1.3610,103.9890,48.0,3.4,-1.1,5.5,0.0,31.5,NFZ-C,280.0,1,93.2
RT-16,UAV-PAPA-16,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,34.0"""

df = pd.read_csv(io.StringIO(csv_raw))

# ------------------------------------------------------------------------------
# 2. PRESENTATION SLIDE CONFIGURATION (16:9 4K)
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.1

fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#020617')

# Master Vertical Allocation with generous row spacing (hspace=0.52)
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

# ------------------------------------------------------------------------------
# 3. HEADER & SUMMARY METRIC CARDS
# ------------------------------------------------------------------------------
header_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=master_gs[0], 
                                            width_ratios=[0.42, 0.145, 0.145, 0.145, 0.145], 
                                            wspace=0.08)

ax_title = fig.add_subplot(header_gs[0])
ax_title.set_facecolor(BG_PANEL)
ax_title.axis('off')
ax_title.text(0.0, 0.65, "PRE-GNN METEOROLOGICAL ROUTING ANALYSIS", 
              fontsize=14.0, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Spatial NFZ Invariant Violations & Weather Impedance Prior to Attention GNN Re-routing", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

total_nodes = len(df)
breach_nodes = df['nfz_breach_pre_gnn'].sum()
mean_cost = df['baseline_edge_cost'].mean()
peak_rain = df['rain_intensity_mm_hr'].max()

kpis = [
    ("TOTAL EVAL NODES", f"{total_nodes} Waypoints", CYAN),
    ("NFZ BREACH FLAGS", f"{breach_nodes} Incursions", ROSE),
    ("MEAN BASELINE COST", f"{mean_cost:.1f} Units", AMBER),
    ("PEAK PRECIPITATION", f"{peak_rain:.1f} mm/h", EMERALD),
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

# ------------------------------------------------------------------------------
# 4. ROW 1: SPATIAL MAP & EDGE COST PROXIMITY
# ------------------------------------------------------------------------------
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], 
                                          width_ratios=[0.56, 0.44], wspace=0.25)

# Plot 1: Geospatial Coordinates, Flight Trajectories, and NFZ Perimeters
ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)

nfz_zones = [
    (103.8200, 1.3530, 1400 / 111000.0, "NFZ-A", (0, 0.016)),
    (103.8950, 1.3700, 1600 / 111000.0, "NFZ-B", (0, 0.018)),
    (103.9915, 1.3644, 2200 / 111000.0, "NFZ-C", (0, 0.023))
]

for cx, cy, rad, name, offset in nfz_zones:
    circ = Circle((cx, cy), rad, color=ROSE, alpha=0.22, edgecolor=ROSE, linewidth=1.8, linestyle='--')
    ax1.add_patch(circ)
    # Displaced label above perimeter to avoid overlapping breach markers
    ax1.text(cx + offset[0], cy + offset[1], name, color='#fda4af', fontsize=9, fontweight='bold', 
             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='#020617', edgecolor=ROSE, alpha=0.85))

# Plot routes
for r_id, grp in df.groupby('route_id'):
    ax1.plot(grp['longitude'], grp['latitude'], color=CYAN, linewidth=1.2, alpha=0.45, linestyle=':')

# Scatter points: Safe vs Breach
safe_pts = df[df['nfz_breach_pre_gnn'] == 0]
breach_pts = df[df['nfz_breach_pre_gnn'] == 1]

ax1.scatter(safe_pts['longitude'], safe_pts['latitude'], color=EMERALD, s=36, alpha=0.85, label='Safe Waypoint (Pre-GNN)')
ax1.scatter(breach_pts['longitude'], breach_pts['latitude'], color=ROSE, s=75, marker='X', edgecolors='#ffffff', linewidths=0.8, label='NFZ Buffer Incursion (<500m)')

ax1.set_title("Unoptimized Multi-UAV Trajectories & NFZ Violations", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax1.set_xlabel("Longitude (°E)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax1.set_ylabel("Latitude (°N)", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax1.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper left')

# Plot 2: Baseline Edge Cost vs. Distance to Prohibited Airspace
ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)

sc2 = ax2.scatter(df['dist_to_nfz_m'], df['baseline_edge_cost'], c=df['wind_gust_mps'], 
                  cmap='plasma', s=45, alpha=0.85, edgecolors='none')
cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.05, fraction=0.046)
cbar2.set_label("Wind Gust Factor (m/s)", color=TEXT_MUTED, fontsize=8.5, labelpad=14)
cbar2.ax.tick_params(labelsize=8, colors=TEXT_MUTED, pad=4)

ax2.axvline(500.0, color=ROSE, linestyle='--', linewidth=1.8, label='500m Safety Buffer Limit')
ax2.set_title("Edge Cost Penalty vs. NFZ Proximity", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax2.set_xlabel("Distance to Nearest NFZ Perimeter (m)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax2.set_ylabel("Unweighted Baseline Edge Cost", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax2.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)
ax2.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper right')

# ------------------------------------------------------------------------------
# 5. ROW 2: METEOROLOGICAL IMPEDANCE & INVASION AUDIT
# ------------------------------------------------------------------------------
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.36)

# Plot 3: Wind Vector Components (u vs v)
ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)

ax3.quiver(df['longitude'], df['latitude'], df['wind_u_mps'], df['wind_v_mps'], 
           color=CYAN, scale=45, width=0.005, alpha=0.85)
ax3.set_title("Atmospheric Wind Field Vector Dispersion", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax3.set_xlabel("Longitude (°E)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax3.set_ylabel("Latitude (°N)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax3.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# Plot 4: Precipitation Drag Penalty by Route
ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)

route_rain = df.groupby('route_id')['rain_intensity_mm_hr'].mean()
ax4.bar(route_rain.index, route_rain.values, color=VIOLET, alpha=0.75, edgecolor=BORDER_COLOR)
ax4.set_title("Mean Rain Resistance Rate per Route (mm/h)", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax4.set_xlabel("Route Identifier", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax4.set_ylabel("Rain Intensity (mm/h)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax4.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=7.5, pad=6)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

# Plot 5: Pre-GNN Incursion Distribution by NFZ Sector
ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)

nfz_breach_counts = df[df['nfz_breach_pre_gnn'] == 1]['nearest_nfz_id'].value_counts()
ax5.bar(nfz_breach_counts.index, nfz_breach_counts.values, color=ROSE, alpha=0.8, edgecolor='#ffffff', linewidth=1.0)
ax5.set_title("Pre-GNN Airspace Breach Count by NFZ", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax5.set_xlabel("Restricted Zone ID", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax5.set_ylabel("Incursion Count", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax5.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# ------------------------------------------------------------------------------
# 6. EXPORT ARTIFACT
# ------------------------------------------------------------------------------
output_filename = "routing_pre_gnn_slide.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Exported Pre-GNN Meteorological Routing Slide: {output_filename}")