"""
================================================================================
AERO-AI // METEOROLOGICAL ROUTING POST-GNN VISUALIZATION ENGINE
Extracts post-GNN conflict-free routing telemetry and exports a 16:9 presentation slide.
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch

# ------------------------------------------------------------------------------
# 1. LOAD POST-GNN DATASET
# ------------------------------------------------------------------------------
csv_data = """route_id,uav_id,waypoint_idx,latitude,longitude,alt_m,wind_u_mps,wind_v_mps,wind_gust_mps,rain_intensity_mm_hr,temp_c,nearest_nfz_id,dist_to_nfz_m,nfz_breach_post_gnn,gnn_optimized_edge_cost
RT-01,UAV-ALPHA-01,0,1.3400,103.7050,48.2,3.8,-2.6,6.8,18.4,30.5,NFZ-A,12850.4,0,11.2
RT-01,UAV-ALPHA-01,1,1.3520,103.7650,50.0,4.2,-1.9,7.4,12.2,31.2,NFZ-A,6120.8,0,14.5
RT-01,UAV-ALPHA-01,2,1.3380,103.8050,48.5,2.4,-1.2,4.1,0.0,32.2,NFZ-A,1820.5,0,15.8
RT-01,UAV-ALPHA-01,3,1.3250,103.8120,48.0,2.1,-1.8,3.9,0.0,32.6,NFZ-A,1950.0,0,16.1
RT-01,UAV-ALPHA-01,4,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,12.4
RT-02,UAV-BRAVO-02,0,1.3150,103.6850,55.0,5.1,-3.5,8.2,26.5,29.8,NFZ-A,15640.2,0,13.8
RT-02,UAV-BRAVO-02,1,1.3320,103.7400,55.0,4.6,-2.8,7.0,15.8,30.4,NFZ-A,9120.0,0,16.4
RT-02,UAV-BRAVO-02,2,1.3310,103.7950,52.0,2.8,-1.9,4.8,0.0,32.1,NFZ-A,2680.0,0,17.1
RT-02,UAV-BRAVO-02,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,12.2
RT-03,UAV-CHARLIE-03,0,1.3700,103.7250,42.6,3.2,-1.4,5.1,6.4,31.0,NFZ-A,10780.5,0,10.9
RT-03,UAV-CHARLIE-03,1,1.3780,103.7850,45.0,2.6,-0.5,4.2,0.0,31.9,NFZ-A,4120.0,0,13.5
RT-03,UAV-CHARLIE-03,2,1.3820,103.8280,44.0,1.8,0.6,3.0,0.0,32.4,NFZ-A,3380.0,0,14.8
RT-03,UAV-CHARLIE-03,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,15.2
RT-04,UAV-DELTA-04,0,1.3500,103.7600,38.0,4.0,-2.1,6.3,8.0,31.4,NFZ-A,6850.1,0,9.8
RT-04,UAV-DELTA-04,1,1.3360,103.7880,40.0,3.1,-2.4,5.2,0.0,32.6,NFZ-A,3810.0,0,12.7
RT-04,UAV-DELTA-04,2,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,11.5
RT-05,UAV-ECHO-05,0,1.4350,103.7850,60.0,1.5,1.2,2.8,0.0,32.1,NFZ-A,9820.6,0,15.1
RT-05,UAV-ECHO-05,1,1.3950,103.7780,58.0,1.9,0.9,3.2,0.0,32.4,NFZ-A,6420.0,0,16.2
RT-05,UAV-ECHO-05,2,1.3420,103.7820,55.0,2.6,-1.4,4.2,0.0,32.7,NFZ-A,4480.0,0,17.0
RT-05,UAV-ECHO-05,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.4
RT-06,UAV-FOXTROT-06,0,1.4250,103.8300,50.5,1.8,0.9,3.0,0.0,32.5,NFZ-A,8120.3,0,14.2
RT-06,UAV-FOXTROT-06,1,1.3920,103.8450,48.0,2.0,0.2,3.1,0.0,32.7,NFZ-A,5180.0,0,15.6
RT-06,UAV-FOXTROT-06,2,1.3450,103.8480,46.0,2.4,-0.8,4.0,0.0,32.8,NFZ-A,3240.0,0,16.2
RT-06,UAV-FOXTROT-06,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.8
RT-07,UAV-GOLF-07,0,1.4100,103.8100,58.0,1.9,0.7,3.1,0.0,32.2,NFZ-A,6420.7,0,13.1
RT-07,UAV-GOLF-07,1,1.3850,103.7850,52.0,2.2,0.3,3.4,0.0,32.4,NFZ-A,5120.0,0,14.6
RT-07,UAV-GOLF-07,2,1.3380,103.7880,50.0,2.8,-1.8,4.5,0.0,32.8,NFZ-A,3920.0,0,15.9
RT-07,UAV-GOLF-07,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,12.5
RT-08,UAV-HOTEL-08,0,1.4450,103.8150,44.0,1.6,1.0,2.9,0.0,32.0,NFZ-A,10320.1,0,15.9
RT-08,UAV-HOTEL-08,1,1.3980,103.8420,42.0,2.0,0.4,3.2,0.0,32.5,NFZ-A,5640.0,0,16.8
RT-08,UAV-HOTEL-08,2,1.3480,103.8450,40.0,2.5,-0.9,4.1,0.0,32.8,NFZ-A,2980.0,0,17.4
RT-08,UAV-HOTEL-08,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.9
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
RT-13,UAV-MIKE-13,0,1.3950,103.8850,43.0,2.8,-1.2,4.5,0.0,31.2,NFZ-B,3120.4,0,12.1
RT-13,UAV-MIKE-13,1,1.3880,103.8650,42.0,2.5,-1.8,4.6,0.0,31.6,NFZ-B,3840.0,0,14.2
RT-13,UAV-MIKE-13,2,1.3410,103.8350,40.0,2.8,-2.4,5.1,0.0,32.4,NFZ-B,4980.0,0,15.8
RT-13,UAV-MIKE-13,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,14.2
RT-14,UAV-NOVEMBER-14,0,1.3850,103.9350,39.0,3.1,-1.5,5.2,0.0,31.5,NFZ-B,4850.0,0,13.5
RT-14,UAV-NOVEMBER-14,1,1.3520,103.9280,38.0,3.4,-1.8,5.5,0.0,31.8,NFZ-B,4120.0,0,15.0
RT-14,UAV-NOVEMBER-14,2,1.3210,103.8650,36.0,3.1,-2.6,5.3,0.0,32.6,NFZ-B,4480.0,0,16.2
RT-14,UAV-NOVEMBER-14,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.9
RT-15,UAV-OSCAR-15,0,1.3350,103.9350,47.0,3.4,-1.8,5.8,0.0,32.0,NFZ-C,7120.0,0,13.8
RT-15,UAV-OSCAR-15,1,1.3280,103.9520,45.0,3.1,-2.1,5.2,0.0,32.2,NFZ-C,5180.0,0,14.9
RT-15,UAV-OSCAR-15,2,1.3120,103.8750,44.0,3.0,-2.8,5.4,0.0,32.7,NFZ-C,8920.0,0,16.4
RT-15,UAV-OSCAR-15,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,14.1
RT-16,UAV-PAPA-16,0,1.3200,103.9100,53.0,3.6,-2.0,6.0,0.0,32.4,NFZ-C,9450.0,0,12.5
RT-16,UAV-PAPA-16,1,1.3150,103.9350,50.0,3.2,-2.4,5.4,0.0,32.1,NFZ-C,6820.0,0,14.2
RT-16,UAV-PAPA-16,2,1.3050,103.8650,48.0,3.1,-2.9,5.2,0.0,32.8,NFZ-C,9120.0,0,15.6
RT-16,UAV-PAPA-16,3,1.3000,103.8000,0.0,2.5,-3.1,4.9,0.0,33.0,NFZ-A,6240.1,0,13.0"""

df = pd.read_csv(io.StringIO(csv_data))

# ------------------------------------------------------------------------------
# 2. PRESENTATION SLIDE CONFIGURATION (16:9 4K)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 3. HEADER & KPI CARDS
# ------------------------------------------------------------------------------
header_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=master_gs[0], 
                                            width_ratios=[0.42, 0.145, 0.145, 0.145, 0.145], 
                                            wspace=0.08)

ax_title = fig.add_subplot(header_gs[0])
ax_title.set_facecolor(BG_PANEL)
ax_title.axis('off')
ax_title.text(0.0, 0.65, "POST-GNN METEOROLOGICAL ROUTING OPTIMIZATION", 
              fontsize=13.5, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Spatial NFZ Avoidance & Attention-Weighted Energy Path Planning Across Singapore", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

total_nodes = len(df)
breach_nodes = df['nfz_breach_post_gnn'].sum()
mean_cost = df['gnn_optimized_edge_cost'].mean()
min_dist_to_nfz = df['dist_to_nfz_m'].min()

kpis = [
    ("OPTIMIZED NODES", f"{total_nodes} Waypoints", CYAN),
    ("ACTIVE BREACHES", f"{breach_nodes} (0.00%)", EMERALD),
    ("MEAN ROUTE COST", f"{mean_cost:.1f} Units", AMBER),
    ("MIN NFZ CLEARANCE", f"{min_dist_to_nfz:.0f} m", VIOLET),
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
# 4. ROW 1: OPTIMIZED TANGENT GEOSPATIAL MAP & POST-GNN EDGE COST PROFILE
# ------------------------------------------------------------------------------
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], 
                                          width_ratios=[0.56, 0.44], wspace=0.25)

# Plot 1: Geospatial Coordinates, GNN Tangent Detour Paths, and NFZ Perimeters
ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)

nfz_zones = [
    (103.8200, 1.3530, 1400 / 111000.0, "NFZ-A", (0, 0.016)),
    (103.8950, 1.3700, 1600 / 111000.0, "NFZ-B", (0, 0.018)),
    (103.9915, 1.3644, 2200 / 111000.0, "NFZ-C", (0, 0.023))
]

for cx, cy, rad, name, offset in nfz_zones:
    circ = Circle((cx, cy), rad, color=ROSE, alpha=0.18, edgecolor=ROSE, linewidth=1.8, linestyle='--')
    ax1.add_patch(circ)
    # Safety buffer boundary (500m outward)
    buffer_circ = Circle((cx, cy), rad + (500 / 111000.0), color=AMBER, alpha=0.08, edgecolor=AMBER, linewidth=1.2, linestyle=':')
    ax1.add_patch(buffer_circ)
    
    ax1.text(cx + offset[0], cy + offset[1], name, color='#fda4af', fontsize=9, fontweight='bold', 
             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='#020617', edgecolor=ROSE, alpha=0.85))

# Plot optimized routes
for r_id, grp in df.groupby('route_id'):
    ax1.plot(grp['longitude'], grp['latitude'], color=EMERALD, linewidth=1.8, alpha=0.75)

# Scatter points: Verified Safe Waypoints
ax1.scatter(df['longitude'], df['latitude'], color=CYAN, s=42, edgecolors='#ffffff', linewidths=0.8, label='GNN Tangent Waypoint (Conflict-Free)')
# Base depot hub
ax1.scatter([103.8000], [1.3000], color='#ffffff', s=90, marker='*', edgecolors=CYAN, linewidths=1.2, label='Depot Hub (Queenstown Base)')

ax1.set_title("Attention-Based GNN Re-routed Trajectories (Zero Breaches)", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax1.set_xlabel("Longitude (°E)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax1.set_ylabel("Latitude (°N)", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax1.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper left')

# Plot 2: Optimized Edge Cost vs. Distance to Prohibited Airspace
ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)

sc2 = ax2.scatter(df['dist_to_nfz_m'], df['gnn_optimized_edge_cost'], c=df['wind_gust_mps'], 
                  cmap='viridis', s=45, alpha=0.9, edgecolors='none')
cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.05, fraction=0.046)
cbar2.set_label("Wind Gust Factor (m/s)", color=TEXT_MUTED, fontsize=8.5, labelpad=14)
cbar2.ax.tick_params(labelsize=8, colors=TEXT_MUTED, pad=4)

ax2.axvline(500.0, color=ROSE, linestyle='--', linewidth=1.8, label='500m Hard Safety Buffer')
ax2.set_title("Post-GNN Edge Cost vs. Distance to NFZ", fontsize=11.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax2.set_xlabel("Distance to Nearest NFZ Perimeter (m)", fontsize=9.5, color=TEXT_MUTED, labelpad=12)
ax2.set_ylabel("Optimized GNN Edge Cost", fontsize=9.5, color=TEXT_MUTED, labelpad=14)
ax2.set_ylim(5, 25)
ax2.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.5, pad=6)
ax2.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper right')

# ------------------------------------------------------------------------------
# 5. ROW 2: WIND ATTENTION VECTOR, COST DELTA, AND DISTANCE DISTRIBUTION
# ------------------------------------------------------------------------------
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.36)

# Plot 3: Wind Field Vectors along Optimized Paths
ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)

ax3.quiver(df['longitude'], df['latitude'], df['wind_u_mps'], df['wind_v_mps'], 
           color=EMERALD, scale=45, width=0.005, alpha=0.85)
ax3.set_title("Attention-Aligned Wind Drift Streamlines", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax3.set_xlabel("Longitude (°E)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax3.set_ylabel("Latitude (°N)", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax3.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# Plot 4: Mean Edge Cost Comparison by Route
ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)

route_cost = df.groupby('route_id')['gnn_optimized_edge_cost'].mean()
ax4.bar(route_cost.index, route_cost.values, color=CYAN, alpha=0.75, edgecolor=BORDER_COLOR)
ax4.set_title("Mean Optimized Route Cost per UAV", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax4.set_xlabel("Route Identifier", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax4.set_ylabel("GNN Edge Cost", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax4.set_ylim(0, 22)
ax4.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=7.5, pad=6)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

# Plot 5: Clearance Distance Histogram (Showing complete shift >= 1800m)
ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)

ax5.hist(df['dist_to_nfz_m'], bins=12, color=EMERALD, alpha=0.75, edgecolor='#020617')
ax5.axvline(500.0, color=ROSE, linestyle='--', linewidth=2.0, label='500m Safety Margin')
ax5.set_title("Post-GNN Distance-to-NFZ Distribution", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=18)
ax5.set_xlabel("Clearance Distance to NFZ (m)", fontsize=8.5, color=TEXT_MUTED, labelpad=12)
ax5.set_ylabel("Waypoint Count", fontsize=8.5, color=TEXT_MUTED, labelpad=14)
ax5.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
ax5.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=8.0, labelcolor=TEXT_LIGHT, loc='upper right')

# ------------------------------------------------------------------------------
# 6. EXPORT ARTIFACT
# ------------------------------------------------------------------------------
output_filename = "met_routing_post_gnn_slide.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Exported Post-GNN Meteorological Routing Slide: {output_filename}")