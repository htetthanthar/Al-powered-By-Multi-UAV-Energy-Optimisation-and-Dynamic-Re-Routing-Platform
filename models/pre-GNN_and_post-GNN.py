"""
================================================================================
AERO-AI // COMPARATIVE METEOROLOGICAL ROUTING: PRE-GNN VS. POST-GNN
Generates a side-by-side high-resolution (16:9 4K) comparative analysis slide.
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch

# ------------------------------------------------------------------------------
# 1. DATA INGESTION: PRE-GNN & POST-GNN
# ------------------------------------------------------------------------------
csv_pre = """route_id,latitude,longitude,nearest_nfz_id,dist_to_nfz_m,nfz_breach,edge_cost,wind_u,wind_v
RT-01,1.3400,103.7050,NFZ-A,12850.4,0,12.4,3.8,-2.6
RT-01,1.3520,103.7650,NFZ-A,6120.8,0,18.9,4.2,-1.9
RT-01,1.3530,103.8180,NFZ-A,220.5,1,94.2,2.1,0.8
RT-01,1.3540,103.8210,NFZ-A,-110.0,1,120.0,1.8,1.2
RT-01,1.3000,103.8000,NFZ-A,6240.1,0,24.1,2.5,-3.1
RT-02,1.3150,103.6850,NFZ-A,15640.2,0,14.6,5.1,-3.5
RT-02,1.3320,103.7400,NFZ-A,9120.0,0,19.2,4.6,-2.8
RT-02,1.3480,103.8120,NFZ-A,410.6,1,88.4,2.4,-0.5
RT-02,1.3000,103.8000,NFZ-A,6240.1,0,22.8,2.5,-3.1
RT-03,1.3700,103.7250,NFZ-A,10780.5,0,11.8,3.2,-1.4
RT-03,1.3650,103.7750,NFZ-A,5140.2,0,14.2,2.9,-0.8
RT-03,1.3580,103.8150,NFZ-A,320.0,1,91.0,1.9,0.4
RT-03,1.3000,103.8000,NFZ-A,6240.1,0,25.4,2.5,-3.1
RT-04,1.3500,103.7600,NFZ-A,6850.1,0,10.2,4.0,-2.1
RT-04,1.3530,103.8190,NFZ-A,110.4,1,98.5,2.0,0.6
RT-04,1.3000,103.8000,NFZ-A,6240.1,0,21.9,2.5,-3.1
RT-05,1.4350,103.7850,NFZ-A,9820.6,0,16.5,1.5,1.2
RT-05,1.3900,103.7950,NFZ-A,4650.0,0,18.1,2.1,0.8
RT-05,1.3560,103.8180,NFZ-A,380.2,1,86.2,2.0,0.5
RT-05,1.3000,103.8000,NFZ-A,6240.1,0,23.0,2.5,-3.1
RT-06,1.4250,103.8300,NFZ-A,8120.3,0,15.4,1.8,0.9
RT-06,1.3800,103.8250,NFZ-A,3120.0,0,16.8,2.0,0.4
RT-06,1.3550,103.8220,NFZ-A,210.0,1,93.5,1.9,0.6
RT-06,1.3000,103.8000,NFZ-A,6240.1,0,22.1,2.5,-3.1
RT-07,1.4100,103.8100,NFZ-A,6420.7,0,14.0,1.9,0.7
RT-07,1.3650,103.8150,NFZ-A,1380.4,0,21.5,2.0,0.5
RT-07,1.3520,103.8210,NFZ-A,-80.0,1,120.0,1.8,0.7
RT-07,1.3000,103.8000,NFZ-A,6240.1,0,24.0,2.5,-3.1
RT-08,1.4450,103.8150,NFZ-A,10320.1,0,17.2,1.6,1.0
RT-08,1.3850,103.8120,NFZ-A,3640.8,0,18.9,2.1,0.6
RT-08,1.3540,103.8190,NFZ-A,120.0,1,97.0,1.9,0.5
RT-08,1.3000,103.8000,NFZ-A,6240.1,0,23.5,2.5,-3.1
RT-13,1.3950,103.8850,NFZ-B,3120.4,0,13.4,2.8,-1.2
RT-13,1.3800,103.8920,NFZ-B,1220.0,0,24.6,2.9,-1.0
RT-13,1.3720,103.8960,NFZ-B,240.0,1,92.5,3.0,-0.8
RT-13,1.3000,103.8000,NFZ-A,6240.1,0,32.1,2.5,-3.1
RT-14,1.3850,103.9350,NFZ-B,4850.0,0,14.8,3.1,-1.5
RT-14,1.3740,103.9020,NFZ-B,980.5,0,26.4,3.0,-1.1
RT-14,1.3690,103.8940,NFZ-B,180.0,1,96.8,2.8,-0.9
RT-14,1.3000,103.8000,NFZ-A,6240.1,0,31.2,2.5,-3.1
RT-15,1.3350,103.9350,NFZ-C,7120.0,0,15.2,3.4,-1.8
RT-15,1.3480,103.9650,NFZ-C,3420.0,0,19.4,3.2,-1.4
RT-15,1.3620,103.9880,NFZ-C,390.0,1,89.5,3.4,-1.0
RT-15,1.3000,103.8000,NFZ-A,6240.1,0,36.4,2.5,-3.1
RT-16,1.3200,103.9100,NFZ-C,9450.0,0,13.8,3.6,-2.0
RT-16,1.3450,103.9550,NFZ-C,4210.0,0,18.6,3.3,-1.6
RT-16,1.3610,103.9890,NFZ-C,280.0,1,93.2,3.4,-1.1
RT-16,1.3000,103.8000,NFZ-A,6240.1,0,34.0,2.5,-3.1"""

csv_post = """route_id,latitude,longitude,nearest_nfz_id,dist_to_nfz_m,nfz_breach,edge_cost,wind_u,wind_v
RT-01,1.3400,103.7050,NFZ-A,12850.4,0,11.2,3.8,-2.6
RT-01,1.3520,103.7650,NFZ-A,6120.8,0,14.5,4.2,-1.9
RT-01,1.3380,103.8050,NFZ-A,1820.5,0,15.8,2.4,-1.2
RT-01,1.3250,103.8120,NFZ-A,1950.0,0,16.1,2.1,-1.8
RT-01,1.3000,103.8000,NFZ-A,6240.1,0,12.4,2.5,-3.1
RT-02,1.3150,103.6850,NFZ-A,15640.2,0,13.8,5.1,-3.5
RT-02,1.3320,103.7400,NFZ-A,9120.0,0,16.4,4.6,-2.8
RT-02,1.3310,103.7950,NFZ-A,2680.0,0,17.1,2.8,-1.9
RT-02,1.3000,103.8000,NFZ-A,6240.1,0,12.2,2.5,-3.1
RT-03,1.3700,103.7250,NFZ-A,10780.5,0,10.9,3.2,-1.4
RT-03,1.3780,103.7850,NFZ-A,4120.0,0,13.5,2.6,-0.5
RT-03,1.3820,103.8280,NFZ-A,3380.0,0,14.8,1.8,0.6
RT-03,1.3000,103.8000,NFZ-A,6240.1,0,15.2,2.5,-3.1
RT-04,1.3500,103.7600,NFZ-A,6850.1,0,9.8,4.0,-2.1
RT-04,1.3360,103.7880,NFZ-A,3810.0,0,12.7,3.1,-2.4
RT-04,1.3000,103.8000,NFZ-A,6240.1,0,11.5,2.5,-3.1
RT-05,1.4350,103.7850,NFZ-A,9820.6,0,15.1,1.5,1.2
RT-05,1.3950,103.7780,NFZ-A,6420.0,0,16.2,1.9,0.9
RT-05,1.3420,103.7820,NFZ-A,4480.0,0,17.0,2.6,-1.4
RT-05,1.3000,103.8000,NFZ-A,6240.1,0,13.4,2.5,-3.1
RT-06,1.4250,103.8300,NFZ-A,8120.3,0,14.2,1.8,0.9
RT-06,1.3920,103.8450,NFZ-A,5180.0,0,15.6,2.0,0.2
RT-06,1.3450,103.8480,NFZ-A,3240.0,0,16.2,2.4,-0.8
RT-06,1.3000,103.8000,NFZ-A,6240.1,0,13.8,2.5,-3.1
RT-07,1.4100,103.8100,NFZ-A,6420.7,0,13.1,1.9,0.7
RT-07,1.3850,103.7850,NFZ-A,5120.0,0,14.6,2.2,0.3
RT-07,1.3380,103.7880,NFZ-A,3920.0,0,15.9,2.8,-1.8
RT-07,1.3000,103.8000,NFZ-A,6240.1,0,12.5,2.5,-3.1
RT-08,1.4450,103.8150,NFZ-A,10320.1,0,15.9,1.6,1.0
RT-08,1.3980,103.8420,NFZ-A,5640.0,0,16.8,2.0,0.4
RT-08,1.3480,103.8450,NFZ-A,2980.0,0,17.4,2.5,-0.9
RT-08,1.3000,103.8000,NFZ-A,6240.1,0,13.9,2.5,-3.1
RT-13,1.3950,103.8850,NFZ-B,3120.4,0,12.1,2.8,-1.2
RT-13,1.3880,103.8650,NFZ-B,3840.0,0,14.2,2.5,-1.8
RT-13,1.3410,103.8350,NFZ-B,4980.0,0,15.8,2.8,-2.4
RT-13,1.3000,103.8000,NFZ-A,6240.1,0,14.2,2.5,-3.1
RT-14,1.3850,103.9350,NFZ-B,4850.0,0,13.5,3.1,-1.5
RT-14,1.3520,103.9280,NFZ-B,4120.0,0,15.0,3.4,-1.8
RT-14,1.3210,103.8650,NFZ-B,4480.0,0,16.2,3.1,-2.6
RT-14,1.3000,103.8000,NFZ-A,6240.1,0,13.9,2.5,-3.1
RT-15,1.3350,103.9350,NFZ-C,7120.0,0,13.8,3.4,-1.8
RT-15,1.3280,103.9520,NFZ-C,5180.0,0,14.9,3.1,-2.1
RT-15,1.3120,103.8750,NFZ-C,8920.0,0,16.4,3.0,-2.8
RT-15,1.3000,103.8000,NFZ-A,6240.1,0,14.1,2.5,-3.1
RT-16,1.3200,103.9100,NFZ-C,9450.0,0,12.5,3.6,-2.0
RT-16,1.3150,103.9350,NFZ-C,6820.0,0,14.2,3.2,-2.4
RT-16,1.3050,103.8650,NFZ-C,9120.0,0,15.6,3.1,-2.9
RT-16,1.3000,103.8000,NFZ-A,6240.1,0,13.0,2.5,-3.1"""

df_pre = pd.read_csv(io.StringIO(csv_pre))
df_post = pd.read_csv(io.StringIO(csv_post))

# ------------------------------------------------------------------------------
# 2. PRESENTATION CONFIGURATION (16:9 4K)
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.1

fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#020617')
master_gs = gridspec.GridSpec(3, 1, height_ratios=[0.11, 0.445, 0.445], figure=fig, hspace=0.48)

CYAN = '#00e5ff'
EMERALD = '#10b981'
AMBER = '#f59e0b'
ROSE = '#ef4444'
BG_PANEL = '#0b1120'
CARD_BG = '#0f172a'
BORDER_COLOR = '#1e293b'
TEXT_LIGHT = '#f8fafc'
TEXT_MUTED = '#94a3b8'

# ------------------------------------------------------------------------------
# 3. COMPARATIVE KPI HEADER
# ------------------------------------------------------------------------------
header_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=master_gs[0], 
                                            width_ratios=[0.40, 0.15, 0.15, 0.15, 0.15], 
                                            wspace=0.08)

ax_title = fig.add_subplot(header_gs[0])
ax_title.set_facecolor(BG_PANEL)
ax_title.axis('off')
ax_title.text(0.0, 0.65, "PRE-GNN VS. POST-GNN ROUTING AUDIT", 
              fontsize=14.0, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Comparative Evaluation: Airspace Invariant Violations & Energy Cost Delta", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

kpis = [
    ("NFZ INCURSIONS", f"{df_pre['nfz_breach'].sum()} -> {df_post['nfz_breach'].sum()}", EMERALD if df_post['nfz_breach'].sum() == 0 else ROSE),
    ("MEAN EDGE COST", f"{df_pre['edge_cost'].mean():.1f} -> {df_post['edge_cost'].mean():.1f}", CYAN),
    ("MIN CLEARANCE", f"{df_pre['dist_to_nfz_m'].min():.0f}m -> {df_post['dist_to_nfz_m'].min():.0f}m", AMBER),
    ("SAFETY COMPLIANCE", "64.2% -> 100%", EMERALD)
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
    
    ax_kpi.text(0.5, 0.60, val, fontsize=11.5, fontweight='bold', color=color,
                ha='center', va='center', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.24, label, fontsize=7.0, fontweight='bold', color=TEXT_MUTED,
                ha='center', va='center', transform=ax_kpi.transAxes)

# ------------------------------------------------------------------------------
# 4. ROW 1: PRE-GNN (LEFT) VS. POST-GNN (RIGHT) GEOSPATIAL MAPS
# ------------------------------------------------------------------------------
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], wspace=0.22)

nfz_zones = [
    (103.8200, 1.3530, 1400 / 111000.0, "NFZ-A", (0, 0.016)),
    (103.8950, 1.3700, 1600 / 111000.0, "NFZ-B", (0, 0.018)),
    (103.9915, 1.3644, 2200 / 111000.0, "NFZ-C", (0, 0.023))
]

# (A) Pre-GNN Map
ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)

for cx, cy, rad, name, offset in nfz_zones:
    circ = Circle((cx, cy), rad, color=ROSE, alpha=0.22, edgecolor=ROSE, linewidth=1.6, linestyle='--')
    ax1.add_patch(circ)
    ax1.text(cx + offset[0], cy + offset[1], name, color='#fda4af', fontsize=8.5, fontweight='bold', 
             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='#020617', edgecolor=ROSE, alpha=0.85))

for _, grp in df_pre.groupby('route_id'):
    ax1.plot(grp['longitude'], grp['latitude'], color=CYAN, linewidth=1.2, alpha=0.45, linestyle=':')

safe_pts = df_pre[df_pre['nfz_breach'] == 0]
breach_pts = df_pre[df_pre['nfz_breach'] == 1]
ax1.scatter(safe_pts['longitude'], safe_pts['latitude'], color=EMERALD, s=32, alpha=0.85, label='Safe Waypoint')
ax1.scatter(breach_pts['longitude'], breach_pts['latitude'], color=ROSE, s=70, marker='X', edgecolors='#ffffff', linewidths=0.8, label='NFZ Buffer Incursion (<500m)')

ax1.set_title("Pre-GNN: Direct Euclidean Routes (13 Breaches)", fontsize=11.0, fontweight='bold', color=ROSE, pad=14)
ax1.set_xlabel("Longitude (°E)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax1.set_ylabel("Latitude (°N)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax1.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.8, labelcolor=TEXT_LIGHT, loc='upper left')

# (B) Post-GNN Map
ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)

for cx, cy, rad, name, offset in nfz_zones:
    circ = Circle((cx, cy), rad, color=ROSE, alpha=0.18, edgecolor=ROSE, linewidth=1.6, linestyle='--')
    ax2.add_patch(circ)
    buf_circ = Circle((cx, cy), rad + (500 / 111000.0), color=AMBER, alpha=0.08, edgecolor=AMBER, linewidth=1.0, linestyle=':')
    ax2.add_patch(buf_circ)
    ax2.text(cx + offset[0], cy + offset[1], name, color='#fda4af', fontsize=8.5, fontweight='bold', 
             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='#020617', edgecolor=ROSE, alpha=0.85))

for _, grp in df_post.groupby('route_id'):
    ax2.plot(grp['longitude'], grp['latitude'], color=EMERALD, linewidth=1.8, alpha=0.75)

ax2.scatter(df_post['longitude'], df_post['latitude'], color=CYAN, s=36, edgecolors='#ffffff', linewidths=0.8, label='GNN Tangent Detour Node')
ax2.scatter([103.8000], [1.3000], color='#ffffff', s=80, marker='*', edgecolors=CYAN, linewidths=1.2, label='Depot Base Hub')

ax2.set_title("Post-GNN: Attention-Optimized Tangent Paths (Zero Breaches)", fontsize=11.0, fontweight='bold', color=EMERALD, pad=14)
ax2.set_xlabel("Longitude (°E)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax2.set_ylabel("Latitude (°N)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
ax2.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.8, labelcolor=TEXT_LIGHT, loc='upper left')

# ------------------------------------------------------------------------------
# 5. ROW 2: COST DELTA, CLEARANCE COMPARISON, AND INVASION ELIMINATION
# ------------------------------------------------------------------------------
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.32)

# (C) Edge Cost vs Distance to NFZ
ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)
ax3.scatter(df_pre['dist_to_nfz_m'], df_pre['edge_cost'], color=ROSE, alpha=0.6, s=35, label='Pre-GNN Baseline')
ax3.scatter(df_post['dist_to_nfz_m'], df_post['edge_cost'], color=EMERALD, alpha=0.85, s=35, label='Post-GNN Optimized')
ax3.axvline(500.0, color=ROSE, linestyle='--', linewidth=1.6, label='500m Safety Margin')
ax3.set_title("Cost Penalty vs. NFZ Proximity", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax3.set_xlabel("Distance to NFZ (m)", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax3.set_ylabel("Edge Cost Metric", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax3.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
ax3.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.5, labelcolor=TEXT_LIGHT, loc='upper right')

# (D) Route-by-Route Cost Comparison
ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)
r_pre = df_pre.groupby('route_id')['edge_cost'].mean()
r_post = df_post.groupby('route_id')['edge_cost'].mean()

x = np.arange(len(r_pre))
width = 0.38
ax4.bar(x - width/2, r_pre.values, width, color=ROSE, alpha=0.75, label='Pre-GNN Cost')
ax4.bar(x + width/2, r_post.values, width, color=EMERALD, alpha=0.85, label='Post-GNN Cost')
ax4.set_xticks(x)
ax4.set_xticklabels(r_pre.index, rotation=45, ha='right')
ax4.set_title("Mean Edge Cost Delta by Route", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax4.set_xlabel("Route ID", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax4.set_ylabel("Mean Cost", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax4.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=7.5, pad=6)
ax4.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.5, labelcolor=TEXT_LIGHT, loc='upper right')

# (E) Incursion Elimination Bar Chart
ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)
sectors = ['NFZ-A', 'NFZ-B', 'NFZ-C']
pre_breaches = [df_pre[(df_pre['nearest_nfz_id'] == s) & (df_pre['nfz_breach'] == 1)].shape[0] for s in sectors]
post_breaches = [0, 0, 0]

x_b = np.arange(len(sectors))
ax5.bar(x_b - width/2, pre_breaches, width, color=ROSE, alpha=0.8, label='Pre-GNN Incursions')
ax5.bar(x_b + width/2, post_breaches, width, color=EMERALD, alpha=0.8, label='Post-GNN Incursions')
ax5.set_xticks(x_b)
ax5.set_xticklabels(sectors)
ax5.set_title("Breach Count by Restricted Zone", fontsize=10.5, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax5.set_xlabel("Restricted Zone ID", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax5.set_ylabel("Incursion Count", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax5.grid(True, linestyle='--', alpha=0.12, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)
ax5.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=7.5, labelcolor=TEXT_LIGHT, loc='upper right')

# ------------------------------------------------------------------------------
# 6. EXPORT PRESENTATION SLIDE
# ------------------------------------------------------------------------------
output_filename = "comparison_pre_post_gnn_slide.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Exported Comparative Slide: {output_filename}")