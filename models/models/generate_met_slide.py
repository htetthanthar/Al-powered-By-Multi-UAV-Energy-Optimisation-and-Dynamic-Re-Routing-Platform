"""
================================================================================
AERO-AI // POWERPOINT PRESENTATION SLIDE GENERATOR: METEOROLOGICAL ROUTING
Generates a 16:9 widescreen presentation graphic for spatial route optimization.
================================================================================
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------------------------
# 1. LOAD METEOROLOGICAL DATASET
# ------------------------------------------------------------------------------
csv_data = """timestamp_utc,station_id,latitude,longitude,temperature_c,relative_humidity_pct,wind_speed_mps,wind_direction_deg,rainfall_mm_hr,visibility_km,barometric_pressure_hpa,air_density_kg_m3
2026-08-30T06:00:00Z,S24,1.3644,103.9915,31.4,68.2,3.4,140.0,0.0,10.0,1008.2,1.162
2026-08-30T06:00:00Z,S50,1.3400,103.7050,30.2,74.5,4.8,210.0,12.4,6.5,1007.8,1.168
2026-08-30T06:00:00Z,S109,1.4350,103.7850,32.1,65.0,2.1,90.0,0.0,10.0,1008.5,1.159
2026-08-30T06:00:00Z,S111,1.3150,103.6850,29.8,78.1,6.2,230.0,28.6,4.2,1007.4,1.171
2026-08-30T06:00:00Z,S44,1.2850,103.7850,33.0,62.4,5.5,180.0,0.0,10.0,1008.0,1.155
2026-08-30T06:00:00Z,S121,1.3700,103.8950,31.8,69.0,3.0,120.0,0.0,10.0,1008.3,1.164
2026-08-30T06:00:00Z,S60,1.4250,103.8300,32.5,66.5,2.8,150.0,0.0,10.0,1008.4,1.157
2026-08-30T06:00:00Z,S115,1.2750,103.8250,33.4,60.8,4.9,170.0,0.0,10.0,1008.1,1.153
2026-08-30T06:00:00Z,S81,1.3050,103.7650,30.9,72.0,3.9,200.0,1.2,9.0,1007.9,1.166
2026-08-30T06:00:00Z,S90,1.3950,103.8850,31.2,70.5,3.2,130.0,0.0,10.0,1008.2,1.165"""

df = pd.read_csv(io.StringIO(csv_data))

# ------------------------------------------------------------------------------
# 2. PRESENTATION LAYOUT & STYLING SETUP
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#020617')
master_gs = gridspec.GridSpec(3, 1, height_ratios=[0.12, 0.44, 0.44], figure=fig, hspace=0.36)

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
ax_title.text(0.0, 0.65, "METEOROLOGICAL ROUTING & GNN LAYER", 
              fontsize=13.5, fontweight='bold', color=CYAN, transform=ax_title.transAxes, va='center')
ax_title.text(0.0, 0.22, "Singapore Sector Real-Time Atmospheric Feed & Spatial Invariants", 
              fontsize=9.0, color=TEXT_MUTED, transform=ax_title.transAxes, va='center')

kpis = [
    ("MEAN TEMP", f"{df['temperature_c'].mean():.1f} °C", AMBER),
    ("MAX WIND GUST", f"{df['wind_speed_mps'].max():.1f} m/s", ROSE),
    ("PEAK RAIN RATE", f"{df['rainfall_mm_hr'].max():.1f} mm/h", CYAN),
    ("AIR DENSITY (AVG)", f"{df['air_density_kg_m3'].mean():.3f} kg/m³", EMERALD),
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
# 4. ROW 1: SPATIAL MAP DISTRIBUTION & WIND/RAINFALL CORRELATIONS
# ------------------------------------------------------------------------------
row1_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=master_gs[1], width_ratios=[0.55, 0.45], wspace=0.22)

# Plot 1: Singapore Station Coordinates colored by Rainfall
ax1 = fig.add_subplot(row1_gs[0])
ax1.set_facecolor(BG_PANEL)
sc1 = ax1.scatter(df['longitude'], df['latitude'], c=df['rainfall_mm_hr'], cmap='Blues', s=120, edgecolors=CYAN, linewidths=1.5)
cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.03, fraction=0.046)
cbar1.set_label("Rainfall Rate (mm/h)", color=TEXT_MUTED, fontsize=8.5, labelpad=8)
cbar1.ax.tick_params(labelsize=8, colors=TEXT_MUTED)

for _, row in df.iterrows():
    ax1.text(row['longitude'] + 0.008, row['latitude'], row['station_id'], color=TEXT_LIGHT, fontsize=8, fontweight='bold')

ax1.set_title("NEA Weather Station Spatial Grid (Singapore)", fontsize=11.0, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax1.set_xlabel("Longitude (°E)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax1.set_ylabel("Latitude (°N)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax1.grid(True, linestyle='--', alpha=0.1, color='#ffffff')
ax1.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# Plot 2: Wind Speed vs. Rainfall Intensity
ax2 = fig.add_subplot(row1_gs[1])
ax2.set_facecolor(BG_PANEL)
ax2.scatter(df['wind_speed_mps'], df['rainfall_mm_hr'], color=ROSE, s=55, alpha=0.85, edgecolors='none')
ax2.set_title("Convective Precipitation vs. Wind Velocity", fontsize=11.0, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax2.set_xlabel("Wind Speed (m/s)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax2.set_ylabel("Rainfall Rate (mm/h)", fontsize=9.0, color=TEXT_MUTED, labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.1, color='#ffffff')
ax2.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# ------------------------------------------------------------------------------
# 5. ROW 2: ATMOSPHERIC PROFILES (TEMPERATURE, HUMIDITY, AIR DENSITY)
# ------------------------------------------------------------------------------
row2_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=master_gs[2], wspace=0.32)

# Plot 3: Temperature Distribution
ax3 = fig.add_subplot(row2_gs[0])
ax3.set_facecolor(BG_PANEL)
ax3.bar(df['station_id'], df['temperature_c'], color=AMBER, alpha=0.75, edgecolor=BORDER_COLOR)
ax3.set_title("Ambient Temperature (°C) by Station", fontsize=10.0, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax3.set_xlabel("Station ID", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax3.set_ylabel("Temperature (°C)", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax3.grid(True, linestyle='--', alpha=0.1, color='#ffffff')
ax3.tick_params(colors=TEXT_MUTED, labelsize=7.5, pad=6)
plt.setp(ax3.get_xticklabels(), rotation=30)

# Plot 4: Relative Humidity
ax4 = fig.add_subplot(row2_gs[1])
ax4.set_facecolor(BG_PANEL)
ax4.plot(df['station_id'], df['relative_humidity_pct'], color=CYAN, marker='o', linewidth=2, markersize=6)
ax4.set_title("Relative Humidity (%) Profile", fontsize=10.0, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax4.set_xlabel("Station ID", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax4.set_ylabel("Humidity (%)", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax4.grid(True, linestyle='--', alpha=0.1, color='#ffffff')
ax4.tick_params(colors=TEXT_MUTED, labelsize=7.5, pad=6)
plt.setp(ax4.get_xticklabels(), rotation=30)

# Plot 5: Air Density
ax5 = fig.add_subplot(row2_gs[2])
ax5.set_facecolor(BG_PANEL)
ax5.hist(df['air_density_kg_m3'], bins=6, color=VIOLET, alpha=0.75, edgecolor='#020617')
ax5.set_title("Air Density Frequency ($kg/m^3$)", fontsize=10.0, fontweight='bold', color=TEXT_LIGHT, pad=14)
ax5.set_xlabel("Air Density ($kg/m^3$)", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax5.set_ylabel("Sample Count", fontsize=8.5, color=TEXT_MUTED, labelpad=10)
ax5.grid(True, linestyle='--', alpha=0.1, color='#ffffff')
ax5.tick_params(colors=TEXT_MUTED, labelsize=8.0, pad=6)

# ------------------------------------------------------------------------------
# 6. EXPORT PRESENTATION SLIDE
# ------------------------------------------------------------------------------
output_filename = "met_route_presentation_slide.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Meteorological routing presentation slide generated: {output_filename}")