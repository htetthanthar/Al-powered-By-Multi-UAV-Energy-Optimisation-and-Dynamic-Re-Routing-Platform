# dashboard.py
import re
import pathlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
UAV_CMD_DIR = ROOT_DIR / "uav-command"


def _build_embedded_engine_js() -> str:
    models_src = (UAV_CMD_DIR / "models.mjs").read_text(encoding="utf-8")
    engine_src = (UAV_CMD_DIR / "engine.mjs").read_text(encoding="utf-8")
    models_clean = re.sub(r"^\s*export\s+", "", models_src, flags=re.MULTILINE)
    engine_clean = re.sub(r"^\s*import\s+.*?;\s*$", "", engine_src, flags=re.MULTILINE)
    engine_clean = re.sub(r"^\s*export\s+", "", engine_clean, flags=re.MULTILINE)
    return (
        "window.__createStratumEmbeddedSim = (function() {\n"
        + models_clean
        + "\n"
        + engine_clean
        + "\nreturn function() { return new Simulation(createModels()); };\n})();"
    )


def _build_stratum_3d_html(embedded_engine_js: str) -> str:
    index_src = (UAV_CMD_DIR / "public" / "index.html").read_text(encoding="utf-8")
    style_src = (UAV_CMD_DIR / "public" / "style.css").read_text(encoding="utf-8")
    app_src = (UAV_CMD_DIR / "public" / "app.js").read_text(encoding="utf-8")
    cesium_cdn = "https://cdn.jsdelivr.net/npm/cesium@1.124.0/Build/Cesium"

    worker_shim = """
    const _OrigBlob = window.Blob;
    window.Blob = function(parts, opts) {
      if (Array.isArray(parts) && typeof parts[0] === 'string') {
        parts = parts.map(p => typeof p === 'string'
          ? p.replace(/^\\s*import\\s+"([^"]+)";\\s*$/, (_, spec) => {
              const base = String(window.CESIUM_BASE_URL || '').replace(/\\/+$/, '');
              const url = spec.startsWith('http')
                ? spec.replace('/Workers//', '/Workers/')
                : base + '/Workers/' + spec.replace(/^Workers\\/+/, '').replace(/\\.js$/, '') + '.js';
              return 'const __q=[];const __h=e=>__q.push(e);self.onmessage=__h;import("' + url + '").then(()=>{if(self.onmessage!==__h){for(const e of __q)self.onmessage(e);}__q.length=0;});';
            })
          : p
        );
      }
      return new _OrigBlob(parts, opts);
    };
    """

    sync_patch = """
    <script>
    if (window.Cesium && window.Cesium.HeightmapTerrainData) {
      window.Cesium.HeightmapTerrainData.prototype.createMesh = function(options) {
        return Promise.resolve(this._createMeshSync(options));
      };
    }
    </script>
    """

    html = index_src.replace(
        '<link rel="stylesheet" href="/cesium/Widgets/widgets.css"><link rel="stylesheet" href="/style.css">',
        f'<link rel="stylesheet" href="{cesium_cdn}/Widgets/widgets.css"><style>{style_src}</style>',
    )
    html = html.replace(
        "<script>window.CESIUM_BASE_URL='/cesium/';</script><script src=\"/cesium/Cesium.js\"></script><script type=\"module\" src=\"/app.js\"></script>",
        f"<script>{worker_shim}window.CESIUM_BASE_URL='{cesium_cdn}/';window.__STRATUM_EMBEDDED_SIM__=true;{embedded_engine_js}</script>"
        f'<script src="{cesium_cdn}/Cesium.js"></script>'
        f"{sync_patch}"
        f"<script type=\"module\">{app_src}</script>",
    )
    return html


EMBEDDED_ENGINE_JS = _build_embedded_engine_js()

st.set_page_config(
    page_title="STRATUM — Autonomous UAV Fleet Command",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #090a0d !important;
    }
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #9ca3af !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #f07632 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TACTICAL AIRSPACE RADAR HUD: PRODUCTION OPTIMIZED
# ---------------------------------------------------------
hud_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #090a0d;
            color: #f3f4f6;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow-x: hidden;
            padding: 8px 10px 24px 10px;
        }

        /* Reference Top Telemetry Header Bar */
        .hud-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(21, 23, 28, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 13px 20px;
            margin: 0 0 10px 0;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.55);
            backdrop-filter: blur(16px);
        }
        .hud-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .hud-brand-mark {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, rgba(240, 118, 50, 0.24), rgba(240, 118, 50, 0.06));
            border: 1px solid rgba(240, 118, 50, 0.42);
            color: #f07632;
            font-weight: 800;
            font-size: 15px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
        }
        .hud-title {
            color: #f3f4f6;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .hud-subtitle {
            color: #8e95a5;
            font-size: 11px;
            margin-top: 2px;
            letter-spacing: 0.03em;
        }
        .hud-status-wrap {
            display: flex;
            align-items: center;
            gap: 16px;
            padding-left: 18px;
            border-left: 1px solid rgba(255, 255, 255, 0.08);
        }
        .status-pill-header {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 11px;
            border-radius: 7px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            background: rgba(74, 222, 128, 0.14);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.28);
        }

        /* Platform Health Style Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin: 0 0 10px 0;
        }
        .metric-card {
            position: relative;
            background: rgba(21, 23, 28, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 14px 12px 26px;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(14px);
            overflow: hidden;
        }
        .metric-card.accent-glow {
            background:
                radial-gradient(125% 130% at 8% 95%, rgba(198, 96, 42, 0.42) 0%, rgba(140, 62, 26, 0.16) 48%, rgba(24, 27, 34, 0.95) 85%);
            border-color: rgba(240, 118, 50, 0.42);
        }
        .metric-card::before {
            content: "";
            position: absolute;
            left: 12px;
            top: 15px;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4ade80;
            box-shadow: 0 0 8px rgba(74, 222, 128, 0.8);
        }
        .metric-card.accent-glow::before {
            background: #f07632;
            box-shadow: 0 0 8px rgba(240, 118, 50, 0.85);
        }
        .metric-label {
            font-size: 10px;
            color: #8e95a5;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.06em;
        }
        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 17px;
            font-weight: 700;
            color: #f3f4f6;
            margin-top: 4px;
        }
        .metric-delta {
            font-size: 11px;
            color: #8e95a5;
            font-weight: 500;
            margin-top: 2px;
        }
        .metric-card.accent-glow .metric-delta {
            color: #f7b288;
        }

        /* Map Container */
        #map-container {
            position: relative;
            margin: 0 0 12px 0;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            overflow: hidden;
            height: 490px;
            background: #0d0f14;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.6);
        }
        #map { width: 100%; height: 100%; background: #0d0f14; }

        /* Tactical dark satellite grading + subtle grid overlay */
        .leaflet-tile-pane {
            filter: brightness(0.66) contrast(1.16) saturate(0.70) sepia(0.08);
        }
        #map-container::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 400;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px),
                radial-gradient(circle at 50% 50%, transparent 58%, rgba(9, 10, 13, 0.55) 100%);
            background-size: 56px 56px, 56px 56px, 100% 100%;
        }

        /* Compass Rose Overlay (from reference UI) */
        .map-compass {
            position: absolute;
            top: 14px;
            left: 14px;
            z-index: 1000;
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: rgba(18, 20, 26, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            display: grid;
            place-items: center;
            pointer-events: none;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            font-weight: 700;
            color: #8e95a5;
        }
        .map-compass span { position: absolute; }
        .map-compass .c-n { top: 5px; color: #f07632; }
        .map-compass .c-s { bottom: 5px; }
        .map-compass .c-e { right: 6px; }
        .map-compass .c-w { left: 6px; }
        .map-compass-needle {
            width: 14px;
            height: 14px;
            display: grid;
            place-items: center;
        }

        /* Map Legend Overlay (bottom-left from reference UI) */
        .map-legend-bar {
            position: absolute;
            bottom: 14px;
            left: 14px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 7px 13px;
            border-radius: 10px;
            background: rgba(18, 20, 26, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: #9ca3af;
            pointer-events: none;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        /* Control Group */
        .map-controls-group {
            position: absolute;
            top: 14px;
            right: 14px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(18, 20, 26, 0.90);
            padding: 7px 9px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.09);
            backdrop-filter: blur(14px);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
        }

        .ctrl-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', sans-serif;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
            padding: 8px 13px;
            border-radius: 8px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.16s ease;
            outline: none;
        }

        .reset-btn {
            background: rgba(255, 255, 255, 0.05);
            color: #e5e7eb;
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        .reset-btn:hover {
            background: rgba(255, 255, 255, 0.10);
            color: #ffffff;
            border-color: rgba(255, 255, 255, 0.20);
        }

        .zone-btn {
            background: linear-gradient(180deg, rgba(240, 118, 50, 0.24), rgba(196, 86, 30, 0.16));
            color: #ff9d66;
            border: 1px solid rgba(240, 118, 50, 0.45);
        }
        .zone-btn:hover {
            background: linear-gradient(180deg, rgba(240, 118, 50, 0.34), rgba(196, 86, 30, 0.24));
            color: #ffffff;
        }
        .zone-btn.active-mode {
            background: #f07632 !important;
            color: #090a0d !important;
            box-shadow: 0 0 18px rgba(240, 118, 50, 0.65);
        }

        .clear-zone-btn {
            background: rgba(255, 255, 255, 0.04);
            color: #9ca3af;
            border: 1px solid rgba(255, 255, 255, 0.09);
        }
        .clear-zone-btn:hover {
            background: rgba(248, 113, 113, 0.16);
            color: #fca5a5;
            border-color: rgba(248, 113, 113, 0.38);
        }

        .emergency-btn {
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.28) 0%, rgba(185, 28, 28, 0.24) 100%);
            color: #fca5a5;
            border: 1px solid rgba(248, 113, 113, 0.45);
            letter-spacing: 0.03em;
        }
        .emergency-btn:hover {
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.45) 0%, rgba(185, 28, 28, 0.38) 100%);
            color: #ffffff;
            transform: translateY(-1px);
        }

        #draw-hint {
            display: none;
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(240, 118, 50, 0.95);
            color: #090a0d;
            font-weight: 800;
            font-size: 11px;
            padding: 8px 18px;
            border-radius: 999px;
            box-shadow: 0 8px 24px rgba(240, 118, 50, 0.55);
            letter-spacing: 0.04em;
        }

        /* Telemetry Panel */
        .telemetry-panel {
            margin: 0;
            background: rgba(21, 23, 28, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
            backdrop-filter: blur(16px);
        }
        .telemetry-title {
            font-size: 11px;
            font-weight: 700;
            color: #8e95a5;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .table-scroll-container {
            max-height: 430px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            background: rgba(13, 15, 19, 0.65);
        }
        .table-scroll-container::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        .table-scroll-container::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.14);
            border-radius: 999px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            color: #f3f4f6;
        }
        th {
            position: sticky;
            top: 0;
            z-index: 10;
            text-align: left;
            padding: 10px 11px;
            color: #8e95a5;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(20, 22, 28, 0.98);
            white-space: nowrap;
        }
        td {
            padding: 8px 11px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #d1d5db;
            font-weight: 500;
            white-space: nowrap;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
        }
        tbody tr:nth-child(even) {
            background-color: rgba(255, 255, 255, 0.018);
        }
        tbody tr:hover {
            background:
                radial-gradient(95% 140% at 6% 90%, rgba(198, 96, 42, 0.20) 0%, rgba(255, 255, 255, 0.03) 75%);
        }

        .drone-id-tag {
            font-family: 'Inter', sans-serif;
            color: #f3f4f6;
            font-weight: 700;
            font-size: 12px;
        }
        .tag-model {
            font-family: 'Inter', sans-serif;
            color: #8e95a5;
            font-size: 11px;
        }
        .tag-ok { color: #4ade80; font-weight: 600; }
        .tag-warn { color: #fb923c; font-weight: 600; }
        .tag-crit { color: #f87171; font-weight: 700; }
        .tag-heading { color: #5eead4; font-weight: 600; }

        /* Status badge inside table */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .status-badge.ok {
            background: rgba(240, 118, 50, 0.16);
            color: #ff9d66;
            border: 1px solid rgba(240, 118, 50, 0.32);
        }
        .status-badge.warn {
            background: rgba(251, 191, 36, 0.14);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.30);
        }
        .status-badge.crit {
            background: rgba(248, 113, 113, 0.16);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.35);
        }

        /* Row Action Button */
        .row-cmd-btn {
            padding: 5px 11px;
            font-family: 'Inter', sans-serif;
            font-size: 10.5px;
            font-weight: 700;
            border-radius: 7px;
            cursor: pointer;
            outline: none;
            transition: all 0.15s ease;
        }
        .btn-rtl-recall {
            background: rgba(248, 113, 113, 0.15);
            color: #fca5a5;
            border: 1px solid rgba(248, 113, 113, 0.36);
        }
        .btn-rtl-recall:hover {
            background: rgba(239, 68, 68, 0.35);
            color: #ffffff;
            border-color: rgba(248, 113, 113, 0.65);
        }
        .btn-rtl-resume {
            background: rgba(74, 222, 128, 0.16);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.36);
        }
        .btn-rtl-resume:hover {
            background: rgba(74, 222, 128, 0.32);
            color: #ffffff;
        }

        /* Reference Map Drone Marker (Circular dark badge + quadcopter icon + callsign pill) */
        .custom-drone-wrap {
            background: transparent !important;
            border: none !important;
        }
        .drone-node-wrap {
            position: relative;
            width: 34px;
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .drone-badge-ring {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: rgba(20, 23, 29, 0.94);
            border: 1.5px solid #f07632;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.65), 0 0 10px rgba(240, 118, 50, 0.28);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .drone-icon {
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transform-origin: center center;
            will-change: transform;
        }
        .drone-callsign-pill {
            position: absolute;
            left: 34px;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(17, 19, 24, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f3f4f6;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9.5px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 5px;
            white-space: nowrap;
            pointer-events: none;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.45);
        }

        /* Dark Leaflet popups & zoom controls */
        .leaflet-popup-content-wrapper, .leaflet-popup-tip {
            background: rgba(21, 23, 28, 0.96) !important;
            color: #f3f4f6 !important;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 10px !important;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.65) !important;
        }
        .leaflet-bar a {
            background-color: rgba(21, 23, 28, 0.92) !important;
            color: #e5e7eb !important;
            border-color: rgba(255, 255, 255, 0.10) !important;
        }
        .leaflet-bar a:hover {
            background-color: rgba(34, 38, 46, 0.96) !important;
            color: #f07632 !important;
        }
    </style>
</head>
<body>

    <div class="hud-header">
        <div class="hud-brand">
            <div class="hud-brand-mark">❖</div>
            <div>
                <div class="hud-title">AUTONOMOUS UAV FLEET MISSION COMMAND & SENSOR RADAR</div>
                <div class="hud-subtitle">16-UNIT INLAND AIRSPACE TELEMETRY • BMS THERMAL PROFILING • INDIVIDUAL RECALL BUS</div>
            </div>
        </div>
        <div class="hud-status-wrap">
            <div style="text-align: right;">
                <div id="fleet-status-badge" class="status-pill-header">▶ ALL SYSTEMS NOMINAL</div>
                <div style="font-size: 10.5px; color: #8e95a5; margin-top: 4px;">SMT GEOFENCE & THERMAL INVARIANT: SAT</div>
            </div>
        </div>
    </div>

    <div class="metric-grid">
        <div class="metric-card accent-glow">
            <div class="metric-label">Airborne Fleet</div>
            <div class="metric-val" id="metric-fleet">16 Units</div>
            <div class="metric-delta">Singapore Inland 100%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">C2 Data Link</div>
            <div class="metric-val">99.8%</div>
            <div class="metric-delta">MAVLink 2.0 (AES-256)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fleet Avg SoC</div>
            <div class="metric-val" id="metric-soc">83.2%</div>
            <div class="metric-delta" id="metric-soc-delta">Active Balance OK</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg BMS Temp</div>
            <div class="metric-val" id="metric-temp">34.8 °C</div>
            <div class="metric-delta">Nominal Thermal Range</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Airspace Barriers</div>
            <div class="metric-val" id="metric-nfz-count">3 Active NFZs</div>
            <div class="metric-delta">Geofence Enforced</div>
        </div>
    </div>

    <div id="map-container">
        <div class="map-compass">
            <span class="c-n">N</span>
            <span class="c-e">E</span>
            <span class="c-s">S</span>
            <span class="c-w">W</span>
            <div class="map-compass-needle">
                <svg width="12" height="12" viewBox="0 0 24 24">
                    <path d="M12 2L16 12H8L12 2Z" fill="#f07632"/>
                    <path d="M12 22L8 12H16L12 22Z" fill="#6b7280"/>
                </svg>
            </div>
        </div>
        <div class="map-controls-group">
            <button id="btn-draw" class="ctrl-btn zone-btn" onclick="toggleDrawMode()">➕ CREATE RESTRICTED ZONE</button>
            <button class="ctrl-btn clear-zone-btn" onclick="clearCustomZones()">🗑️ CLEAR CUSTOM</button>
            <button class="ctrl-btn reset-btn" onclick="resetAllMissions()">🔄 RESET ALL</button>
            <button class="ctrl-btn emergency-btn" onclick="triggerGlobalEmergencyRTL()">🚨 GLOBAL RTL</button>
        </div>
        <div class="map-legend-bar">
            <span class="legend-item"><span class="legend-dot" style="background:#f07632;"></span> DRONES</span>
            <span class="legend-item"><span class="legend-dot" style="background:#5eead4;"></span> LIDAR / RANGE RINGS</span>
            <span class="legend-item"><span class="legend-dot" style="background:#f87171;"></span> RESTRICTED NFZ</span>
            <span class="legend-item"><span class="legend-dot" style="background:#4ade80;"></span> CENTRAL DEPOT</span>
        </div>
        <div id="draw-hint">🎯 CLICK ANYWHERE ON MAP TO PLACE CUSTOM RESTRICTED GEOFENCE</div>
        <div id="map"></div>
    </div>

    <div class="telemetry-panel">
        <div class="telemetry-title">
            <span>📡 LIVE SENSOR TELEMETRY BUS, ESC/BMS THERMALS & AVIONICS INVARIANTS (16 UNITS)</span>
            <span style="font-size:11px; color:#8e95a5; text-transform:none; letter-spacing:0;">Click [🚨 RECALL] on any UAV to initiate individual emergency landing</span>
        </div>
        <div class="table-scroll-container">
            <table>
                <thead>
                    <tr>
                        <th>ACTION</th>
                        <th>CALLSIGN</th>
                        <th>AIRFRAME MODEL</th>
                        <th>STATUS</th>
                        <th>BMS/ESC TEMP</th>
                        <th>HEADING</th>
                        <th>GNSS FIX</th>
                        <th>ALT (AGL)</th>
                        <th>AIRSPEED</th>
                        <th>LIDAR RANGE</th>
                        <th>OBSTACLE CLEAR</th>
                        <th>OPTICAL FLOW</th>
                        <th>BATTERY SOC</th>
                    </tr>
                </thead>
                <tbody id="telemetry-body"></tbody>
            </table>
        </div>
    </div>

    <script>
        function calculateBearing(lat1, lon1, lat2, lon2) {
            const radLat1 = lat1 * Math.PI / 180.0;
            const radLat2 = lat2 * Math.PI / 180.0;
            const dLon = (lon2 - lon1) * Math.PI / 180.0;

            const y = Math.sin(dLon) * Math.cos(radLat2);
            const x = Math.cos(radLat1) * Math.sin(radLat2) - 
                      Math.sin(radLat1) * Math.cos(radLat2) * Math.cos(dLon);

            const bearingRad = Math.atan2(y, x);
            return (bearingRad * 180.0 / Math.PI + 360.0) % 360.0;
        }

        function lerpAngle(current, target, factor) {
            let diff = (target - current) % 360;
            if (diff < -180) diff += 360;
            if (diff > 180) diff -= 360;
            return (current + diff * factor + 360) % 360;
        }

        const map = L.map('map', {
            center: [1.3520, 103.8200],
            zoom: 11.5,
            zoomControl: false,
            attributionControl: false
        });

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19
        }).addTo(map);

        L.control.zoom({ position: 'bottomright' }).addTo(map);

        const DEPOT = { lat: 1.3000, lon: 103.8000, name: "Central Hub Depot" };

        const depotIcon = L.divIcon({
            className: 'depot-marker',
            html: '<div style="background:#4ade80; width:16px; height:16px; border-radius:50%; border:2px solid #15171c; box-shadow:0 0 14px #4ade80;"></div>',
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });
        L.marker([DEPOT.lat, DEPOT.lon], { icon: depotIcon }).addTo(map).bindPopup("<b>Logistics Central Depot Hub (Queenstown)</b>");

        let isDrawMode = false;
        const SYNC_API_BASE = "http://127.0.0.1:4174";
        let syncToken = null;
        let syncOnline = false;
        let zoneLayers = new Map();
        let zoneHash = "";

        function syncNFZLayers(zones) {
            const nextHash = JSON.stringify(zones.map(z => [z.id, z.lat, z.lon, z.radius]));
            if (nextHash === zoneHash) return;
            zoneHash = nextHash;

            zoneLayers.forEach(layer => map.removeLayer(layer));
            zoneLayers.clear();

            zones.forEach(nfz => {
                const isCustom = !!nfz.isCustom;
                const color = isCustom ? '#f07632' : '#f87171';
                const layer = L.circle([nfz.lat, nfz.lon], {
                    radius: nfz.radius,
                    color: color,
                    weight: 1.8,
                    dashArray: '6, 6',
                    fillColor: color,
                    fillOpacity: isCustom ? 0.22 : 0.16
                }).addTo(map).bindPopup(`<b>${nfz.name}</b><br>Radius: ${nfz.radius}m (+500m buffer)<br>Synchronized Airspace Exclusion`);
                zoneLayers.set(nfz.id || nfz.name, layer);
            });

            const elNfz = document.getElementById('metric-nfz-count');
            if (elNfz) elNfz.innerText = `${zones.length} Active NFZs`;
        }

        async function sendSyncCommand(action, payload = {}) {
            if (syncOnline && syncToken) {
                try {
                    const resp = await fetch(`${SYNC_API_BASE}/api/control`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-UAV-Token': syncToken
                        },
                        body: JSON.stringify({ action, ...payload })
                    });
                    if (resp.ok) {
                        const snap = await resp.json();
                        applyAuthoritativeSnapshot(snap, false);
                        return;
                    }
                } catch (err) {
                    console.warn("Sync command error, falling back to embedded engine:", err);
                }
            }
            if (window.__localRadarSim) {
                try {
                    window.__localRadarSim.command({ action, ...payload });
                    applyAuthoritativeSnapshot(window.__localRadarSim.snapshot(), true);
                } catch (err) {
                    console.warn("Embedded sim command error:", err);
                }
            }
        }

        function toggleDrawMode() {
            isDrawMode = !isDrawMode;
            const btn = document.getElementById('btn-draw');
            const hint = document.getElementById('draw-hint');

            if (isDrawMode) {
                btn.classList.add('active-mode');
                btn.innerText = "✖ CANCEL CREATION";
                hint.style.display = "block";
                map.getContainer().style.cursor = "crosshair";
            } else {
                btn.classList.remove('active-mode');
                btn.innerText = "➕ CREATE RESTRICTED ZONE";
                hint.style.display = "none";
                map.getContainer().style.cursor = "";
            }
        }

        map.on('click', function(e) {
            if (!isDrawMode) return;
            sendSyncCommand('add_zone', {
                lat: e.latlng.lat,
                lon: e.latlng.lng,
                radius: 1200
            });
            toggleDrawMode();
        });

        function clearCustomZones() {
            sendSyncCommand('clear_zones');
        }

        // ---------------------------------------------------------
        // 16 UAV FLEET CONFIGURATION (SYNCHRONIZED WITH STRATUM 3D)
        // ---------------------------------------------------------
        let globalEmergencyRTL = false;

        const fleet = [
            { id: "UAV-ALPHA-01", model: "Matrice 350 RTK", lat: 1.3000, lon: 103.8000, targetLat: 1.3400, targetLon: 103.7050, currentBearing: 0.0, targetBearing: 0.0, alt_m: 48.2, speed_kmh: 64.8, soc_pct: 89.5, temp_c: 32.4, lidar_range_m: 45.0, obstacle_clear_m: 18.0, optical_flow: "99% (Locked)", gnss_fix: "RTK FIXED (14mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-BRAVO-02", model: "EVTOL Long-Range", lat: 1.3000, lon: 103.8000, targetLat: 1.4250, targetLon: 103.8350, currentBearing: 0.0, targetBearing: 0.0, alt_m: 55.0, speed_kmh: 64.8, soc_pct: 88.1, temp_c: 34.1, lidar_range_m: 50.0, obstacle_clear_m: 18.0, optical_flow: "95% (Tracking)", gnss_fix: "RTK FIXED (18mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-CHARLIE-03", model: "Heavy-Lift Quad", lat: 1.3000, lon: 103.8000, targetLat: 1.3532, targetLon: 103.9450, currentBearing: 0.0, targetBearing: 0.0, alt_m: 42.6, speed_kmh: 64.8, soc_pct: 94.2, temp_c: 31.8, lidar_range_m: 35.0, obstacle_clear_m: 18.0, optical_flow: "98% (Locked)", gnss_fix: "RTK FIXED (11mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-DELTA-04", model: "Inspection Hexa", lat: 1.3000, lon: 103.8000, targetLat: 1.2820, targetLon: 103.8600, currentBearing: 0.0, targetBearing: 0.0, alt_m: 38.0, speed_kmh: 64.8, soc_pct: 86.7, temp_c: 36.2, lidar_range_m: 40.0, obstacle_clear_m: 18.0, optical_flow: "92% (Tracking)", gnss_fix: "RTK FIXED (22mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-ECHO-05", model: "Quantum VTOL V2", lat: 1.3000, lon: 103.8000, targetLat: 1.3880, targetLon: 103.7620, currentBearing: 0.0, targetBearing: 0.0, alt_m: 58.5, speed_kmh: 64.8, soc_pct: 91.8, temp_c: 33.0, lidar_range_m: 55.0, obstacle_clear_m: 18.0, optical_flow: "99% (Locked)", gnss_fix: "RTK FIXED (12mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-FOXTROT-06", model: "Falcon Express", lat: 1.3000, lon: 103.8000, targetLat: 1.4050, targetLon: 103.9050, currentBearing: 0.0, targetBearing: 0.0, alt_m: 50.5, speed_kmh: 64.8, soc_pct: 92.4, temp_c: 35.5, lidar_range_m: 48.0, obstacle_clear_m: 18.0, optical_flow: "96% (Tracking)", gnss_fix: "RTK FIXED (15mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-GOLF-07", model: "Solar-Endurance 6S", lat: 1.3000, lon: 103.8000, targetLat: 1.3500, targetLon: 103.8500, currentBearing: 0.0, targetBearing: 0.0, alt_m: 57.0, speed_kmh: 64.8, soc_pct: 96.0, temp_c: 30.5, lidar_range_m: 42.0, obstacle_clear_m: 18.0, optical_flow: "97% (Locked)", gnss_fix: "RTK FIXED (13mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-HOTEL-08", model: "Urban Med-Courier", lat: 1.3000, lon: 103.8000, targetLat: 1.3120, targetLon: 103.9300, currentBearing: 0.0, targetBearing: 0.0, alt_m: 44.0, speed_kmh: 64.8, soc_pct: 89.3, temp_c: 37.2, lidar_range_m: 38.0, obstacle_clear_m: 18.0, optical_flow: "94% (Tracking)", gnss_fix: "RTK FIXED (19mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-INDIA-09", model: "Scout Quad V3", lat: 1.3000, lon: 103.8000, targetLat: 1.3200, targetLon: 103.7550, currentBearing: 0.0, targetBearing: 0.0, alt_m: 40.0, speed_kmh: 64.8, soc_pct: 88.0, temp_c: 32.8, lidar_range_m: 40.0, obstacle_clear_m: 18.0, optical_flow: "98% (Locked)", gnss_fix: "RTK FIXED (12mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-JULIETT-10", model: "SkyRanger G2", lat: 1.3000, lon: 103.8000, targetLat: 1.4380, targetLon: 103.7880, currentBearing: 0.0, targetBearing: 0.0, alt_m: 56.0, speed_kmh: 64.8, soc_pct: 91.4, temp_c: 33.7, lidar_range_m: 52.0, obstacle_clear_m: 18.0, optical_flow: "96% (Tracking)", gnss_fix: "RTK FIXED (16mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-KILO-11", model: "AeroPulse X4", lat: 1.3000, lon: 103.8000, targetLat: 1.3820, targetLon: 103.8480, currentBearing: 0.0, targetBearing: 0.0, alt_m: 46.0, speed_kmh: 64.8, soc_pct: 89.0, temp_c: 34.6, lidar_range_m: 44.0, obstacle_clear_m: 18.0, optical_flow: "97% (Locked)", gnss_fix: "RTK FIXED (14mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-LIMA-12", model: "LogiCopter 8X", lat: 1.3000, lon: 103.8000, targetLat: 1.3720, targetLon: 103.8880, currentBearing: 0.0, targetBearing: 0.0, alt_m: 52.0, speed_kmh: 64.8, soc_pct: 90.2, temp_c: 31.9, lidar_range_m: 46.0, obstacle_clear_m: 18.0, optical_flow: "99% (Locked)", gnss_fix: "RTK FIXED (10mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-MIKE-13", model: "SurveyWing RTK", lat: 1.3000, lon: 103.8000, targetLat: 1.3350, targetLon: 103.7420, currentBearing: 0.0, targetBearing: 0.0, alt_m: 43.0, speed_kmh: 64.8, soc_pct: 87.1, temp_c: 33.2, lidar_range_m: 41.0, obstacle_clear_m: 18.0, optical_flow: "95% (Tracking)", gnss_fix: "RTK FIXED (15mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-NOVEMBER-14", model: "MedEvac Hornet", lat: 1.3000, lon: 103.8000, targetLat: 1.2780, targetLon: 103.8220, currentBearing: 0.0, targetBearing: 0.0, alt_m: 39.0, speed_kmh: 64.8, soc_pct: 86.8, temp_c: 36.8, lidar_range_m: 39.0, obstacle_clear_m: 18.0, optical_flow: "93% (Tracking)", gnss_fix: "RTK FIXED (20mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-OSCAR-15", model: "StealthGrid 4S", lat: 1.3000, lon: 103.8000, targetLat: 1.3220, targetLon: 103.8750, currentBearing: 0.0, targetBearing: 0.0, alt_m: 47.0, speed_kmh: 64.8, soc_pct: 88.5, temp_c: 34.0, lidar_range_m: 43.0, obstacle_clear_m: 18.0, optical_flow: "98% (Locked)", gnss_fix: "RTK FIXED (13mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] },
            { id: "UAV-PAPA-16", model: "Titan Cargo Heavy", lat: 1.3000, lon: 103.8000, targetLat: 1.3820, targetLon: 103.9620, currentBearing: 0.0, targetBearing: 0.0, alt_m: 53.0, speed_kmh: 64.8, soc_pct: 92.0, temp_c: 32.1, lidar_range_m: 47.0, obstacle_clear_m: 18.0, optical_flow: "99% (Locked)", gnss_fix: "RTK FIXED (11mm)", isIndividualRTL: false, status: "OUTBOUND", rawStatus: "EN ROUTE", routePoints: [] }
        ];

        // ---------------------------------------------------------
        // INITIALIZE STATIC TABLE DOM & POPUPS ONCE
        // ---------------------------------------------------------
        const tbody = document.getElementById('telemetry-body');
        let tableRowsHtml = "";

        fleet.forEach(drone => {
            tableRowsHtml += `
                <tr id="row-${drone.id}">
                    <td>
                        <button id="btn-${drone.id}" class="row-cmd-btn btn-rtl-recall" onclick="window.toggleIndividualDroneRTL('${drone.id}')">🚨 RECALL</button>
                    </td>
                    <td class="drone-id-tag">${drone.id}</td>
                    <td class="tag-model">${drone.model}</td>
                    <td><span id="status-${drone.id}" class="status-badge ok">• ${drone.status}</span></td>
                    <td id="temp-${drone.id}" class="tag-ok">${drone.temp_c.toFixed(1)} °C</td>
                    <td id="heading-${drone.id}" class="tag-heading">${drone.currentBearing.toFixed(0)}°</td>
                    <td class="tag-ok">${drone.gnss_fix}</td>
                    <td id="alt-${drone.id}">${drone.alt_m.toFixed(1)} m</td>
                    <td id="speed-${drone.id}">${drone.speed_kmh.toFixed(1)} km/h</td>
                    <td>${drone.lidar_range_m} m</td>
                    <td id="clear-${drone.id}">18.0 m (Clear)</td>
                    <td>${drone.optical_flow}</td>
                    <td id="soc-${drone.id}" class="tag-ok">${drone.soc_pct.toFixed(1)}%</td>
                </tr>
            `;
        });
        tbody.innerHTML = tableRowsHtml;

        // Initialize Map Elements & Static Popups (Reference circular badge + dashed range rings)
        fleet.forEach((drone, idx) => {
            const isMintAccent = idx % 3 === 1;
            const ringColor = isMintAccent ? '#5eead4' : '#f07632';

            drone.polyline = L.polyline([[drone.lat, drone.lon], [drone.targetLat, drone.targetLon]], {
                color: ringColor,
                weight: 2,
                dashArray: '5, 6',
                opacity: 0.78
            }).addTo(map);

            drone.lidarCircle = L.circle([drone.lat, drone.lon], {
                radius: drone.lidar_range_m * 9.5,
                color: ringColor,
                weight: 1.4,
                dashArray: '4, 6',
                fillColor: ringColor,
                fillOpacity: 0.07
            }).addTo(map);

            const shortLabel = drone.id.replace('UAV-', '');
            const iconHtml = `
                <div class="drone-node-wrap">
                    <div id="ring-${drone.id}" class="drone-badge-ring" style="border-color:${ringColor};">
                        <div id="icon-${drone.id}" class="drone-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                <circle cx="6" cy="6" r="3" stroke="${ringColor}" stroke-width="1.8"/>
                                <circle cx="18" cy="6" r="3" stroke="${ringColor}" stroke-width="1.8"/>
                                <circle cx="6" cy="18" r="3" stroke="${ringColor}" stroke-width="1.8"/>
                                <circle cx="18" cy="18" r="3" stroke="${ringColor}" stroke-width="1.8"/>
                                <path d="M8 8L16 16M16 8L8 16" stroke="#e5e7eb" stroke-width="1.8" stroke-linecap="round"/>
                                <path d="M12 3L14.5 9H9.5L12 3Z" fill="${ringColor}"/>
                            </svg>
                        </div>
                    </div>
                    <span class="drone-callsign-pill">${shortLabel}</span>
                </div>
            `;
            drone.marker = L.marker([drone.lat, drone.lon], {
                icon: L.divIcon({
                    className: 'custom-drone-wrap',
                    html: iconHtml,
                    iconSize: [34, 34],
                    iconAnchor: [17, 17]
                })
            }).addTo(map);

            drone.marker.bindPopup(`<div id="popup-${drone.id}" style="font-size:12px; min-width:180px;">Loading...</div>`);
        });

        function updateDroneStyle(drone) {
            const inRTL = drone.rawStatus === 'RETURNING' || drone.isIndividualRTL;
            const isHold = drone.rawStatus === 'HOLDING';
            const btn = document.getElementById(`btn-${drone.id}`);
            const ringEl = document.getElementById(`ring-${drone.id}`);

            if (inRTL || drone.rawStatus === 'LANDED') {
                drone.polyline.setStyle({ color: '#f87171', dashArray: null, weight: 2.6 });
                drone.lidarCircle.setStyle({ color: '#f87171', fillColor: '#f87171' });
                if (btn) {
                    btn.className = "row-cmd-btn btn-rtl-resume";
                    btn.innerText = "🔄 RESUME";
                }
                if (ringEl) {
                    ringEl.style.borderColor = '#f87171';
                    ringEl.style.boxShadow = '0 4px 12px rgba(0,0,0,0.65), 0 0 12px rgba(248, 113, 113, 0.55)';
                }
            } else if (isHold) {
                drone.polyline.setStyle({ color: '#fbbf24', dashArray: '5, 6', weight: 2 });
                drone.lidarCircle.setStyle({ color: '#fbbf24', fillColor: '#fbbf24' });
                if (btn) {
                    btn.className = "row-cmd-btn btn-rtl-resume";
                    btn.innerText = "▶ RESUME";
                }
                if (ringEl) {
                    ringEl.style.borderColor = '#fbbf24';
                    ringEl.style.boxShadow = '0 4px 12px rgba(0,0,0,0.65), 0 0 10px rgba(251, 191, 36, 0.4)';
                }
            } else {
                const idx = fleet.indexOf(drone);
                const baseColor = (idx % 3 === 1) ? '#5eead4' : '#f07632';
                drone.polyline.setStyle({ color: baseColor, dashArray: '5, 6', weight: 2 });
                drone.lidarCircle.setStyle({ color: baseColor, fillColor: baseColor });
                if (btn) {
                    btn.className = "row-cmd-btn btn-rtl-recall";
                    btn.innerText = "🚨 RECALL";
                }
                if (ringEl) {
                    ringEl.style.borderColor = baseColor;
                    ringEl.style.boxShadow = '0 4px 12px rgba(0,0,0,0.65), 0 0 10px rgba(240, 118, 50, 0.28)';
                }
            }
        }

        // ---------------------------------------------------------
        // BIDIRECTIONAL CONTROLLER FUNCTIONS (SYNCED WITH STRATUM)
        // ---------------------------------------------------------
        window.toggleIndividualDroneRTL = function(droneId) {
            const drone = fleet.find(d => d.id === droneId);
            if (!drone) return;
            const shouldResume = (drone.rawStatus === 'RETURNING' || drone.rawStatus === 'HOLDING' || drone.rawStatus === 'LANDED');
            sendSyncCommand(shouldResume ? 'resume' : 'return', { id: droneId });
        };

        function triggerGlobalEmergencyRTL() {
            sendSyncCommand('return_all');
        }

        function resetAllMissions() {
            sendSyncCommand('reset');
        }

        // ---------------------------------------------------------
        // AUTHORITATIVE SNAPSHOT RECEIVER
        // ---------------------------------------------------------
        function applyAuthoritativeSnapshot(snap, isLocalFallback = false) {
            if (!snap || !Array.isArray(snap.drones)) return;
            if (!isLocalFallback) syncOnline = true;
            globalEmergencyRTL = !!snap.globalEmergencyRTL;

            const badge = document.getElementById('fleet-status-badge');
            if (badge) {
                if (globalEmergencyRTL) {
                    badge.innerText = "● GLOBAL EMERGENCY RTL ACTIVE";
                    badge.style.color = "#f87171";
                    badge.style.background = "rgba(248, 113, 113, 0.16)";
                    badge.style.borderColor = "rgba(248, 113, 113, 0.36)";
                } else if (!snap.running) {
                    badge.innerText = "❚❚ SIMULATION PAUSED IN STRATUM 3D";
                    badge.style.color = "#fbbf24";
                    badge.style.background = "rgba(251, 191, 36, 0.14)";
                    badge.style.borderColor = "rgba(251, 191, 36, 0.32)";
                } else {
                    badge.innerText = isLocalFallback
                        ? "▶ ALL SYSTEMS NOMINAL • AUTONOMOUS GNN+MLP ENGINE"
                        : "▶ ALL SYSTEMS NOMINAL • LIVE 2D/3D SYNC";
                    badge.style.color = "#4ade80";
                    badge.style.background = "rgba(74, 222, 128, 0.14)";
                    badge.style.borderColor = "rgba(74, 222, 128, 0.28)";
                }
            }

            if (Array.isArray(snap.zones)) {
                syncNFZLayers(snap.zones);
            }

            snap.drones.forEach((srv, idx) => {
                const drone = fleet.find(d => d.id === srv.id) || fleet[idx];
                if (!drone) return;

                drone.targetLat = srv.position.lat;
                drone.targetLon = srv.position.lon;
                if (Math.hypot(drone.targetLat - drone.lat, drone.targetLon - drone.lon) > 0.03) {
                    drone.lat = drone.targetLat;
                    drone.lon = drone.targetLon;
                }
                drone.targetBearing = Number(srv.heading || 0);
                drone.rawStatus = srv.status || "EN ROUTE";
                drone.status = srv.tacticalStatus || srv.status || "CRUISE";
                drone.isIndividualRTL = !!srv.isIndividualRTL || srv.status === 'RETURNING';
                drone.alt_m = Number(srv.alt_m ?? srv.altitude ?? drone.alt_m);
                drone.speed_kmh = Number(srv.speed_kmh ?? (srv.speed * 3.6) ?? drone.speed_kmh);
                drone.soc_pct = Number(srv.soc_pct ?? srv.battery ?? drone.soc_pct);
                drone.temp_c = Number(srv.temp_c ?? drone.temp_c);
                drone.obstacle_clear_m = Number(srv.obstacle_clear_m ?? 18.0);
                if (srv.route && Array.isArray(srv.route.points)) {
                    const legIdx = Math.min(srv.leg || 0, Math.max(0, srv.route.points.length - 1));
                    drone.routePoints = srv.route.points.slice(legIdx + 1).map(p => [p.lat, p.lon]);
                }

                updateDroneStyle(drone);

                const inEmergency = drone.rawStatus === 'RETURNING' || drone.rawStatus === 'LANDED';
                const elStatus = document.getElementById(`status-${drone.id}`);
                if (elStatus) {
                    elStatus.innerText = `• ${drone.status}`;
                    elStatus.className = inEmergency
                        ? 'status-badge crit'
                        : ((drone.status === 'GEOFENCE_AVOIDING' || drone.status === 'HOLDING') ? 'status-badge warn' : 'status-badge ok');
                }

                const elTemp = document.getElementById(`temp-${drone.id}`);
                if (elTemp) {
                    elTemp.innerText = `${drone.temp_c.toFixed(1)} °C`;
                    elTemp.className = drone.temp_c >= 42.0 ? 'tag-crit' : (drone.temp_c >= 37.0 ? 'tag-warn' : 'tag-ok');
                }

                const elHeading = document.getElementById(`heading-${drone.id}`);
                if (elHeading) elHeading.innerText = `${drone.targetBearing.toFixed(0)}°`;

                const elAlt = document.getElementById(`alt-${drone.id}`);
                if (elAlt) elAlt.innerText = `${drone.alt_m.toFixed(1)} m`;

                const elSpeed = document.getElementById(`speed-${drone.id}`);
                if (elSpeed) elSpeed.innerText = `${drone.speed_kmh.toFixed(1)} km/h`;

                const elClear = document.getElementById(`clear-${drone.id}`);
                if (elClear) elClear.innerText = `${drone.obstacle_clear_m.toFixed(1)} m (Clear)`;

                const elSoc = document.getElementById(`soc-${drone.id}`);
                if (elSoc) {
                    elSoc.innerText = `${drone.soc_pct.toFixed(1)}%`;
                    elSoc.className = drone.soc_pct < 30 ? 'tag-crit' : 'tag-ok';
                }

                if (drone.marker.isPopupOpen()) {
                    drone.marker.setPopupContent(`
                        <div style="font-size:12px; min-width:185px; font-family:'Inter',sans-serif;">
                            <b style="color:#f3f4f6;">${drone.id}</b> <span style="color:#8e95a5;">(${drone.model})</span><br>
                            Status: <b style="color:${inEmergency ? '#f87171' : '#f07632'}">${drone.status}</b><br>
                            Temp: <b>${drone.temp_c.toFixed(1)}°C</b> | SoC: <b>${drone.soc_pct.toFixed(1)}%</b><br>
                            <button style="margin-top:7px; width:100%; padding:5px 8px; font-weight:700; background:${inEmergency ? 'rgba(74,222,128,0.22)' : 'rgba(248,113,113,0.22)'}; color:${inEmergency ? '#4ade80' : '#fca5a5'}; border:1px solid ${inEmergency ? 'rgba(74,222,128,0.45)' : 'rgba(248,113,113,0.45)'}; border-radius:6px; cursor:pointer;" onclick="window.toggleIndividualDroneRTL('${drone.id}')">
                                ${inEmergency ? '🔄 RESUME PATROL' : '🚨 RECALL THIS DRONE'}
                            </button>
                        </div>
                    `);
                }
            });

            if (snap.metrics) {
                const elFleet = document.getElementById('metric-fleet');
                if (elFleet) elFleet.innerText = `${snap.metrics.airborne} / ${snap.drones.length} Units`;
                const elSoc = document.getElementById('metric-soc');
                if (elSoc && snap.metrics.avgSoc !== undefined) elSoc.innerText = `${snap.metrics.avgSoc.toFixed(1)}%`;
                const elTemp = document.getElementById('metric-temp');
                if (elTemp && snap.metrics.avgTemp !== undefined) elTemp.innerText = `${snap.metrics.avgTemp.toFixed(1)} °C`;
            }
        }

        window.__localRadarSim = typeof window.__createStratumEmbeddedSim === 'function'
            ? window.__createStratumEmbeddedSim()
            : null;
        if (window.__localRadarSim) {
            applyAuthoritativeSnapshot(window.__localRadarSim.snapshot(), true);
        }
        setInterval(() => {
            if (!syncOnline && window.__localRadarSim) {
                window.__localRadarSim.tick(0.5);
                applyAuthoritativeSnapshot(window.__localRadarSim.snapshot(), true);
            }
        }, 500);

        async function connectSyncBus() {
            try {
                const resp = await fetch(`${SYNC_API_BASE}/api/state`);
                if (!resp.ok) throw new Error("State service unreachable");
                const initData = await resp.json();
                syncToken = initData.csrf;
                applyAuthoritativeSnapshot(initData, false);

                const es = new EventSource(`${SYNC_API_BASE}/api/events`);
                es.onmessage = (evt) => {
                    try {
                        applyAuthoritativeSnapshot(JSON.parse(evt.data), false);
                    } catch (e) {}
                };
                es.onerror = () => {
                    syncOnline = false;
                };
            } catch (err) {
                syncOnline = false;
                setTimeout(connectSyncBus, 4000);
            }
        }

        connectSyncBus();

        // ---------------------------------------------------------
        // 60Hz VISUAL INTERPOLATION LOOP (LOCKED TO SERVER STATE)
        // ---------------------------------------------------------
        function updateSimulation() {
            fleet.forEach((drone) => {
                drone.lat += (drone.targetLat - drone.lat) * 0.22;
                drone.lon += (drone.targetLon - drone.lon) * 0.22;
                drone.currentBearing = lerpAngle(drone.currentBearing, drone.targetBearing, 0.22);

                const iconEl = document.getElementById(`icon-${drone.id}`);
                if (iconEl) {
                    iconEl.style.transform = `rotate(${drone.currentBearing.toFixed(1)}deg)`;
                }

                drone.marker.setLatLng([drone.lat, drone.lon]);
                drone.lidarCircle.setLatLng([drone.lat, drone.lon]);
                if (drone.routePoints && drone.routePoints.length > 0) {
                    drone.polyline.setLatLngs([[drone.lat, drone.lon], ...drone.routePoints]);
                } else {
                    drone.polyline.setLatLngs([[drone.lat, drone.lon], [drone.targetLat, drone.targetLon]]);
                }
            });

            requestAnimationFrame(updateSimulation);
        }

        requestAnimationFrame(updateSimulation);
    </script>
</body>
</html>
"""

hud_html_with_engine = hud_html.replace(
    "<script>\n        function calculateBearing",
    f"<script>{EMBEDDED_ENGINE_JS}</script>\n    <script>\n        function calculateBearing",
)

tab_radar, tab_3d, tab_hud = st.tabs([
    "📡 2D TACTICAL SENSOR RADAR & BMS TELEMETRY",
    "🌐 STRATUM 3D MULTI-UAV COMMAND & NEURAL SURROGATE STUDIO",
    "⚡ INTEGRATED WEATHER, FLIR THERMAL & VSLAM COMMAND",
])

with tab_radar:
    components.html(hud_html_with_engine, height=1250, scrolling=False)

with tab_3d:
    components.html(_build_stratum_3d_html(EMBEDDED_ENGINE_JS), height=960, scrolling=False)

with tab_hud:
    index_hud_path = ROOT_DIR / "index.html"
    if index_hud_path.exists():
        components.html(index_hud_path.read_text(encoding="utf-8"), height=960, scrolling=True)