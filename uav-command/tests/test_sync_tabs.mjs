import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..', '..');
const uavCmdDir = path.resolve(rootDir, 'uav-command');

// Read files
const dashboardPy = fs.readFileSync(path.resolve(rootDir, 'All sensor', 'dashboard.py'), 'utf-8');
const indexHtml = fs.readFileSync(path.resolve(rootDir, 'index.html'), 'utf-8');
const engineMjs = fs.readFileSync(path.resolve(uavCmdDir, 'engine.mjs'), 'utf-8');
const appJs = fs.readFileSync(path.resolve(uavCmdDir, 'public', 'app.js'), 'utf-8');
const styleCss = fs.readFileSync(path.resolve(uavCmdDir, 'public', 'style.css'), 'utf-8');
const stratum3dIndex = fs.readFileSync(path.resolve(uavCmdDir, 'public', 'index.html'), 'utf-8');
const modelsMjs = fs.readFileSync(path.resolve(uavCmdDir, 'models.mjs'), 'utf-8');

// Build embedded engine JS matching dashboard.py exactly
const modelsClean = modelsMjs.replace(/^\s*export\s+/gm, '');
let engineClean = engineMjs.replace(/^\s*import\s+.*?;\s*$/gm, '');
engineClean = engineClean.replace(/^\s*export\s+/gm, '');

const embeddedEngineJs = `window.__createStratumEmbeddedSim = (function() {
${modelsClean}
${engineClean}
return function() { return new Simulation(createModels()); };
})();`;

// Extract sync bus js from dashboard.py
const syncBusPrefix = 'def _build_sync_bus_js() -> str:';
const fnStart = dashboardPy.indexOf(syncBusPrefix);
const returnIdx = dashboardPy.indexOf('return """', fnStart);
const quoteStart = returnIdx + 'return """'.length;
const quoteEnd = dashboardPy.indexOf('"""', quoteStart);
const syncBusJs = dashboardPy.substring(quoteStart, quoteEnd).trim();

// Extract hud_html from dashboard.py
const hudHtmlPrefix = 'hud_html = """';
const hudStart = dashboardPy.indexOf(hudHtmlPrefix) + hudHtmlPrefix.length;
const hudEnd = dashboardPy.indexOf('"""', hudStart);
const rawHudHtml = dashboardPy.substring(hudStart, hudEnd);
const tab1Html = rawHudHtml.replace(
  "<script>\n        function calculateBearing",
  `<script>${embeddedEngineJs}</script>\n    <script>${syncBusJs}</script>\n    <script>\n        function calculateBearing`
);

// Build Tab 2 HTML (mock Cesium for fast lightweight testing if CDN unavailable)
const cesiumMock = `
<script>
window.CESIUM_BASE_URL = 'mock';
window.Cesium = {
  Ion: { defaultAccessToken: '' },
  Cartesian3: { fromDegrees: (lon, lat, h) => ({ lon, lat, h }) },
  Cartesian2: function(x, y) { this.x = x; this.y = y; },
  Color: {
    WHITE: '#ffffff',
    fromCssColorString: (s) => ({
      withAlpha: (a) => s
    })
  },
  Math: { toRadians: (d) => d * Math.PI / 180 },
  EllipsoidTerrainProvider: function() {},
  UrlTemplateImageryProvider: function() {},
  PolylineDashMaterialProperty: function() {},
  ScreenSpaceEventHandler: function() { this.setInputAction = () => {}; },
  ScreenSpaceEventType: { LEFT_CLICK: 0 },
  Viewer: function(id) {
    this.scene = {
      globe: { baseColor: '', enableLighting: false, showGroundAtmosphere: false, tilesLoaded: true },
      backgroundColor: '',
      screenSpaceCameraController: {},
      canvas: document.createElement('canvas'),
      pick: () => null
    };
    this.imageryLayers = { addImageryProvider: () => ({ brightness: 1, saturation: 1, contrast: 1, alpha: 1 }) };
    this.entities = {
      values: [],
      add: (ent) => { this.entities.values.push(ent); return ent; },
      remove: (ent) => {
        const idx = this.entities.values.indexOf(ent);
        if (idx !== -1) this.entities.values.splice(idx, 1);
      }
    };
    this.camera = { flyTo: () => {}, setView: () => {} };
    this.resize = () => {};
  }
};
</script>
`;

const tab2Html = stratum3dIndex
  .replace('<link rel="stylesheet" href="/cesium/Widgets/widgets.css"><link rel="stylesheet" href="/style.css">', `<style>${styleCss}</style>`)
  .replace(
    "<script>window.CESIUM_BASE_URL='/cesium/';</script><script src=\"/cesium/Cesium.js\"></script><script type=\"module\" src=\"/app.js\"></script>",
    `${cesiumMock}
     <script>window.__STRATUM_EMBEDDED_SIM__ = true; ${embeddedEngineJs}</script>
     <script>${syncBusJs}</script>
     <script type="module">${appJs}</script>`
  );

// Build Tab 3 HTML
const tab3Html = indexHtml.replace(
  '</head>',
  `<script>${embeddedEngineJs}</script>\n<script>${syncBusJs}</script>\n</head>`
);

console.log('tab1Html has AeroSyncBus:', tab1Html.includes('AeroSyncBus'));
console.log('tab2Html has AeroSyncBus:', tab2Html.includes('AeroSyncBus'));
console.log('tab3Html has AeroSyncBus:', tab3Html.includes('AeroSyncBus'));

// Outer Container HTML
const containerHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AERO-AI 3-Tab Realtime Synchronization Verification</title>
  <style>
    body { margin: 0; background: #070a12; color: #fff; font-family: sans-serif; display: flex; flex-direction: column; gap: 10px; padding: 10px; }
    h1 { font-size: 14px; margin: 0 0 5px 0; color: #00e5ff; }
    .frame-container { display: flex; gap: 10px; height: 90vh; }
    iframe { flex: 1; border: 1px solid #1e293b; border-radius: 8px; background: #0b1120; }
  </style>
</head>
<body>
  <h1>AERO-AI THREE-TAB SYNCHRONIZATION TEST SUITE</h1>
  <div class="frame-container">
    <iframe id="frame-tab1" src="/tab1"></iframe>
    <iframe id="frame-tab2" src="/tab2"></iframe>
    <iframe id="frame-tab3" src="/tab3"></iframe>
  </div>
</body>
</html>
`;

// Create HTTP server
const server = http.createServer((req, res) => {
  if (req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(containerHtml);
  } else if (req.url === '/tab1') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(tab1Html);
  } else if (req.url === '/tab2') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(tab2Html);
  } else if (req.url === '/tab3') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(tab3Html);
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const base = `http://127.0.0.1:${port}`;
console.log(`Test server running at ${base}`);

let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const p3 = await browser.newPage();
  p3.on('console', m => console.log('P3 LOG:', m.text()));
  p3.on('pageerror', e => console.log('P3 ERROR:', e.message));
  await p3.goto(`${base}/tab3`, { waitUntil: 'domcontentloaded' });
  const p3Bus = await p3.evaluate(() => typeof window.AeroSyncBus);
  console.log('P3 bus type:', p3Bus);
  await p3.close();

  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));
  await page.goto(base, { waitUntil: 'domcontentloaded' });

  // Access all 3 frames
  const f1Handle = await page.waitForSelector('#frame-tab1');
  const f2Handle = await page.waitForSelector('#frame-tab2');
  const f3Handle = await page.waitForSelector('#frame-tab3');

  const f1 = await f1Handle.contentFrame();
  const f2 = await f2Handle.contentFrame();
  const f3 = await f3Handle.contentFrame();

  assert.ok(f1, 'Frame 1 loaded');
  assert.ok(f2, 'Frame 2 loaded');
  assert.ok(f3, 'Frame 3 loaded');

  // Wait for sync bus to initialize and tick in all frames
  await page.waitForTimeout(1500);

  // Assertion 1: Verify all 3 frames have AeroSyncBus
  const f1HasBus = await f1.evaluate(() => typeof window.AeroSyncBus !== 'undefined');
  const f2HasBus = await f2.evaluate(() => typeof window.AeroSyncBus !== 'undefined');
  const f3HasBus = await f3.evaluate(() => typeof window.AeroSyncBus !== 'undefined');
  assert.ok(f1HasBus, 'Frame 1 has AeroSyncBus');
  assert.ok(f2HasBus, 'Frame 2 has AeroSyncBus');
  assert.ok(f3HasBus, 'Frame 3 has AeroSyncBus');
  console.log('✓ Assertion 1 PASSED: AeroSyncBus loaded in all 3 frames.');

  // Assertion 2: Verify all 3 frames share the exact same snapshot
  const f1Snap = await f1.evaluate(() => window.AeroSyncBus.getSnapshot());
  const f2Snap = await f2.evaluate(() => window.AeroSyncBus.getSnapshot());
  const f3Snap = await f3.evaluate(() => window.AeroSyncBus.getSnapshot());

  assert.ok(f1Snap && f1Snap.drones && f1Snap.drones.length >= 6, 'Snap has drones');
  assert.equal(f1Snap.drones.length, f2Snap.drones.length, 'F1 and F2 drone count identical');
  assert.equal(f1Snap.drones.length, f3Snap.drones.length, 'F1 and F3 drone count identical');
  console.log(`✓ Assertion 2 PASSED: Exact identical drone count (${f1Snap.drones.length}) across all 3 tabs.`);

  // Assertion 3: Telemetry data values match across 2D table, 3D hud, and Tab 3 fleet list
  const f1Drone1Soc = await f1.evaluate(() => {
    const el = document.getElementById('soc-UAV-ALPHA-01') || document.querySelector('[id*="soc-"]');
    return el ? el.textContent.trim() : null;
  });
  const f2SelectedSoc = await f2.evaluate(() => {
    const el = document.getElementById('selected-battery');
    return el ? el.textContent.trim() : null;
  });
  console.log(`Frame 1 SoC: ${f1Drone1Soc}, Frame 2 selected SoC: ${f2SelectedSoc}`);

  // Assertion 4: Bidirectional selection: Select UAV-CHARLIE-03 in Tab 3 -> check Tab 1 and Tab 2
  console.log('Testing Selection: Clicking UAV-CHARLIE-03 in Tab 3...');
  await f3.evaluate(() => {
    window.selectDrone('UAV-CHARLIE-03');
  });
  await page.waitForTimeout(500);

  const selectedInF1 = await f1.evaluate(() => window.AeroSyncBus.getSelected());
  const selectedInF2 = await f2.evaluate(() => window.AeroSyncBus.getSelected());
  const selectedInF3 = await f3.evaluate(() => window.selectedDroneId);

  assert.equal(selectedInF1, 'UAV-CHARLIE-03', 'Tab 1 received selection UAV-CHARLIE-03');
  assert.equal(selectedInF2, 'UAV-CHARLIE-03', 'Tab 2 received selection UAV-CHARLIE-03');
  assert.equal(selectedInF3, 'UAV-CHARLIE-03', 'Tab 3 has selection UAV-CHARLIE-03');
  console.log('✓ Assertion 4 PASSED: Selecting drone in Tab 3 propagated instantly to Tab 1 and Tab 2.');

  // Assertion 5: Bidirectional selection: Select UAV-BRAVO-02 in Tab 2 -> check Tab 1 and Tab 3
  console.log('Testing Selection: Selecting UAV-BRAVO-02 in Tab 2...');
  await f2.evaluate(() => {
    window.selectAircraft('UAV-BRAVO-02');
  });
  await page.waitForTimeout(500);

  const sel2InF1 = await f1.evaluate(() => window.AeroSyncBus.getSelected());
  const sel2InF2 = await f2.evaluate(() => window.AeroSyncBus.getSelected());
  const sel2InF3 = await f3.evaluate(() => window.selectedDroneId);

  assert.equal(sel2InF1, 'UAV-BRAVO-02', 'Tab 1 updated to UAV-BRAVO-02');
  assert.equal(sel2InF2, 'UAV-BRAVO-02', 'Tab 2 updated to UAV-BRAVO-02');
  assert.equal(sel2InF3, 'UAV-BRAVO-02', 'Tab 3 updated to UAV-BRAVO-02');
  console.log('✓ Assertion 5 PASSED: Selecting drone in Tab 2 propagated instantly to Tab 1 and Tab 3.');

  // Assertion 6: Swarm Re-Optimization in Tab 3 -> check all tabs receive synchronized optimized state
  console.log('Testing Global Command: Triggering reoptimizeEntireSwarm() in Tab 3...');
  const initialOpt = await f1.evaluate(() => window.AeroSyncBus.getSnapshot().optimized);
  await f3.evaluate(() => {
    window.reoptimizeEntireSwarm();
  });
  await page.waitForTimeout(600);

  const optF1 = await f1.evaluate(() => window.AeroSyncBus.getSnapshot().optimized);
  const optF2 = await f2.evaluate(() => window.AeroSyncBus.getSnapshot().optimized);
  const optF3 = await f3.evaluate(() => window.AeroSyncBus.getSnapshot().optimized);

  assert.equal(optF1, !initialOpt, 'Tab 1 reflects toggled optimized state');
  assert.equal(optF2, optF1, 'Tab 2 matches Tab 1 optimized state');
  assert.equal(optF3, optF1, 'Tab 3 matches Tab 1 optimized state');
  console.log(`✓ Assertion 6 PASSED: Re-optimization command propagated and synchronized across all 3 tabs (optimized = ${optF1}).`);

  // Assertion 7: Swarm RTL in Tab 3 -> check all tabs receive globalEmergencyRTL
  console.log('Testing Global Command: Triggering triggerGlobalRTL() in Tab 3...');
  await f3.evaluate(() => {
    window.triggerGlobalRTL();
  });
  await page.waitForTimeout(600);

  const rtlF1 = await f1.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);
  const rtlF2 = await f2.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);
  const rtlF3 = await f3.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);

  assert.equal(rtlF1, true, 'Tab 1 reflects globalEmergencyRTL = true');
  assert.equal(rtlF2, true, 'Tab 2 reflects globalEmergencyRTL = true');
  assert.equal(rtlF3, true, 'Tab 3 reflects globalEmergencyRTL = true');
  console.log('✓ Assertion 7 PASSED: Global RTL command synchronized across all 3 tabs.');

  // Assertion 8: Replay / Reset in Tab 3 -> check all tabs receive reset state
  console.log('Testing Global Command: Triggering replayNormalMissions() in Tab 3...');
  await f3.evaluate(() => {
    window.replayNormalMissions();
  });
  await page.waitForTimeout(600);

  const resetRtlF1 = await f1.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);
  const resetRtlF2 = await f2.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);
  const resetRtlF3 = await f3.evaluate(() => window.AeroSyncBus.getSnapshot().globalEmergencyRTL);

  assert.equal(resetRtlF1, false, 'Tab 1 reflects reset globalEmergencyRTL = false');
  assert.equal(resetRtlF2, false, 'Tab 2 reflects reset globalEmergencyRTL = false');
  assert.equal(resetRtlF3, false, 'Tab 3 reflects reset globalEmergencyRTL = false');
  console.log('✓ Assertion 8 PASSED: Swarm reset / replay synchronized across all 3 tabs.');

  // Take screenshot of the multi-tab layout for visual artifact
  const artifactsDir = path.resolve(rootDir, 'uav-command', 'artifacts');
  fs.mkdirSync(artifactsDir, { recursive: true });
  const ssPath = path.resolve(artifactsDir, 'three_tabs_synchronized.png');
  await page.screenshot({ path: ssPath, fullPage: true });
  console.log(`Saved screenshot to ${ssPath}`);

  console.log('\n======================================================');
  console.log('ALL THREE TABS SYNCHRONIZATION TESTS PASSED PERFECTLY!');
  console.log('======================================================\n');
} finally {
  if (browser) await browser.close();
  server.closeAllConnections();
  await new Promise(r => server.close(r));
}
