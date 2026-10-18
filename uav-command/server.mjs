import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname, sep } from 'node:path';
import { randomBytes } from 'node:crypto';
import { createModels } from './models.mjs';
import { Simulation } from './engine.mjs';

const base = fileURLToPath(new URL('.', import.meta.url));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.wasm': 'application/wasm' };
export function createApp(models = createModels(), publicOrigin = process.env.STRATUM_PUBLIC_ORIGIN) {
  if (publicOrigin && (new URL(publicOrigin).origin !== publicOrigin || !publicOrigin.startsWith('https://'))) throw new Error('STRATUM_PUBLIC_ORIGIN must be a canonical HTTPS origin');
  const sim = new Simulation(models), csrf = randomBytes(24).toString('hex'), clients = new Set();
  let last = Date.now();
  const timer = setInterval(() => {
    const now = Date.now(); sim.tick((now - last) / 1000); last = now;
    const payload = `data: ${JSON.stringify(sim.snapshot())}\n\n`;
    for (const client of clients) { if (client.writableLength > 1024 * 1024) { client.destroy(); clients.delete(client); } else client.write(payload); }
  }, 250);
  const isAllowedOrigin = (origin, host) => {
    if (!origin) return true;
    if (origin === `http://${host}` || origin === publicOrigin) return true;
    if (origin === 'null') return true;
    return /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(origin);
  };
  const server = http.createServer(async (req, res) => {
    const host = req.headers.host;
    // Loopback-only service; reject alternate Hosts and cross-origin mutation requests.
    if (!host || !/^(127\.0\.0\.1|localhost):\d+$/.test(host)) { res.writeHead(403); res.end('Local access only'); return; }
    const origin = req.headers.origin;
    const corsHeaders = isAllowedOrigin(origin, host) ? {
      'Access-Control-Allow-Origin': origin || '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-UAV-Token'
    } : {};
    if (req.method === 'OPTIONS') {
      res.writeHead(204, corsHeaders);
      res.end();
      return;
    }
    const url = new URL(req.url, `http://${host}`);
    const json = (status, body) => { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...corsHeaders }); res.end(JSON.stringify(body)); };
    try {
      if (req.method === 'GET' && url.pathname === '/api/state') return json(200, { ...sim.snapshot(), csrf });
      if (req.method === 'GET' && url.pathname === '/api/events') {
        if (clients.size >= 32) return json(503, { error: 'Viewer limit reached' });
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive', ...corsHeaders });
        res.write(`data: ${JSON.stringify(sim.snapshot())}\n\n`); clients.add(res);
        req.on('close', () => clients.delete(res)); return;
      }
      if (req.method === 'POST' && url.pathname === '/api/control') {
        if (req.headers['x-uav-token'] !== csrf || !isAllowedOrigin(origin, host) || !req.headers['content-type']?.startsWith('application/json')) return json(403, { error: 'Invalid origin or command token' });
        let body = '';
        for await (const chunk of req) { body += chunk; if (Buffer.byteLength(body) > 2048) return json(413, { error: 'Command too large' }); }
        try { sim.command(JSON.parse(body)); } catch (error) { return json(400, { error: error.message }); }
        const snap = sim.snapshot();
        const payload = `data: ${JSON.stringify(snap)}\n\n`;
        for (const client of clients) { if (client.writableLength <= 1024 * 1024) client.write(payload); }
        return json(200, snap);
      }
      if (req.method !== 'GET') return json(405, { error: 'Method not allowed' });
      const vendor = url.pathname.startsWith('/cesium/');
      const root = resolve(base, vendor ? 'node_modules/cesium/Build/Cesium' : 'public');
      const name = decodeURIComponent(vendor ? url.pathname.slice(8) : url.pathname === '/' ? 'index.html' : url.pathname.slice(1));
      const path = resolve(root, name);
      if (!path.startsWith(root + sep)) return json(403, { error: 'Forbidden' });
      const content = await readFile(path);
      res.writeHead(200, { 'Content-Type': types[extname(path)] || 'application/octet-stream', 'X-Content-Type-Options': 'nosniff', 'Cache-Control': vendor ? 'public, max-age=86400' : 'no-cache' }); res.end(content);
    } catch (error) { if (!res.headersSent) json(404, { error: 'Resource unavailable' }); else res.end(); }
  });
  server.on('close', () => { clearInterval(timer); for (const c of clients) c.end(); });
  return { server, sim };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4174);
  const { server } = createApp();
  server.listen(port, '127.0.0.1', () => console.log(`UAV COMMAND: http://127.0.0.1:${port} · SIMULATION ONLY · No hardware or database writes`));
  server.on('error', e => { console.error(e.message); process.exitCode = 1; });
}
