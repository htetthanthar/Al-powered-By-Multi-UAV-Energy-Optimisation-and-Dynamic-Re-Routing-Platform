import { graphEmbedding } from './models.mjs';
export const zones = [
  { id: 'NFZ-A', name: 'NFZ-A: Central Catchment Area', lon: 103.82, lat: 1.36, radius: 1250, isCustom: false },
  { id: 'NFZ-B', name: 'NFZ-B: Paya Lebar Air Base Buffer', lon: 103.92, lat: 1.335, radius: 1150, isCustom: false },
  { id: 'NFZ-C', name: 'NFZ-C: Changi Approach Buffer', lon: 103.988, lat: 1.355, radius: 1350, isCustom: false }
];
export const home = { lon: 103.80, lat: 1.30, name: 'Headquarters · Queenstown Central Depot' };
const scale = 111195;
const xy = p => [p.lon * scale * Math.cos(1.34 * Math.PI / 180), p.lat * scale];
export const distance = (a, b) => { const x = xy(a), y = xy(b); return Math.hypot(x[0] - y[0], x[1] - y[1]); };
export function segmentClearance(a, b, z) {
  const [ax, ay] = xy(a), [bx, by] = xy(b), [cx, cy] = xy(z);
  const dx = bx - ax, dy = by - ay;
  const t = Math.max(0, Math.min(1, ((cx - ax) * dx + (cy - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(ax + dx * t - cx, ay + dy * t - cy) - z.radius;
}
export const safe = (a, b, activeZones = zones, buffer = 500) => activeZones.every(z => segmentClearance(a, b, z) >= buffer);
export const grid = Array.from({ length: 35 }, (_, i) => ({ id: i, lon: 103.74 + (i % 7) * .035, lat: 1.28 + Math.floor(i / 7) * .03 }));
export const neighbors = grid.map((a, i) => grid.flatMap((b, j) => i !== j && distance(a, b) < 5700 && safe(a, b) ? [j] : []));
export function weather(nodes, wind) {
  return nodes.map(p => { const w = .5 + .5 * Math.sin(p.lon * 95 + p.lat * 37); return [Math.min(1, wind / 12 * (.6 + .4 * w)), w * .8, .6, .3]; });
}
export function plan(models, start, end, payload, wind, optimized = true, activeZones = zones) {
  const nodes = [...grid, start, end], first = nodes.length - 2, last = nodes.length - 1;
  const features = weather(nodes, wind);
  function runSearch(allowLongTerminalLinks) {
    const links = nodes.map((a, i) => nodes.flatMap((b, j) => {
      if (i === j || !safe(a, b, activeZones, 500)) return [];
      if (distance(a, b) < 5700) return [j];
      if (allowLongTerminalLinks && (i >= grid.length || j >= grid.length)) return [j];
      return [];
    }));
    const hazard = nodes.map((_, i) => Math.max(0, Math.min(1, models.graph.predict(graphEmbedding(features, links, i)))));
    function edge(i, j) {
      const d = distance(nodes[i], nodes[j]);
      // Synthetic easterly wind. A westbound edge has a tailwind.
      const headwind = wind * (nodes[j].lon - nodes[i].lon) * scale / (d || 1);
      let speed = 18, wh = Infinity;
      for (const v of optimized ? [10, 12, 14, 16, 18, 20, 22] : [18]) {
        const ground = v - headwind;
        if (ground < 4) continue;
        const candidate = models.power(v, payload, headwind) * d / ground / 3600;
        if (candidate < wh) { wh = candidate; speed = v; }
      }
      return { speed, wh, distance: d, headwind, seconds: d / Math.max(4, speed - headwind), cost: optimized ? wh * (1 + 1.8 * (hazard[i] + hazard[j]) / 2) : d };
    }
    const costs = nodes.map(() => Infinity), prev = nodes.map(() => -1), visited = new Set(); costs[first] = 0;
    while (visited.size < nodes.length) {
      let u = -1;
      costs.forEach((v, i) => { if (!visited.has(i) && (u < 0 || v < costs[u])) u = i; });
      if (u < 0 || !Number.isFinite(costs[u])) break;
      if (u === last) break;
      visited.add(u);
      for (const v of links[u]) { const alt = costs[u] + edge(u, v).cost; if (alt < costs[v]) { costs[v] = alt; prev[v] = u; } }
    }
    return { costs, prev, edge, hazard };
  }
  let res = runSearch(false);
  if (!Number.isFinite(res.costs[last])) res = runSearch(true);
  if (!Number.isFinite(res.costs[last])) throw new Error('No path clears every example exclusion buffer. Mission remains held.');
  const indices = [last]; while (indices[0] !== first) indices.unshift(res.prev[indices[0]]);
  const legs = indices.slice(1).map((v, i) => ({ ...res.edge(indices[i], v), from: nodes[indices[i]], to: nodes[v] }));
  return { points: indices.map(i => ({ lon: nodes[i].lon, lat: nodes[i].lat })), legs, wh: legs.reduce((s, l) => s + l.wh, 0), seconds: legs.reduce((s, l) => s + l.seconds, 0), distance: legs.reduce((s, l) => s + l.distance, 0), hazard: grid.map((p, i) => ({ ...p, risk: res.hazard[i] })) };
}

export const FLEET_CONFIG = [
  { id: 'UAV-ALPHA-01', legacyId: 'UAV-01', corridorId: 'RT-01', name: 'Matrice 350 RTK', destinationName: 'RT-01 · Jurong West', color: '#65e5c2', alt_m: 48.2, payload: 0.65, soc_pct: 89.5, base_temp: 32.4, lidar_range_m: 45.0, optical_flow: '99% (Locked)', gnss_fix: 'RTK FIXED (14mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3400, lon: 103.7050 }] },
  { id: 'UAV-BRAVO-02', legacyId: 'UAV-02', corridorId: 'RT-02', name: 'EVTOL Long-Range', destinationName: 'RT-02 · Yishun North', color: '#73b9ff', alt_m: 55.0, payload: 0.80, soc_pct: 78.1, base_temp: 34.1, lidar_range_m: 50.0, optical_flow: '95% (Tracking)', gnss_fix: 'RTK FIXED (18mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.4250, lon: 103.8350 }] },
  { id: 'UAV-CHARLIE-03', legacyId: 'UAV-03', corridorId: 'RT-03', name: 'Heavy-Lift Quad', destinationName: 'RT-03 · Tampines Regional', color: '#dbb1fc', alt_m: 42.6, payload: 1.45, soc_pct: 94.2, base_temp: 31.8, lidar_range_m: 35.0, optical_flow: '98% (Locked)', gnss_fix: 'RTK FIXED (11mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3532, lon: 103.9450 }] },
  { id: 'UAV-DELTA-04', legacyId: 'UAV-04', corridorId: 'RT-04', name: 'Inspection Hexa', destinationName: 'RT-04 · Marina Bay Downtown', color: '#f4cc7b', alt_m: 38.0, payload: 0.50, soc_pct: 68.7, base_temp: 36.2, lidar_range_m: 40.0, optical_flow: '92% (Tracking)', gnss_fix: 'RTK FIXED (22mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.2820, lon: 103.8600 }] },
  { id: 'UAV-ECHO-05', legacyId: 'UAV-05', corridorId: 'RT-05', name: 'Quantum VTOL V2', destinationName: 'RT-05 · Bukit Panjang', color: '#f698ad', alt_m: 58.5, payload: 0.90, soc_pct: 91.8, base_temp: 33.0, lidar_range_m: 55.0, optical_flow: '99% (Locked)', gnss_fix: 'RTK FIXED (12mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3880, lon: 103.7620 }] },
  { id: 'UAV-FOXTROT-06', legacyId: 'UAV-06', corridorId: 'RT-06', name: 'Falcon Express', destinationName: 'RT-06 · Punggol Digital', color: '#c5e780', alt_m: 50.5, payload: 1.10, soc_pct: 82.4, base_temp: 35.5, lidar_range_m: 48.0, optical_flow: '96% (Tracking)', gnss_fix: 'RTK FIXED (15mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.4050, lon: 103.9050 }] },
  { id: 'UAV-GOLF-07', legacyId: 'UAV-07', corridorId: 'RT-07', name: 'Solar-Endurance 6S', destinationName: 'RT-07 · Bishan Central', color: '#38bdf8', alt_m: 58.0, payload: 0.40, soc_pct: 96.0, base_temp: 30.5, lidar_range_m: 42.0, optical_flow: '97% (Locked)', gnss_fix: 'RTK FIXED (13mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3500, lon: 103.8500 }] },
  { id: 'UAV-HOTEL-08', legacyId: 'UAV-08', corridorId: 'RT-08', name: 'Urban Med-Courier', destinationName: 'RT-08 · Bedok Coastal', color: '#fb923c', alt_m: 44.0, payload: 0.75, soc_pct: 75.3, base_temp: 37.2, lidar_range_m: 38.0, optical_flow: '94% (Tracking)', gnss_fix: 'RTK FIXED (19mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3120, lon: 103.9300 }] },
  { id: 'UAV-INDIA-09', legacyId: 'UAV-09', corridorId: 'RT-09', name: 'Scout Quad V3', destinationName: 'RT-09 · Clementi West', color: '#a78bfa', alt_m: 40.0, payload: 0.55, soc_pct: 88.0, base_temp: 32.8, lidar_range_m: 40.0, optical_flow: '98% (Locked)', gnss_fix: 'RTK FIXED (12mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3200, lon: 103.7550 }] },
  { id: 'UAV-JULIETT-10', legacyId: 'UAV-10', corridorId: 'RT-10', name: 'SkyRanger G2', destinationName: 'RT-10 · Woodlands North', color: '#34d399', alt_m: 56.0, payload: 0.85, soc_pct: 85.4, base_temp: 33.7, lidar_range_m: 52.0, optical_flow: '96% (Tracking)', gnss_fix: 'RTK FIXED (16mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.4380, lon: 103.7880 }] },
  { id: 'UAV-KILO-11', legacyId: 'UAV-11', corridorId: 'RT-11', name: 'AeroPulse X4', destinationName: 'RT-11 · Ang Mo Kio', color: '#facc15', alt_m: 46.0, payload: 0.70, soc_pct: 81.0, base_temp: 34.6, lidar_range_m: 44.0, optical_flow: '97% (Locked)', gnss_fix: 'RTK FIXED (14mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3820, lon: 103.8480 }] },
  { id: 'UAV-LIMA-12', legacyId: 'UAV-12', corridorId: 'RT-12', name: 'LogiCopter 8X', destinationName: 'RT-12 · Serangoon / Hougang', color: '#fb7185', alt_m: 52.0, payload: 1.25, soc_pct: 90.2, base_temp: 31.9, lidar_range_m: 46.0, optical_flow: '99% (Locked)', gnss_fix: 'RTK FIXED (10mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3720, lon: 103.8880 }] },
  { id: 'UAV-MIKE-13', legacyId: 'UAV-13', corridorId: 'RT-13', name: 'SurveyWing RTK', destinationName: 'RT-13 · Jurong East', color: '#2dd4bf', alt_m: 43.0, payload: 0.60, soc_pct: 87.1, base_temp: 33.2, lidar_range_m: 41.0, optical_flow: '95% (Tracking)', gnss_fix: 'RTK FIXED (15mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3350, lon: 103.7420 }] },
  { id: 'UAV-NOVEMBER-14', legacyId: 'UAV-14', corridorId: 'RT-14', name: 'MedEvac Hornet', destinationName: 'RT-14 · HarbourFront South', color: '#818cf8', alt_m: 39.0, payload: 0.95, soc_pct: 76.8, base_temp: 36.8, lidar_range_m: 39.0, optical_flow: '93% (Tracking)', gnss_fix: 'RTK FIXED (20mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.2780, lon: 103.8220 }] },
  { id: 'UAV-OSCAR-15', legacyId: 'UAV-15', corridorId: 'RT-15', name: 'StealthGrid 4S', destinationName: 'RT-15 · Kallang Basin', color: '#4ade80', alt_m: 47.0, payload: 0.65, soc_pct: 84.5, base_temp: 34.0, lidar_range_m: 43.0, optical_flow: '98% (Locked)', gnss_fix: 'RTK FIXED (13mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3220, lon: 103.8750 }] },
  { id: 'UAV-PAPA-16', legacyId: 'UAV-16', corridorId: 'RT-16', name: 'Titan Cargo Heavy', destinationName: 'RT-16 · Changi North', color: '#f472b6', alt_m: 53.0, payload: 1.60, soc_pct: 92.0, base_temp: 32.1, lidar_range_m: 47.0, optical_flow: '99% (Locked)', gnss_fix: 'RTK FIXED (11mm)', waypoints: [{ lat: home.lat, lon: home.lon }, { lat: 1.3820, lon: 103.9620 }] }
];

export class Simulation {
  constructor(models) { this.models = models; this.reset(); }
  reset() {
    this.running = true; this.rate = 20; this.wind = 4; this.optimized = true; this.elapsed = 0; this.events = []; this.history = []; this.sequence = 0;
    this.globalEmergencyRTL = false;
    this.customZoneSeq = 0;
    this.zones = zones.map(z => ({ ...z }));
    this.drones = FLEET_CONFIG.map(cfg => {
      const startPos = { lat: home.lat, lon: home.lon };
      const destPos = { ...cfg.waypoints[1] };
      return {
        id: cfg.id,
        legacyId: cfg.legacyId,
        corridorId: cfg.corridorId,
        name: cfg.name,
        model: cfg.name,
        destinationName: cfg.destinationName,
        missionTarget: { ...destPos },
        color: cfg.color,
        waypoints: cfg.waypoints.map(w => ({ ...w })),
        wpIndex: 1,
        position: startPos,
        destination: destPos,
        nominalPayload: cfg.payload,
        payload: cfg.payload,
        initialBattery: cfg.soc_pct,
        battery: cfg.soc_pct,
        soc_pct: cfg.soc_pct,
        used: 0,
        status: 'EN ROUTE',
        tacticalStatus: 'OUTBOUND',
        isIndividualRTL: false,
        nominalAltitude: cfg.alt_m,
        altitude: cfg.alt_m,
        alt_m: cfg.alt_m,
        base_temp: cfg.base_temp,
        temp_c: cfg.base_temp,
        lidar_range_m: cfg.lidar_range_m,
        obstacle_clear_m: 18.0,
        optical_flow: cfg.optical_flow,
        gnss_fix: cfg.gnss_fix,
        heading: 0,
        speed: 18,
        speed_kmh: 64.8,
        power: 280,
        trail: [{ ...startPos }],
        leg: 0,
        progress: 0
      };
    });
    this.drones.forEach(d => {
      this.assign(d, d.destination);
      const firstLeg = d.route.legs[0];
      if (firstLeg) {
        d.speed = firstLeg.speed;
        d.speed_kmh = +(firstLeg.speed * 3.6).toFixed(1);
        d.power = this.models.power(firstLeg.speed, d.payload, firstLeg.headwind);
        const dLon = (firstLeg.to.lon - firstLeg.from.lon) * Math.cos(d.position.lat * Math.PI / 180);
        const dLat = firstLeg.to.lat - firstLeg.from.lat;
        d.heading = (Math.atan2(dLon, dLat) * 180 / Math.PI + 360) % 360;
      }
    });
    this.event('Simulation initialized · All 16 UAVs departing Headquarters for sector destinations.');
  }
  event(message) { this.events.unshift({ id: ++this.sequence, at: new Date().toISOString(), message }); this.events = this.events.slice(0, 40); }
  findDrone(id) {
    return this.drones.find(d => d.id === id || d.legacyId === id);
  }
  assign(d, destination) {
    let baseline, route;
    try {
      baseline = plan(this.models, d.position, destination, d.payload, this.wind, false, this.zones);
      route = plan(this.models, d.position, destination, d.payload, this.wind, this.optimized, this.zones);
    } catch (err) {
      if (this.zones.length > zones.length) {
        baseline = plan(this.models, d.position, destination, d.payload, this.wind, false, zones);
        route = plan(this.models, d.position, destination, d.payload, this.wind, this.optimized, zones);
      } else {
        throw err;
      }
    }
    d.destination = { ...destination }; d.baseline = baseline; d.route = route; d.leg = 0; d.progress = 0; d.status = 'EN ROUTE';
  }
  command(cmd) {
    const actions = ['pause', 'play', 'reset', 'optimize', 'hold', 'resume', 'return', 'return_all', 'wind', 'rate', 'add_zone', 'clear_zones'];
    if (!cmd || !actions.includes(cmd.action)) throw new Error('Unsupported simulation command');
    const d = cmd.id ? this.findDrone(cmd.id) : undefined;
    if (['hold', 'resume', 'return'].includes(cmd.action) && !d) throw new Error('Select a valid UAV');
    if (cmd.action === 'return' && d.status === 'LANDED') throw new Error('Aircraft is already landed. Reset the scenario to launch a new mission.');
    if (cmd.action === 'wind' && (typeof cmd.value !== 'number' || !Number.isFinite(cmd.value) || cmd.value < 0 || cmd.value > 10)) throw new Error('Wind must be 0–10 m/s');
    if (cmd.action === 'rate' && ![1, 5, 20, 60].includes(cmd.value)) throw new Error('Unsupported simulation speed');
    switch (cmd.action) {
      case 'pause': this.running = false; break;
      case 'play': this.running = true; break;
      case 'reset': this.reset(); return;
      case 'hold':
        if (d.status !== 'LANDED') { d.status = 'HOLDING'; d.tacticalStatus = 'HOLDING'; }
        break;
      case 'resume':
        if (d.status === 'HOLDING') {
          d.status = d.isIndividualRTL ? 'RETURNING' : 'EN ROUTE';
          d.tacticalStatus = d.isIndividualRTL ? (this.globalEmergencyRTL ? 'EMERGENCY_RTL' : 'INDIVIDUAL_RTL') : (d.wpIndex === 0 ? 'RETURN_TO_HQ' : 'OUTBOUND');
        } else if (d.status === 'RETURNING' || d.status === 'LANDED') {
          d.isIndividualRTL = false;
          d.altitude = d.nominalAltitude;
          d.alt_m = d.nominalAltitude;
          if (d.status === 'LANDED' || distance(d.position, home) < 50) {
            d.wpIndex = 1;
            d.payload = d.nominalPayload;
            d.battery = d.initialBattery || 92.0;
            d.soc_pct = +d.battery.toFixed(1);
            d.trail = [{ ...d.position }];
          }
          this.assign(d, d.waypoints[d.wpIndex]);
          d.status = 'EN ROUTE';
          d.tacticalStatus = d.wpIndex === 0 ? 'RETURN_TO_HQ' : 'OUTBOUND';
          if (this.drones.every(u => !u.isIndividualRTL)) this.globalEmergencyRTL = false;
        }
        break;
      case 'return':
        this.assign(d, home);
        d.status = 'RETURNING';
        d.isIndividualRTL = true;
        d.tacticalStatus = this.globalEmergencyRTL ? 'EMERGENCY_RTL' : 'INDIVIDUAL_RTL';
        break;
      case 'return_all':
        this.globalEmergencyRTL = true;
        for (const u of this.drones) {
          if (u.status !== 'LANDED') {
            this.assign(u, home);
            u.status = 'RETURNING';
            u.isIndividualRTL = true;
            u.tacticalStatus = 'EMERGENCY_RTL';
          }
        }
        break;
      case 'add_zone': {
        const lat = Number(cmd.lat), lon = Number(cmd.lon), radius = Number(cmd.radius || 1200);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) throw new Error('Invalid zone coordinates');
        const seq = ++this.customZoneSeq;
        this.zones.push({
          id: `CUSTOM-NFZ-0${seq}`,
          name: cmd.name || `CUSTOM-NFZ-0${seq}: Private Restricted Area`,
          lat, lon, radius,
          isCustom: true
        });
        for (const u of this.drones.filter(u => u.status === 'EN ROUTE' || u.status === 'RETURNING')) {
          try {
            const prevStatus = u.status;
            this.assign(u, u.destination);
            u.status = prevStatus;
          } catch {}
        }
        break;
      }
      case 'clear_zones':
        this.zones = zones.map(z => ({ ...z }));
        this.customZoneSeq = 0;
        for (const u of this.drones.filter(u => u.status === 'EN ROUTE' || u.status === 'RETURNING')) {
          try {
            const prevStatus = u.status;
            this.assign(u, u.destination);
            u.status = prevStatus;
          } catch {}
        }
        break;
      case 'rate': this.rate = cmd.value; break;
      case 'optimize':
      case 'wind': {
        const oldWind = this.wind, oldOptimized = this.optimized;
        if (cmd.action === 'wind') this.wind = cmd.value;
        else this.optimized = !this.optimized;
        const proposed = [];
        try {
          for (const u of this.drones.filter(u => u.status !== 'LANDED')) {
            const copy = { ...u }; this.assign(copy, u.destination); copy.status = u.status; proposed.push([u, copy]);
          }
          for (const [original, copy] of proposed) Object.assign(original, copy);
        } catch (error) { this.wind = oldWind; this.optimized = oldOptimized; throw error; }
        break;
      }
    }
    this.event(`${cmd.action.toUpperCase()}${d ? ` · ${d.id}` : ''}${cmd.value !== undefined ? ` · ${cmd.value}` : ''} · synchronized fleet`);
  }
  tick(dt) {
    if (!this.running) return;
    const step = Math.min(2, Math.max(0, dt)) * this.rate; this.elapsed += step;
    for (let idx = 0; idx < this.drones.length; idx++) {
      const d = this.drones[idx];
      if (d.status === 'LANDED') {
        d.power = 0; d.speed = 0; d.speed_kmh = 0; d.tacticalStatus = 'LANDED_AT_DEPOT';
        continue;
      }
      if (d.battery <= 20 && d.status !== 'RETURNING' && d.status !== 'HOLDING') {
        try { this.assign(d, home); d.status = 'RETURNING'; d.isIndividualRTL = true; d.tacticalStatus = 'INDIVIDUAL_RTL'; this.event(`${d.id} 20% SoC reserve threshold · returning to depot`); } catch { d.status = 'HOLDING'; d.tacticalStatus = 'HOLDING'; }
      }
      if (d.battery <= 0) {
        d.status = 'LANDED'; d.tacticalStatus = 'LANDED_AT_DEPOT'; d.altitude = 0; d.alt_m = 0; d.power = 0; d.speed = 0; d.speed_kmh = 0;
        this.event(`${d.id} simulated battery depleted`);
        continue;
      }
      let remaining = step, consumed = 0;
      if (d.status === 'HOLDING') {
        d.speed = 0; d.speed_kmh = 0; d.power = this.models.power(8, d.payload, 0); consumed = d.power * step / 3600;
        d.tacticalStatus = 'HOLDING';
      }
      while (remaining > 0 && ['EN ROUTE', 'RETURNING'].includes(d.status)) {
        const leg = d.route.legs[d.leg];
        if (!leg) {
          if (d.status === 'RETURNING') {
            d.position = { lat: home.lat, lon: home.lon };
            d.status = 'LANDED'; d.tacticalStatus = 'LANDED_AT_DEPOT'; d.altitude = 0; d.alt_m = 0; d.power = 0; d.speed = 0; d.speed_kmh = 0;
            this.event(`${d.id} arrived at depot · mission complete`);
            break;
          } else {
            d.position = { ...d.destination };
            d.wpIndex = (d.wpIndex + 1) % d.waypoints.length;
            if (d.wpIndex === 0) {
              d.payload = 0;
              this.assign(d, d.waypoints[0]);
              d.tacticalStatus = 'RETURN_TO_HQ';
              this.event(`${d.id} delivered parcel at ${d.destinationName || 'destination'} · returning to HQ`);
            } else {
              d.payload = d.nominalPayload;
              d.battery = d.initialBattery || 92.0;
              d.soc_pct = +d.battery.toFixed(1);
              d.trail = [{ ...d.position }];
              this.assign(d, d.waypoints[1]);
              d.tacticalStatus = 'OUTBOUND';
              this.event(`${d.id} returned to HQ · reloaded ${d.nominalPayload} kg for ${d.destinationName || 'destination'}`);
            }
            continue;
          }
        }
        const advance = Math.min(remaining, Math.max(0, leg.seconds - d.progress));
        d.progress += advance; remaining -= advance; d.speed = leg.speed; d.speed_kmh = +(leg.speed * 3.6).toFixed(1);
        d.power = this.models.power(leg.speed, d.payload, leg.headwind);
        consumed += d.power * advance / 3600;
        const t = leg.seconds ? Math.min(1, d.progress / leg.seconds) : 1;
        d.position = { lon: leg.from.lon + (leg.to.lon - leg.from.lon) * t, lat: leg.from.lat + (leg.to.lat - leg.from.lat) * t };
        const dLon = (leg.to.lon - leg.from.lon) * Math.cos(d.position.lat * Math.PI / 180);
        const dLat = leg.to.lat - leg.from.lat;
        d.heading = (Math.atan2(dLon, dLat) * 180 / Math.PI + 360) % 360;
        if (t >= 1) { d.leg++; d.progress = 0; }
      }
      d.used += consumed;
      d.battery = Math.max(20.1, d.battery - consumed / 20);
      d.soc_pct = +d.battery.toFixed(1);
      const minZoneClear = Math.min(...this.zones.map(z => distance(d.position, z) - z.radius));
      if (d.status === 'RETURNING') {
        d.tacticalStatus = this.globalEmergencyRTL ? 'EMERGENCY_RTL' : 'INDIVIDUAL_RTL';
      } else if (d.status === 'EN ROUTE') {
        d.tacticalStatus = minZoneClear < 650 ? 'GEOFENCE_AVOIDING' : (d.wpIndex === 0 ? 'RETURN_TO_HQ' : 'OUTBOUND');
      }
      const tempWave = Math.sin(this.elapsed * 0.08 + idx) * 1.2;
      d.temp_c = +(d.base_temp + (100 - d.battery) * 0.08 + (d.status === 'RETURNING' ? 2.5 : 0) + tempWave).toFixed(1);
      d.obstacle_clear_m = +(15.0 + 8.0 * Math.sin(this.elapsed * 0.12 + d.position.lat * 10)).toFixed(1);
      d.trail.push({ ...d.position }); if (d.trail.length > 160) d.trail.shift();
    }
    this.history.push({ time: this.elapsed, power: this.drones.reduce((s, d) => s + (d.power || 0), 0), soc: this.drones.reduce((s, d) => s + d.battery, 0) / this.drones.length });
    if (this.history.length > 120) this.history.shift();
  }
  snapshot() {
    const baseline = this.drones.reduce((s, d) => s + d.baseline.wh, 0), planned = this.drones.reduce((s, d) => s + d.route.wh, 0);
    const avgSoc = +(this.drones.reduce((s, d) => s + d.battery, 0) / this.drones.length).toFixed(1);
    const avgTemp = +(this.drones.reduce((s, d) => s + d.temp_c, 0) / this.drones.length).toFixed(1);
    return {
      timestamp: new Date().toISOString(), source: 'simulation', running: this.running, rate: this.rate, elapsed: this.elapsed,
      wind: this.wind, optimized: this.optimized, globalEmergencyRTL: this.globalEmergencyRTL,
      hardwareConnected: false, database: 'UAV_Database', persistence: 'not connected',
      drones: this.drones, zones: this.zones, home, history: this.history, events: this.events,
      graphEdges: neighbors.flatMap((peers, i) => peers.filter(j => j > i).map(j => [grid[i], grid[j]])),
      metrics: {
        baseline, planned, savings: baseline ? (1 - planned / baseline) * 100 : 0,
        airborne: this.drones.filter(d => d.status !== 'LANDED').length,
        avgSoc, avgTemp,
        clearance: Math.min(...this.drones.flatMap(d => this.zones.map(z => distance(d.position, z) - z.radius)))
      },
      model: this.models.report
    };
  }
}
