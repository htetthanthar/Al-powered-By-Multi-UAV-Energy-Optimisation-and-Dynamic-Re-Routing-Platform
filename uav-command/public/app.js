const $ = id => document.getElementById(id);
let snapshot, token, selected = 'UAV-ALPHA-01', online = false, lastMessage = 0, viewer, graphVisible = false, overhead = false, follow = false, toastTimer, graphKey = '', zoneKey = '', localSim = null;
const aircraft = new Map(), routes = new Map(), trails = new Map(), destinations = new Map(), graphEntities = [], zoneEntities = [];
const C = window.Cesium;
const format = (n, digits = 1) => Number(n || 0).toFixed(digits);
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4500); }
function setOnline(value) { online = value; document.body.classList.toggle('disconnected', !value); $('connection').textContent = value ? '● Stream connected' : '○ Stream offline'; $('connection').classList.toggle('stale', !value); for (const id of ['play','rate','reset','hold','return','wind','optimize']) $(id).disabled = !value; }
async function command(action, extras = {}) {
  if (!online) return toast('Connection unavailable. Commands are disabled.');
  if (localSim) {
    try {
      localSim.command({ action, ...extras });
      receive(localSim.snapshot());
      toast(`${action === 'optimize' ? 'Routes recalculated' : action.toUpperCase()} · synchronized fleet`);
    } catch (error) { toast(error.message); }
    return;
  }
  try {
    const response = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-UAV-Token': token }, body: JSON.stringify({ action, ...extras }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Command failed');
    receive(data); toast(`${action === 'optimize' ? 'Routes recalculated' : action.toUpperCase()} · synchronized fleet`);
  } catch (error) { toast(error.message); }
}
const point = (p, height = 120) => C.Cartesian3.fromDegrees(p.lon, p.lat, height);
function cameraHome() { if (viewer) viewer.camera.flyTo({ destination: C.Cartesian3.fromDegrees(103.837, overhead ? 1.345 : 1.115, overhead ? 33000 : 29000), orientation: { heading: 0, pitch: C.Math.toRadians(overhead ? -90 : -51), roll: 0 }, duration: matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 1.3 }); }
function syncZones(zones) {
  if (!viewer) return;
  const nextKey = JSON.stringify(zones.map(z => [z.id, z.lat, z.lon, z.radius]));
  if (nextKey === zoneKey) return;
  for (const e of zoneEntities) viewer.entities.remove(e);
  zoneEntities.length = 0;
  for (const z of zones) {
    const col = z.isCustom ? '#f59e0b' : '#f07632';
    zoneEntities.push(viewer.entities.add({
      position: point(z, 30),
      ellipse: { semiMajorAxis: z.radius + 500, semiMinorAxis: z.radius + 500, material: C.Color.fromCssColorString(col).withAlpha(z.isCustom ? .22 : .14), outline: true, outlineColor: C.Color.fromCssColorString(col).withAlpha(.88), height: 20 },
      label: { text: z.name || 'EXCLUSION ZONE', font: '600 9px monospace', fillColor: C.Color.fromCssColorString('#ffd3b6'), showBackground: true, backgroundColor: C.Color.fromCssColorString('#14171cd9'), backgroundPadding: new C.Cartesian2(6, 4), pixelOffset: new C.Cartesian2(0, -18), disableDepthTestDistance: Infinity }
    }));
  }
  zoneKey = nextKey;
}
function initMap(data) {
  if (!C) { $('map-error').hidden = false; $('map-error').textContent = 'Cesium could not load. Run npm install in uav-command, then refresh. Telemetry remains available.'; return; }
  try {
    C.Ion.defaultAccessToken = '';
    viewer = new C.Viewer('globe', { baseLayer: false, terrainProvider: new C.EllipsoidTerrainProvider(), animation: false, timeline: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false, baseLayerPicker: false, fullscreenButton: false, selectionIndicator: false, infoBox: false, shouldAnimate: false });
    viewer.scene.globe.baseColor = C.Color.fromCssColorString('#111318');
    viewer.scene.backgroundColor = C.Color.fromCssColorString('#090a0d');
    viewer.scene.globe.enableLighting = false;
    const imagery = new C.UrlTemplateImageryProvider({ url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', credit: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community', maximumLevel: 18 });
    let failures = 0;
    imagery.errorEvent.addEventListener(() => { if (++failures > 4) { $('map-error').hidden = false; $('map-error').textContent = 'Satellite imagery is unavailable. UAV coordinates, plans, and graph overlays remain active on the ellipsoid globe.'; } });
    const layer = viewer.imageryLayers.addImageryProvider(imagery); layer.brightness = .56; layer.saturation = .55; layer.contrast = 1.22;
    viewer.scene.screenSpaceCameraController.minimumZoomDistance = 600;
    viewer.scene.screenSpaceCameraController.maximumZoomDistance = 15000000;
    syncZones(data.zones);
    viewer.entities.add({ position: point(data.home, 35), point: { pixelSize: 13, color: C.Color.fromCssColorString('#4ade80'), outlineColor: C.Color.fromCssColorString('#14171c'), outlineWidth: 3, disableDepthTestDistance: Infinity }, label: { text: 'HEADQUARTERS · QUEENSTOWN DEPOT', font: '600 10px monospace', pixelOffset: new C.Cartesian2(0, 24), fillColor: C.Color.fromCssColorString('#4ade80'), showBackground: true, backgroundColor: C.Color.fromCssColorString('#13161ce6'), backgroundPadding: new C.Cartesian2(8, 5), disableDepthTestDistance: Infinity } });
    const locations = [['MARINA BAY',103.86,1.282],['JURONG EAST',103.742,1.335],['BISHAN',103.849,1.35],['TAMPINES',103.945,1.354],['WOODLANDS',103.785,1.435]];
    for (const [name,lon,lat] of locations) viewer.entities.add({position:point({lon,lat},40),label:{text:name,font:'10px monospace',fillColor:C.Color.fromCssColorString('#9ca3af'),pixelOffset:new C.Cartesian2(0,10),disableDepthTestDistance:Infinity}});
    const handler = new C.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(movement => { const picked = viewer.scene.pick(movement.position); const id = picked?.id?.id; if (typeof id === 'string' && id.startsWith('UAV-')) selectAircraft(id); }, C.ScreenSpaceEventType.LEFT_CLICK);
    cameraHome();
    window.__uavViewer = viewer;
  } catch (error) { $('map-error').hidden = false; $('map-error').textContent = '3D rendering is unavailable in this browser. Try a WebGL-enabled browser. Telemetry and simulation controls remain usable.'; console.error(error); }
}
function updateMap(data) {
  if (!viewer) return;
  syncZones(data.zones);
  for (const d of data.drones) {
    const chosen = d.id === selected;
    const ringHex = chosen ? '#f07632' : d.color;
    const color = C.Color.fromCssColorString(ringHex);
    const targetPos = d.missionTarget || d.waypoints?.[1] || d.destination;
    if (targetPos && !destinations.has(d.id)) {
      destinations.set(d.id, viewer.entities.add({
        position: point(targetPos, 32),
        point: { pixelSize: 7, color: C.Color.fromCssColorString(d.color), outlineColor: C.Color.fromCssColorString('#14171c'), outlineWidth: 2, disableDepthTestDistance: Infinity },
        label: { text: `DEST · ${d.destinationName || d.id}`, font: '600 9px monospace', pixelOffset: new C.Cartesian2(0, 18), fillColor: C.Color.fromCssColorString(d.color), showBackground: true, backgroundColor: C.Color.fromCssColorString('#13161ce6'), backgroundPadding: new C.Cartesian2(6, 4), disableDepthTestDistance: Infinity, show: false }
      }));
    }
    if (destinations.has(d.id)) {
      const destEntity = destinations.get(d.id);
      if (targetPos) destEntity.position = point(targetPos, 32);
      destEntity.point.pixelSize = chosen ? 10 : 6;
      destEntity.label.show = chosen;
    }
    if (!aircraft.has(d.id)) aircraft.set(d.id, viewer.entities.add({ id: d.id, position: point(d.position,d.altitude),
      billboard: { image: 'data:image/svg+xml,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 56 56"><circle cx="28" cy="28" r="25" fill="none" stroke="${d.color}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.75"/><circle cx="28" cy="28" r="17" fill="#181b22" fill-opacity="0.92" stroke="${d.color}" stroke-width="2"/><g fill="none" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round"><path d="M21 21L35 35M21 35L35 21"/><circle cx="20" cy="20" r="4.2" stroke="${d.color}"/><circle cx="36" cy="20" r="4.2" stroke="${d.color}"/><circle cx="20" cy="36" r="4.2" stroke="${d.color}"/><circle cx="36" cy="36" r="4.2" stroke="${d.color}"/></g><path fill="${d.color}" d="M28 19l4 6-4 3-4-3z"/></svg>`), width: 36, height: 36, disableDepthTestDistance: Infinity },
      label: { text: d.id, font: '600 10px monospace', pixelOffset: new C.Cartesian2(0,-28), fillColor: C.Color.WHITE, showBackground: true, backgroundColor: C.Color.fromCssColorString('#15181fe6'), backgroundPadding: new C.Cartesian2(8,5), disableDepthTestDistance: Infinity } }));
    const entity = aircraft.get(d.id); entity.position = point(d.position,d.altitude); entity.billboard.width = chosen ? 46 : 34; entity.billboard.height = chosen ? 46 : 34;
    entity.billboard.rotation = -C.Math.toRadians(d.heading || 0);
    entity.label.text = chosen ? `${d.id} · ${format(d.battery,0)}%` : d.id;
    entity.label.fillColor = chosen ? C.Color.fromCssColorString('#ff985a') : C.Color.WHITE;
    const index = data.drones.indexOf(d);
    entity.label.pixelOffset = new C.Cartesian2(chosen ? 0 : (index % 2 ? 25 : -25), -30 - (index % 3) * 19);
    if (!routes.has(d.id)) {
      routes.set(d.id, { neural: viewer.entities.add({ polyline: { positions: [], width: 2, material: color.withAlpha(.85) } }), baseline: viewer.entities.add({ polyline: { positions: [], width: 1.5, material: new C.PolylineDashMaterialProperty({ color: C.Color.fromCssColorString('#5eead4').withAlpha(.65), dashLength: 14 }) } }) });
      trails.set(d.id, viewer.entities.add({polyline:{positions:[],width:3,material:color.withAlpha(.65)}}));
    }
    const r = routes.get(d.id); r.neural.polyline.positions = d.route.points.map(p=>point(p,d.altitude)); r.neural.polyline.width = chosen ? 3.2 : 1.6;
    r.neural.polyline.material = color.withAlpha(chosen ? .95 : .45);
    r.baseline.show = chosen; r.baseline.polyline.positions = d.baseline.points.map(p=>point(p,d.altitude+8));
    trails.get(d.id).polyline.positions = d.trail.map(p=>point(p,d.altitude));
  }
  const nextGraphKey = `${graphVisible}/${data.wind}`;
  if (nextGraphKey !== graphKey) {
  for (const e of graphEntities) viewer.entities.remove(e); graphEntities.length=0;
  if (graphVisible) {
    for (const edge of data.graphEdges) graphEntities.push(viewer.entities.add({polyline:{positions:edge.map(p=>point(p,55)),width:1,material:C.Color.fromCssColorString('#f07632').withAlpha(.22)}}));
    const nodes = data.drones[0].route.hazard;
    for(const p of nodes) graphEntities.push(viewer.entities.add({position:point(p,60),point:{pixelSize:5,color:C.Color.fromCssColorString(p.risk>.45?'#f07632':'#5eead4')},label:{text:`${p.id} · ${format(p.risk,2)}`,font:'8px monospace',pixelOffset:new C.Cartesian2(0,12),disableDepthTestDistance:Infinity}}));
  }
  graphKey = nextGraphKey;
  }
  if(follow){const d=data.drones.find(d=>d.id===selected);viewer.camera.setView({destination:C.Cartesian3.fromDegrees(d.position.lon,d.position.lat-.028,4500),orientation:{heading:0,pitch:C.Math.toRadians(-52),roll:0}});}
}
function selectAircraft(id) { selected = id; render(); }
function renderFleet() {
  const query=$('fleet-search').value.toLowerCase();
  const filtered=snapshot.drones.filter(d=>`${d.id} ${d.name} ${d.destinationName || ''}`.toLowerCase().includes(query));
  $('fleet-list').replaceChildren(...filtered.map(d=>{
    const button=document.createElement('button'); button.className=`aircraft ${d.id===selected?'selected':''}`; button.style.setProperty('--aircraft',d.color); button.setAttribute('aria-pressed',String(d.id===selected)); button.setAttribute('aria-label',`Select ${d.id}, ${d.name}`);
    button.innerHTML=`<span class="aircraft-symbol">✣</span><span><strong>${d.id}</strong><small>${d.destinationName || d.name} · • ${(d.tacticalStatus || d.status).toLowerCase()}</small></span><span class="soc">${format(d.battery,1)}<small>%</small></span>`;
    button.addEventListener('click',()=>selectAircraft(d.id));return button;
  }));
  if(!filtered.length){const p=document.createElement('p');p.className='empty';p.textContent='No aircraft match your search.';$('fleet-list').append(p);}
}
function linePath(values,w,h,max){return values.map((v,i)=>`${i?'L':'M'}${(i/(values.length-1||1)*w).toFixed(1)},${(h-8-v/(max||1)*(h-16)).toFixed(1)}`).join(' ');}
function render() {
  if(!snapshot)return;
  const d=snapshot.drones.find(d=>d.id===selected)||snapshot.drones[0];
  selected=d.id;
  // Preserve keyboard focus during telemetry refreshes.
  const focus=document.activeElement?.getAttribute('aria-label'); renderFleet();
  if(focus?.startsWith('Select UAV-'))for(const b of $('fleet-list').querySelectorAll('button'))if(b.getAttribute('aria-label')===focus)b.focus({preventScroll:true});
  $('fleet-count').textContent=String(snapshot.metrics.airborne).padStart(2,'0');
  $('selected-id').textContent=d.id;$('selected-name').textContent=`${d.name} · HQ ↔ ${d.destinationName || 'Sector'} · ${format(d.temp_c,1)} °C`;$('selected-status').textContent=`• ${d.tacticalStatus || d.status}`;
  $('selected-battery').textContent=`${format(d.battery)}%`;$('battery-fill').style.width=`${d.battery}%`;
  $('altitude').innerHTML=`${format(d.altitude,1)} <em>m</em>`;$('airspeed').innerHTML=`${format(d.speed)} <em>m/s</em> (${format(d.speed_kmh,1)} km/h)`;
  $('power').innerHTML=`${format(d.power,0)} <em>W</em>`;$('payload').innerHTML=`${format(d.payload,2)} <em>kg</em>`;
  $('coordinates').textContent=`${format(d.position.lat,5)}° N / ${format(d.position.lon,5)}° E · HDG ${format(d.heading,0)}°`;
  $('hold').textContent=d.status==='HOLDING'?'Resume flight':'Hold position'; $('hold').disabled=!online||d.status==='LANDED';
  $('return').textContent=(d.status==='RETURNING'||d.status==='LANDED')?'Resume patrol':'Return to depot';
  $('return').disabled=!online;
  $('play').textContent=snapshot.running?'Ⅱ Pause':'▶ Resume';$('rate').value=String(snapshot.rate);
  $('optimize').setAttribute('aria-pressed',String(snapshot.optimized));
  if(document.activeElement!==$('wind'))$('wind').value=String(snapshot.wind);
  $('wind-value').textContent=`${format(snapshot.wind)} m/s`;
  $('model-accuracy').textContent=`Holdout R² ${format(snapshot.model.r2,3)} · synthetic`;
  $('savings').textContent=`${format(snapshot.metrics.savings)}%`;$('baseline-wh').textContent=`${format(snapshot.metrics.baseline)} Wh`;$('planned-wh').textContent=`${format(snapshot.metrics.planned)} Wh`;
  $('energy-bar').style.width=`${Math.min(100,Math.max(0,snapshot.metrics.planned/(snapshot.metrics.baseline||1)*100))}%`;
  const mins=Math.floor(snapshot.elapsed/60),secs=Math.floor(snapshot.elapsed%60);$('sim-time').textContent=`T+${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
  const power=snapshot.drones.reduce((s,d)=>s+(d.power||0),0);$('total-power').textContent=`${format(power,0)} W`;
  const values=snapshot.history.map(p=>p.power),path=linePath(values,500,110,Math.max(1000,...values)*1.1);
  $('power-line').setAttribute('d',path);$('power-area').setAttribute('d',values.length?`${path} L500,110 L0,110 Z`:'');
  $('events').replaceChildren(...snapshot.events.slice(0,4).map(event=>{const el=document.createElement('div');el.className='event';const t=document.createElement('time');t.textContent=new Date(event.at).toLocaleTimeString('en-GB',{timeZone:'Asia/Singapore',hour:'2-digit',minute:'2-digit'});const p=document.createElement('p');p.textContent=event.message;el.append(t,p);return el;}));
  updateMap(snapshot);
}
function receive(data){snapshot=data;lastMessage=Date.now();setOnline(true);render();}
function openLearning(){
  if(!snapshot)return;const m=snapshot.model;
  $('learning-stats').innerHTML=`<div><strong>${format(m.r2,3)}</strong><span>MLP HOLDOUT R²</span></div><div><strong>${format(m.rmse)} W</strong><span>MLP HOLDOUT RMSE</span></div><div><strong>${format(m.graphRmse,3)}</strong><span>GNN HAZARD RMSE</span></div><div><strong>${m.validationSamples}</strong><span>SYNTHETIC TEST SAMPLES</span></div>`;
  const max=Math.max(...m.history.map(p=>p.mse));$('loss-chart').innerHTML=`<path d="${linePath(m.history.map(p=>p.mse),440,180,max)}" fill="none" stroke="#f07632" stroke-width="2"/>`;
  const limit=Math.max(...m.validation.flatMap(p=>[p.actual,p.predicted]))*1.05;
  $('fit-chart').innerHTML='<path d="M15 165L425 15" stroke="#5eead4" stroke-dasharray="5 5"/>'+m.validation.map(p=>`<circle cx="${15+p.actual/limit*410}" cy="${165-p.predicted/limit*150}" r="2.4" fill="#f07632" opacity=".75"/>`).join('');
  $('learning-dialog').showModal();
}
$('fleet-search').addEventListener('input',renderFleet);$('play').onclick=()=>command(snapshot.running?'pause':'play');
$('rate').onchange=e=>command('rate',{value:Number(e.target.value)});
$('reset').onclick=()=>{if(confirm('Reset the simulation? Current virtual missions and in-memory history will be cleared.'))command('reset');};
$('hold').onclick=()=>command(snapshot.drones.find(d=>d.id===selected).status==='HOLDING'?'resume':'hold',{id:selected});
$('return').onclick=()=>{const d=snapshot.drones.find(u=>u.id===selected);command(d?.status==='RETURNING'||d?.status==='LANDED'?'resume':'return',{id:selected});};
$('optimize').onclick=()=>command('optimize');
$('wind').oninput=e=>$('wind-value').textContent=`${format(e.target.value)} m/s`;$('wind').onchange=e=>command('wind',{value:Number(e.target.value)});
$('home-view').onclick=()=>{follow=false;$('follow').classList.remove('on');cameraHome();};$('view-mode').onclick=()=>{overhead=!overhead;$('view-mode').textContent=overhead?'3D':'2D';follow=false;cameraHome();};
$('follow').onclick=()=>{follow=!follow;$('follow').classList.toggle('on',follow);if(!follow)cameraHome();};
$('layer-toggle').onclick=()=>{graphVisible=!graphVisible;$('layer-toggle').classList.toggle('on',graphVisible);if(snapshot)updateMap(snapshot);};
$('models-tab').onclick=openLearning;$('inspect-model').onclick=openLearning;$('close-model').onclick=()=>$('learning-dialog').close();$('operations-tab').onclick=()=>$('learning-dialog').close();
$('export').onclick=()=>{if(!snapshot)return;const blob=new Blob([JSON.stringify(snapshot,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='stratum-simulation-snapshot.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Exported simulated telemetry; no credentials included.');};
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString('en-GB',{timeZone:'Asia/Singapore'})+' SGT';if(Date.now()-lastMessage>5000)setOnline(false);},1000);
async function connect(){
  try {
    if (window.__STRATUM_EMBEDDED_SIM__) throw new Error('Embedded cloud simulation');
    const response=await fetch('/api/state');if(!response.ok)throw new Error('State service unavailable');const data=await response.json();token=data.csrf;delete data.csrf;initMap(data);receive(data);
    const events=new EventSource('/api/events');events.onmessage=e=>{try{receive(JSON.parse(e.data));}catch(error){toast('Invalid telemetry frame');}};events.onerror=()=>setOnline(false);
  }catch(error){
    if (typeof window.__createStratumEmbeddedSim === 'function') {
      localSim = window.__createStratumEmbeddedSim();
      const data = localSim.snapshot();
      initMap(data);
      receive(data);
      setInterval(() => { localSim.tick(0.5); receive(localSim.snapshot()); }, 500);
      return;
    }
    setOnline(false);toast(error.message);$('map-error').hidden=false;$('map-error').textContent='Start the local server with npm start, then refresh. No telemetry connection is available.';
  }
}
setOnline(false);connect();
