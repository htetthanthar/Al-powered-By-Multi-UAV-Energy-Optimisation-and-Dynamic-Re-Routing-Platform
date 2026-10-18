import test from 'node:test';
import assert from 'node:assert/strict';
import { createModels, graphEmbedding } from '../models.mjs';
import { Simulation, plan, grid, home, safe, zones, segmentClearance, distance } from '../engine.mjs';
import { createApp } from '../server.mjs';
const models=createModels();
test('HTTPS proxy origin is explicit and does not bypass command tokens',async()=>{
  const {server}=createApp(models,'https://64.118.128.181');
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const base=`http://127.0.0.1:${server.address().port}`;
  try {
    const {csrf}=await(await fetch(base+'/api/state')).json();
    for(const [origin,token,expected] of [['https://64.118.128.181',csrf,200],['https://example.com',csrf,403],['https://64.118.128.181','bad',403]]) {
      const result=await fetch(base+'/api/control',{method:'POST',headers:{Origin:origin,'X-UAV-Token':token,'Content-Type':'application/json'},body:JSON.stringify({action:'pause'})});
      assert.equal(result.status,expected);
    }
  } finally { await new Promise(r=>server.close(r)); }
});
test('synthetic neural training improves loss and validates on held-out samples',()=>{
  const r=models.report;assert.ok(r.history.at(-1).mse<r.history[0].mse);assert.ok(r.graphHistory.at(-1).mse<r.graphHistory[0].mse);
  assert.ok(r.r2>=.99,`R2 ${r.r2}`);assert.ok(r.graphRmse<.15);assert.equal(r.validation.length,160);
  assert.notDeepEqual(graphEmbedding([[1,1],[0,0]],[[1],[0]],0),graphEmbedding([[1,1],[1,1]],[[1],[0]],0));
});
test('all planned segments honor buffered illustrative zones under each wind',()=>{
  assert.equal(zones.length,3,'3 modelled NFZs (NFZ-A, NFZ-B, NFZ-C)');
  for(const wind of [0,4,10])for(const n of [4,13,32,30,27,21])for(const optimized of [true,false]){
    const route=plan(models,home,grid[n],1,wind,optimized);assert.ok(Number.isFinite(route.wh));assert.ok(route.wh>=0);
    route.legs.forEach(l=>{assert.ok(safe(l.from,l.to));assert.ok(l.seconds>=0);});
  }
  assert.ok(segmentClearance(zones[0],zones[0],zones[0])<0);
});
test('pause, hold, resume, return and reset affect shared simulation state',()=>{
  const s=new Simulation(models);s.tick(1);const initial=structuredClone(s.drones[0].position);s.command({action:'pause'});s.tick(1);assert.deepEqual(s.drones[0].position,initial);
  s.command({action:'play'});s.command({action:'hold',id:'UAV-01'});s.tick(1);assert.deepEqual(s.drones[0].position,initial);
  s.command({action:'resume',id:'UAV-01'});s.tick(1);assert.notDeepEqual(s.drones[0].position,initial);
  s.command({action:'return',id:'UAV-01'});assert.equal(s.drones[0].status,'RETURNING');assert.deepEqual(s.drones[0].destination,home);
  for(let i=0;i<800;i++)s.tick(1);assert.equal(s.drones[0].status,'LANDED');assert.ok(distance(s.drones[0].position,home)<1);
  assert.throws(()=>s.command({action:'return',id:'UAV-01'}));
  s.command({action:'reset'});assert.equal(s.elapsed,0);assert.equal(s.drones.length,16);assert.equal(s.snapshot().hardwareConnected,false);
});
test('replanning keeps positions and rejects invalid commands',()=>{
  const s=new Simulation(models);s.tick(1);const p=structuredClone(s.drones.map(d=>d.position));s.command({action:'wind',value:7});assert.deepEqual(s.drones.map(d=>d.position),p);
  assert.throws(()=>s.command({action:'wind',value:NaN}));assert.throws(()=>s.command({action:'wind',value:100}));assert.throws(()=>s.command({action:'rate',value:-1}));assert.throws(()=>s.command({action:'hold',id:'bad'}));assert.throws(()=>s.command({action:'arm'}));
});
test('API serves state, protects controls and refuses filesystem traversal',async()=>{
  const {server}=createApp(models);await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
  try{
    const state=await(await fetch(base+'/api/state')).json();assert.equal(state.database,'UAV_Database');assert.equal(state.source,'simulation');
    const bad=await fetch(base+'/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"action":"pause"}'});assert.equal(bad.status,403);
    const ok=await fetch(base+'/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-UAV-Token':state.csrf},body:'{"action":"pause"}'});assert.equal(ok.status,200);assert.equal((await ok.json()).running,false);
    const crossed=await fetch(base+'/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-UAV-Token':state.csrf,Origin:'https://example.com'},body:'{"action":"play"}'});assert.equal(crossed.status,403);
    assert.equal((await fetch(base+'/%2e%2e%2fpackage.json')).status,403);
    assert.equal((await fetch(base+'/')).status,200);
  }finally{await new Promise(r=>server.close(r));}
});
test('all 16 drones depart from headquarters, reach distinct destinations, drop payload, return to headquarters, and loop',()=>{
  const s=new Simulation(models);
  assert.equal(s.drones.length,16);
  const destKeys=new Set();
  for(const d of s.drones){
    assert.ok(distance(d.position,home)<1,`${d.id} starts at headquarters`);
    assert.ok(distance(d.destination,home)>2000,`${d.id} has destination away from headquarters`);
    assert.ok(d.destination.lon>=103.70&&d.destination.lon<=104.00,`${d.id} destination lon in operating box`);
    assert.ok(d.destination.lat>=1.275&&d.destination.lat<=1.450,`${d.id} destination lat in operating box`);
    assert.ok(d.altitude<60,`${d.id} altitude < 60m CAAS ceiling`);
    assert.ok(d.payload>0,`${d.id} carries outbound payload`);
    destKeys.add(`${d.destination.lat},${d.destination.lon}`);
  }
  assert.equal(destKeys.size,16,'all 16 destinations are distinct');
  const seenReturn=new Set(),seenSecondLoop=new Set();
  for(let i=0;i<300;i++){
    s.tick(1);
    for(const d of s.drones){
      if(d.wpIndex===0){
        assert.equal(d.payload,0,`${d.id} dropped parcel payload on return leg`);
        seenReturn.add(d.id);
      }
      if(d.wpIndex===1&&seenReturn.has(d.id)){
        assert.equal(d.payload,d.nominalPayload,`${d.id} reloaded parcel payload at HQ`);
        seenSecondLoop.add(d.id);
      }
    }
  }
  assert.equal(seenReturn.size,16,'all 16 drones returned toward headquarters after reaching destination');
  assert.equal(seenSecondLoop.size,16,'all 16 drones looped at headquarters and departed again');
});
