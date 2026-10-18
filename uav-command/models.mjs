// Original, small neural models trained only on synthetic examples. No flight validation.
export function random(seed = 42) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export class MLP {
  constructor(sizes, seed = 10) {
    const rng = random(seed);
    this.layers = sizes.slice(1).map((n, l) => Array.from({ length: n }, () => Array.from({ length: sizes[l] + 1 }, () => (rng() - .5) * Math.sqrt(6 / sizes[l]))));
  }
  forward(input) {
    const activations = [input];
    this.layers.forEach((layer, l) => activations.push(layer.map(w => {
      const z = w.at(-1) + activations[l].reduce((sum, v, j) => sum + v * w[j], 0);
      return l === this.layers.length - 1 ? z : Math.tanh(z);
    })));
    return activations;
  }
  predict(x) { return this.forward(x).at(-1)[0]; }
  train(samples, epochs = 90) {
    const history = [];
    for (let e = 0; e < epochs; e++) {
      let error = 0;
      const rate = .018 / (1 + e / 80);
      for (const [x, target] of samples) {
        const a = this.forward(x), deltas = Array(this.layers.length);
        error += (a.at(-1)[0] - target) ** 2;
        deltas[deltas.length - 1] = [Math.max(-2, Math.min(2, a.at(-1)[0] - target))];
        for (let l = deltas.length - 2; l >= 0; l--) deltas[l] = a[l + 1].map((value, j) =>
          (1 - value * value) * deltas[l + 1].reduce((s, d, k) => s + d * this.layers[l + 1][k][j], 0));
        this.layers.forEach((layer, l) => layer.forEach((w, k) => {
          for (let j = 0; j < w.length - 1; j++) w[j] -= rate * deltas[l][k] * a[l][j];
          w[w.length - 1] -= rate * deltas[l][k];
        }));
      }
      history.push({ epoch: e + 1, mse: error / samples.length });
    }
    return history;
  }
}
export function physicalPower(speed, payload, headwind, climb = 0, temp = 31) {
  return 210 + 70 * payload + .085 * speed ** 3 + 18 * Math.max(0, headwind) + 60 * Math.max(0, climb) + 8 * Math.max(0, temp - 30);
}
export const energyFeatures = (v, p, w, c = 0, t = 31) => [v / 25, p / 3, w / 10, c / 3, t / 40];
// GraphSAGE-style one-hop mean aggregation: local weather plus neighboring weather.
// The shared trainable MLP maps each node's message-passing embedding to a hazard score.
export function graphEmbedding(features, neighbors, index) {
  const peers = neighbors[index];
  const mean = features[index].map((_, j) => peers.length ? peers.reduce((s, k) => s + features[k][j], 0) / peers.length : features[index][j]);
  return [...features[index], ...mean];
}
export function createModels() {
  const rng = random(806), energy = new MLP([5, 16, 12, 1]), graph = new MLP([8, 12, 1], 26);
  const energySample = () => { const v = 8 + rng() * 16, p = rng() * 3, w = rng() * 18 - 8, c = rng() * 6 - 3, t = 26 + rng() * 12;
    return [energyFeatures(v, p, w, c, t), physicalPower(v, p, w, c, t) / 1200]; };
  const graphSample = () => {
    const f = Array.from({ length: 5 }, () => [rng(), rng(), rng(), rng()]);
    const x = graphEmbedding(f, [[1, 2, 3, 4], [0], [0], [0], [0]], 0);
    return [x, .4 * x[0] ** 2 + .4 * x[1] + .12 * x[4] + .08 * x[5]];
  };
  const history = energy.train(Array.from({ length: 640 }, energySample), 275);
  const graphHistory = graph.train(Array.from({ length: 640 }, graphSample), 70);
  const validation = Array.from({ length: 160 }, energySample).map(([x, y]) => ({ actual: y * 1200, predicted: energy.predict(x) * 1200 }));
  const mean = validation.reduce((s, p) => s + p.actual, 0) / validation.length;
  const sse = validation.reduce((s, p) => s + (p.actual - p.predicted) ** 2, 0);
  const graphValidation = Array.from({ length: 160 }, graphSample);
  return {
    energy, graph,
    power: (v, p, w, c = 0, t = 31) => Math.max(80, energy.predict(energyFeatures(v, p, w, c, t)) * 1200),
    report: { provenance: 'Synthetic training and held-out validation; not flight-tested',
      energyArchitecture: '5 → 16 → 12 → 1 · tanh MLP', graphArchitecture: 'Mean neighbor aggregation → 8 → 12 → 1',
      trainingSamples: 640, validationSamples: 160, history, graphHistory, validation,
      r2: 1 - sse / validation.reduce((s, p) => s + (p.actual - mean) ** 2, 0), rmse: Math.sqrt(sse / validation.length),
      graphRmse: Math.sqrt(graphValidation.reduce((s, [x, y]) => s + (graph.predict(x) - y) ** 2, 0) / graphValidation.length) }
  };
}
