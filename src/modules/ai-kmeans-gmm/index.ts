// FILE: src/modules/ai-kmeans-gmm/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface GMMClusterColor {
  hex: string;
  r: number;
  g: number;
  b: number;
  name: string;
}

export const CLUSTER_PALETTE: GMMClusterColor[] = [
  { hex: '#38bdf8', r: 56, g: 189, b: 248, name: 'Sky' }, // Cluster 0
  { hex: '#10b981', r: 16, g: 185, b: 129, name: 'Emerald' }, // Cluster 1
  { hex: '#f59e0b', r: 245, g: 158, b: 11, name: 'Amber' }, // Cluster 2
  { hex: '#f43f5e', r: 244, g: 63, b: 94, name: 'Rose' }, // Cluster 3
  { hex: '#a855f7', r: 168, g: 85, b: 247, name: 'Violet' }, // Cluster 4
  { hex: '#06b6d4', r: 6, g: 182, b: 212, name: 'Cyan' }, // Cluster 5
];

export interface GMMDataPoint {
  x: number;
  y: number;
  baseClusterIndex?: number;
  baseOffsetX?: number;
  baseOffsetY?: number;
  responsibilities: number[];
  blendedR: number;
  blendedG: number;
  blendedB: number;
  primaryCluster: number;
  secondaryCluster: number;
  secondaryWeight: number;
}

export interface GMMComponent {
  id: number;
  mu: { x: number; y: number };
  sigma: {
    xx: number;
    xy: number;
    yy: number;
  };
  pi: number; // mixing weight
  r1: number; // 1-sigma semi-major
  r2: number; // 1-sigma semi-minor
  angle: number; // orientation angle (rad)
}

export interface GMMState {
  points: GMMDataPoint[];
  clusters: GMMComponent[];
  draggedClusterIndex: number | null;
  logLikelihood: number;
  logLikelihoodHistory: number[];
  iteration: number;
  isConverged: boolean;
  pointSpread: number;
  activePseudocodeLine: number;
}

export interface GMMParams {
  numClusters: number;
  pointSpread: number;
  softnessTemp: number;
  stepRate: number;
}

// Pseudo-random deterministic generator for consistent visualization reload
function createPRNG(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function sampleStandardNormal(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

// Decompose 2x2 covariance matrix into ellipse semi-axes and rotation angle
function decomposeCovariance(sigma: { xx: number; xy: number; yy: number }): {
  r1: number;
  r2: number;
  angle: number;
} {
  const { xx, xy, yy } = sigma;
  const trace = xx + yy;
  const det = Math.max(16, xx * yy - xy * xy);
  const disc = Math.sqrt(Math.max(0, (xx - yy) * (xx - yy) + 4 * xy * xy));
  const lambda1 = Math.max(16, (trace + disc) / 2);
  const lambda2 = Math.max(16, (trace - disc) / 2);
  const r1 = Math.sqrt(lambda1);
  const r2 = Math.sqrt(lambda2);
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  return { r1, r2, angle };
}

// Generate anisotropic clusters of ~90 data points
function generateDataPoints(spread: number): GMMDataPoint[] {
  const rng = createPRNG(1337);
  const points: GMMDataPoint[] = [];

  const clusterConfigs = [
    { cx: -130, cy: -85, angle: -0.52, scaleMajor: 1.8, scaleMinor: 0.65, count: 22 },
    { cx: 125, cy: -75, angle: 0.62, scaleMajor: 1.7, scaleMinor: 0.6, count: 23 },
    { cx: 115, cy: 100, angle: -0.7, scaleMajor: 1.85, scaleMinor: 0.7, count: 22 },
    { cx: -120, cy: 95, angle: 0.44, scaleMajor: 1.65, scaleMinor: 0.65, count: 23 },
  ];

  clusterConfigs.forEach((config, cIdx) => {
    const cosA = Math.cos(config.angle);
    const sinA = Math.sin(config.angle);

    for (let i = 0; i < config.count; i++) {
      const zMajor = sampleStandardNormal(rng) * config.scaleMajor;
      const zMinor = sampleStandardNormal(rng) * config.scaleMinor;

      const offX = (zMajor * cosA - zMinor * sinA) * spread;
      const offY = (zMajor * sinA + zMinor * cosA) * spread;

      points.push({
        x: config.cx + offX,
        y: config.cy + offY,
        baseClusterIndex: cIdx,
        baseOffsetX: (zMajor * cosA - zMinor * sinA),
        baseOffsetY: (zMajor * sinA + zMinor * cosA),
        responsibilities: [0.25, 0.25, 0.25, 0.25],
        blendedR: 148,
        blendedG: 163,
        blendedB: 184,
        primaryCluster: cIdx,
        secondaryCluster: (cIdx + 1) % 4,
        secondaryWeight: 0,
      });
    }
  });

  // Add 4 ambiguous boundary / bridge points between clusters to showcase soft EM competition
  const bridgeOffsets = [
    { x: -5, y: -78 },
    { x: 120, y: 12 },
    { x: -2, y: 98 },
    { x: -125, y: 5 },
  ];
  bridgeOffsets.forEach((b) => {
    points.push({
      x: b.x,
      y: b.y,
      responsibilities: [0.25, 0.25, 0.25, 0.25],
      blendedR: 200,
      blendedG: 200,
      blendedB: 200,
      primaryCluster: 0,
      secondaryCluster: 1,
      secondaryWeight: 0.45,
    });
  });

  return points;
}

// Initial cluster centroid and covariance setups
function createInitialComponents(K: number): GMMComponent[] {
  const components: GMMComponent[] = [];
  const defaultPositions = [
    { x: -80, y: -50 },
    { x: 80, y: -50 },
    { x: 70, y: 70 },
    { x: -70, y: 70 },
    { x: 0, y: -90 },
    { x: 0, y: 90 },
  ];

  for (let k = 0; k < K; k++) {
    const pos = defaultPositions[k % defaultPositions.length];
    const sigma = { xx: 1200, xy: 0, yy: 1200 };
    const { r1, r2, angle } = decomposeCovariance(sigma);

    components.push({
      id: k,
      mu: { x: pos.x, y: pos.y },
      sigma,
      pi: 1.0 / K,
      r1,
      r2,
      angle,
    });
  }

  return components;
}

// E-Step: Calculate soft responsibilities and log-likelihood
function evaluateEStep(
  points: GMMDataPoint[],
  clusters: GMMComponent[],
  temperature: number
): {
  updatedPoints: GMMDataPoint[];
  logLikelihood: number;
} {
  const K = clusters.length;
  let totalLL = 0;

  const updatedPoints = points.map((p) => {
    const densities: number[] = new Array(K);
    const logWeightedDensities: number[] = new Array(K);

    for (let k = 0; k < K; k++) {
      const c = clusters[k];
      const dx = p.x - c.mu.x;
      const dy = p.y - c.mu.y;

      const det = Math.max(16, c.sigma.xx * c.sigma.yy - c.sigma.xy * c.sigma.xy);
      const dM2 = (c.sigma.yy * dx * dx - 2 * c.sigma.xy * dx * dy + c.sigma.xx * dy * dy) / det;
      const safeDM2 = Math.min(100, Math.max(0, dM2));

      // 2D Gaussian PDF
      const pdf = (1 / (2 * Math.PI * Math.sqrt(det))) * Math.exp(-0.5 * safeDM2);
      densities[k] = Math.max(1e-15, pdf);

      // Temperature-scaled log responsibilities
      const rawLog = Math.log(Math.max(1e-12, c.pi)) + Math.log(densities[k]);
      logWeightedDensities[k] = rawLog / Math.max(0.15, temperature);
    }

    // Cumulative mixture density for total log-likelihood
    let mixtureDensity = 0;
    for (let k = 0; k < K; k++) {
      mixtureDensity += clusters[k].pi * densities[k];
    }
    totalLL += Math.log(Math.max(1e-12, mixtureDensity));

    // Soft responsibilities via Log-Sum-Exp trick
    const maxLog = Math.max(...logWeightedDensities);
    let sumExp = 0;
    const expWeights = new Array(K);
    for (let k = 0; k < K; k++) {
      const w = Math.exp(logWeightedDensities[k] - maxLog);
      expWeights[k] = w;
      sumExp += w;
    }

    const responsibilities = expWeights.map((w) => w / (sumExp || 1));

    // For hard K-Means mode when temperature is very low
    if (temperature < 0.28) {
      let bestK = 0;
      let maxR = -Infinity;
      for (let k = 0; k < K; k++) {
        if (responsibilities[k] > maxR) {
          maxR = responsibilities[k];
          bestK = k;
        }
      }
      for (let k = 0; k < K; k++) {
        responsibilities[k] = k === bestK ? 1.0 : 0.0;
      }
    }

    // Compute blended color from responsibility weights
    let blendedR = 0;
    let blendedG = 0;
    let blendedB = 0;
    let primaryK = 0;
    let primaryR = -1;
    let secondaryK = 0;
    let secondaryR = -1;

    for (let k = 0; k < K; k++) {
      const pal = CLUSTER_PALETTE[k % CLUSTER_PALETTE.length];
      const r_ik = responsibilities[k];
      blendedR += pal.r * r_ik;
      blendedG += pal.g * r_ik;
      blendedB += pal.b * r_ik;

      if (r_ik > primaryR) {
        secondaryR = primaryR;
        secondaryK = primaryK;
        primaryR = r_ik;
        primaryK = k;
      } else if (r_ik > secondaryR) {
        secondaryR = r_ik;
        secondaryK = k;
      }
    }

    return {
      ...p,
      responsibilities,
      blendedR: Math.round(blendedR),
      blendedG: Math.round(blendedG),
      blendedB: Math.round(blendedB),
      primaryCluster: primaryK,
      secondaryCluster: secondaryK,
      secondaryWeight: secondaryR,
    };
  });

  return { updatedPoints, logLikelihood: totalLL };
}

// M-Step: Update cluster centroids, covariances, and mixing weights
function evaluateMStep(
  points: GMMDataPoint[],
  currentClusters: GMMComponent[],
  stepRate: number,
  draggedIndex: number | null
): GMMComponent[] {
  const K = currentClusters.length;
  const N = points.length;

  const updatedClusters = currentClusters.map((cluster, k) => {
    // Effective cluster point count N_k
    let Nk = 0;
    for (let i = 0; i < N; i++) {
      Nk += points[i].responsibilities[k];
    }
    Nk = Math.max(1e-4, Nk);

    // Target mixing weight
    const targetPi = (Nk + 0.1) / (N + K * 0.1);

    // Target mean
    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < N; i++) {
      const gamma = points[i].responsibilities[k];
      sumX += gamma * points[i].x;
      sumY += gamma * points[i].y;
    }
    const targetMuX = sumX / Nk;
    const targetMuY = sumY / Nk;

    // Target full 2x2 covariance
    let sumXX = 0;
    let sumXY = 0;
    let sumYY = 0;
    for (let i = 0; i < N; i++) {
      const gamma = points[i].responsibilities[k];
      const dx = points[i].x - targetMuX;
      const dy = points[i].y - targetMuY;
      sumXX += gamma * dx * dx;
      sumXY += gamma * dx * dy;
      sumYY += gamma * dy * dy;
    }

    // Regularize covariance with minimum diagonal floor
    const minVar = 80;
    let targetXX = sumXX / Nk + minVar;
    let targetYY = sumYY / Nk + minVar;
    let targetXY = sumXY / Nk;

    // Positive-definiteness guarantee: xy^2 <= 0.95 * xx * yy
    const maxAllowedXY = Math.sqrt(targetXX * targetYY) * 0.95;
    targetXY = Math.max(-maxAllowedXY, Math.min(maxAllowedXY, targetXY));

    // Smooth visual morphing with stepRate
    const eta = Math.max(0.05, Math.min(1.0, stepRate));

    const nextMuX = (1 - eta) * cluster.mu.x + eta * targetMuX;
    const nextMuY = (1 - eta) * cluster.mu.y + eta * targetMuY;

    const nextXX = (1 - eta) * cluster.sigma.xx + eta * targetXX;
    const nextYY = (1 - eta) * cluster.sigma.yy + eta * targetYY;
    const nextXY = (1 - eta) * cluster.sigma.xy + eta * targetXY;
    const nextPi = (1 - eta) * cluster.pi + eta * targetPi;

    const nextSigma = { xx: nextXX, xy: nextXY, yy: nextYY };
    const { r1, r2, angle } = decomposeCovariance(nextSigma);

    return {
      ...cluster,
      mu: { x: nextMuX, y: nextMuY },
      sigma: nextSigma,
      pi: nextPi,
      r1,
      r2,
      angle,
    };
  });

  // Re-normalize mixing weights
  const sumPi = updatedClusters.reduce((acc, c) => acc + c.pi, 0);
  for (const c of updatedClusters) {
    c.pi = c.pi / (sumPi || 1);
  }

  return updatedClusters;
}

export const GMMClusteringPlugin: AlgorithmPlugin<GMMState, GMMParams> = {
  id: 'ai-kmeans-gmm',
  title: 'Gaussian Mixture Models (EM) & K-Means',
  shortDescription:
    'Unsupervised spatial clustering comparing hard K-Means Voronoi partitions against soft Expectation-Maximization (EM) covariance ellipses.',
  category: 'reinforcement-ai',

  controls: [
    {
      id: 'numClusters',
      label: 'Mixture Components (K)',
      type: 'slider',
      min: 2,
      max: 6,
      step: 1,
      defaultValue: 4,
      description: 'Number of Gaussian mixture components / cluster centroids K.',
    },
    {
      id: 'pointSpread',
      label: 'Cloud Spread (σ)',
      type: 'slider',
      min: 15,
      max: 55,
      step: 5,
      defaultValue: 32,
      unit: 'px',
      description: 'Spatial standard deviation of generated anisotropic point clouds.',
    },
    {
      id: 'softnessTemp',
      label: 'Softness Temperature (T)',
      type: 'slider',
      min: 0.2,
      max: 2.5,
      step: 0.1,
      defaultValue: 1.0,
      description: 'Responsibility temperature (low = hard K-Means, 1.0 = exact EM GMM).',
    },
    {
      id: 'stepRate',
      label: 'Morphing Rate (η)',
      type: 'slider',
      min: 0.1,
      max: 1.0,
      step: 0.1,
      defaultValue: 0.45,
      description: 'EM parameter interpolation rate per step for smooth visual morphing.',
    },
  ],

  documentation: {
    intuition:
      'While standard K-Means uses hard circular Voronoi boundaries, Gaussian Mixture Models (GMM) with Expectation-Maximization (EM) discover orientation, scale, and anisotropy through full 2x2 covariance matrices. The E-Step calculates soft posterior responsibilities gamma_ik, creating probabilistic color blends and competition lines at cluster frontiers. The M-Step maximizes log-likelihood by morphing rotated Gaussian ellipses to embrace the underlying data manifolds.',
    governingEquationLatex:
      '\\textcolor{#f59e0b}{\\gamma_{ik}} = \\frac{\\pi_k \\mathcal{N}(\\textcolor{#38bdf8}{\\mathbf{x}_i} \\mid \\textcolor{#10b981}{\\boldsymbol{\\mu}_k}, \\mathbf{\\Sigma}_k)}{\\sum_{j=1}^K \\pi_j \\mathcal{N}(\\textcolor{#38bdf8}{\\mathbf{x}_i} \\mid \\boldsymbol{\\mu}_j, \\mathbf{\\Sigma}_j)}, \\quad \\textcolor{#10b981}{\\boldsymbol{\\mu}_k^{\\text{new}}} = \\frac{1}{N_k}\\sum_{i=1}^N \\textcolor{#f59e0b}{\\gamma_{ik}}\\textcolor{#38bdf8}{\\mathbf{x}_i}',
    pseudocode: [
      'function GMM_EM_Clustering(X, K, temperature, eta):',
      '  // E-Step: Evaluate soft responsibility weights',
      '  gamma_{ik} = (pi_k * N(x_i | mu_k, Sigma_k))^(1/T) / sum_j (...)',
      '  // M-Step: Re-estimate mixture parameters',
      '  N_k = sum_i gamma_{ik};  pi_k_new = N_k / N',
      '  mu_k_new = (1/N_k) * sum_i (gamma_{ik} * x_i)',
      '  Sigma_k_new = (1/N_k) * sum_i [ gamma_{ik} * (x_i - mu_k)(x_i - mu_k)^T ]',
    ],
  },

  createInitialState(params: GMMParams): GMMState {
    const points = generateDataPoints(params.pointSpread);
    const clusters = createInitialComponents(params.numClusters);
    const { updatedPoints, logLikelihood } = evaluateEStep(points, clusters, params.softnessTemp);

    return {
      points: updatedPoints,
      clusters,
      draggedClusterIndex: null,
      logLikelihood,
      logLikelihoodHistory: [logLikelihood],
      iteration: 0,
      isConverged: false,
      pointSpread: params.pointSpread,
      activePseudocodeLine: 1,
    };
  },

  step(currentState: GMMState, params: GMMParams) {
    let clusters = currentState.clusters;
    let points = currentState.points;

    // Handle dynamic cluster count slider changes smoothly
    if (clusters.length !== params.numClusters) {
      if (params.numClusters > clusters.length) {
        const added = createInitialComponents(params.numClusters).slice(clusters.length);
        clusters = [...clusters, ...added];
      } else {
        clusters = clusters.slice(0, params.numClusters);
      }
    }

    // Handle point spread slider changes dynamically
    if (params.pointSpread !== currentState.pointSpread) {
      points = points.map((p) => {
        if (p.baseClusterIndex !== undefined && p.baseOffsetX !== undefined && p.baseOffsetY !== undefined) {
          const centers = [
            { x: -130, y: -85 },
            { x: 125, y: -75 },
            { x: 115, y: 100 },
            { x: -120, y: 95 },
          ];
          const c = centers[p.baseClusterIndex];
          return {
            ...p,
            x: c.x + p.baseOffsetX * params.pointSpread,
            y: c.y + p.baseOffsetY * params.pointSpread,
          };
        }
        return p;
      });
    }

    // 1. E-Step: Update responsibilities and log-likelihood
    const { updatedPoints, logLikelihood } = evaluateEStep(points, clusters, params.softnessTemp);

    // 2. M-Step: Re-estimate parameters and morph ellipses
    const updatedClusters = evaluateMStep(
      updatedPoints,
      clusters,
      params.stepRate,
      null
    );

    const history = [...currentState.logLikelihoodHistory, logLikelihood].slice(-55);
    const prevLL = currentState.logLikelihood;
    const deltaLL = Math.abs(logLikelihood - prevLL);
    const isConverged = deltaLL < 0.08 && currentState.iteration > 15;

    const nextIter = currentState.iteration + 1;
    const activePseudocodeLine = nextIter % 2 === 0 ? 2 : 5;

    return {
      nextState: {
        points: updatedPoints,
        clusters: updatedClusters,
        draggedClusterIndex: currentState.draggedClusterIndex,
        logLikelihood,
        logLikelihoodHistory: history,
        iteration: nextIter,
        isConverged,
        pointSpread: params.pointSpread,
        activePseudocodeLine,
      },
      activePseudocodeLine,
      isComplete: false,
    };
  },

  getMetrics(state: GMMState, params: GMMParams): MetricReadout[] {
    const isHard = params.softnessTemp < 0.28;
    const uncertainCount = state.points.filter((p) => p.secondaryWeight > 0.15).length;
    const uncertainPct = Math.round((uncertainCount / Math.max(1, state.points.length)) * 100);

    return [
      {
        id: 'll',
        label: 'Log-Likelihood log L',
        value: state.logLikelihood.toFixed(1),
        color: 'amber',
      },
      {
        id: 'mode',
        label: 'Assignment Mode',
        value: isHard ? 'Hard K-Means' : `Soft EM (T=${params.softnessTemp.toFixed(1)})`,
        color: isHard ? 'rose' : 'sky',
      },
      {
        id: 'uncertain',
        label: 'Boundary Ambiguity',
        value: `${uncertainPct}%`,
        color: uncertainPct > 20 ? 'amber' : 'emerald',
      },
      {
        id: 'status',
        label: 'EM Status',
        value: state.isConverged ? 'Converged' : 'Optimizing...',
        color: state.isConverged ? 'emerald' : 'sky',
      },
    ];
  },

  onPointerInteraction(state: GMMState, event: CanvasPointerEvent, params: GMMParams): GMMState {
    const clusters = state.clusters.map((c) => ({ ...c, mu: { ...c.mu }, sigma: { ...c.sigma } }));

    // Check if clicked near an existing cluster centroid
    let hitIndex: number | null = state.draggedClusterIndex;
    if (!event.isDragging) {
      let minDist = Infinity;
      let candidate = -1;
      for (let k = 0; k < clusters.length; k++) {
        const d = Math.hypot(event.worldX - clusters[k].mu.x, event.worldY - clusters[k].mu.y);
        if (d < minDist) {
          minDist = d;
          candidate = k;
        }
      }
      if (minDist < 42) {
        hitIndex = candidate;
      } else {
        hitIndex = null;
      }
    }

    if (hitIndex !== null && clusters[hitIndex]) {
      clusters[hitIndex].mu.x = event.worldX;
      clusters[hitIndex].mu.y = event.worldY;
      const { updatedPoints, logLikelihood } = evaluateEStep(
        state.points,
        clusters,
        params.softnessTemp
      );
      return {
        ...state,
        points: updatedPoints,
        clusters,
        logLikelihood,
        draggedClusterIndex: hitIndex,
        isConverged: false,
      };
    }

    // If clicked far from any centroid, inject a new data point to challenge EM
    if (!event.isDragging) {
      const newPoint: GMMDataPoint = {
        x: event.worldX,
        y: event.worldY,
        responsibilities: new Array(clusters.length).fill(1 / clusters.length),
        blendedR: 245,
        blendedG: 158,
        blendedB: 11,
        primaryCluster: 0,
        secondaryCluster: 1,
        secondaryWeight: 0.5,
      };
      const allPoints = [...state.points, newPoint];
      const { updatedPoints, logLikelihood } = evaluateEStep(
        allPoints,
        clusters,
        params.softnessTemp
      );

      return {
        ...state,
        points: updatedPoints,
        logLikelihood,
        draggedClusterIndex: null,
        isConverged: false,
      };
    }

    return {
      ...state,
      draggedClusterIndex: null,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: GMMState,
    params: GMMParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    const isHard = params.softnessTemp < 0.28;

    // 1. Responsibility Spider Lines for Uncertain Frontier Points (Soft EM Mode)
    if (!isHard) {
      ctx.lineWidth = 1.0;
      for (const p of state.points) {
        if (p.secondaryWeight > 0.18) {
          const c1 = state.clusters[p.primaryCluster];
          const c2 = state.clusters[p.secondaryCluster];

          if (c1 && c2) {
            // Line to secondary competing centroid
            ctx.strokeStyle = `rgba(245, 158, 11, ${p.secondaryWeight * 0.65})`;
            ctx.setLineDash([3, 4]);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(c2.mu.x, c2.mu.y);
            ctx.stroke();

            // Line to primary winning centroid
            const primWeight = p.responsibilities[p.primaryCluster] || 0.5;
            ctx.strokeStyle = `rgba(56, 189, 248, ${primWeight * 0.4})`;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(c1.mu.x, c1.mu.y);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }

    // 2. Gaussian Mixture Model Rotated Covariance Ellipses
    for (let k = 0; k < state.clusters.length; k++) {
      const c = state.clusters[k];
      const pal = CLUSTER_PALETTE[k % CLUSTER_PALETTE.length];
      const { r1, r2, angle } = c;

      ctx.save();
      ctx.translate(c.mu.x, c.mu.y);
      ctx.rotate(angle);

      // 2-Sigma Outer Uncertainty Envelope (95% density)
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(12, 2 * r1), Math.max(12, 2 * r2), 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${pal.r}, ${pal.g}, ${pal.b}, 0.05)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${pal.r}, ${pal.g}, ${pal.b}, 0.35)`;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 1-Sigma Inner Confidence Ellipse (68% density)
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(6, r1), Math.max(6, r2), 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${pal.r}, ${pal.g}, ${pal.b}, 0.14)`;
      ctx.fill();
      ctx.strokeStyle = pal.hex;
      ctx.lineWidth = 2.2;
      ctx.stroke();

      // Principal axis crosshairs showing anisotropic orientation
      ctx.strokeStyle = `rgba(${pal.r}, ${pal.g}, ${pal.b}, 0.4)`;
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(-r1, 0);
      ctx.lineTo(r1, 0);
      ctx.moveTo(0, -r2);
      ctx.lineTo(0, r2);
      ctx.stroke();

      ctx.restore();

      // Centroid Marker μ_k with glow
      ctx.save();
      ctx.translate(c.mu.x, c.mu.y);

      ctx.shadowColor = pal.hex;
      ctx.shadowBlur = 10;
      ctx.fillStyle = pal.hex;
      ctx.beginPath();
      // Diamond marker
      ctx.moveTo(0, -8);
      ctx.lineTo(8, 0);
      ctx.lineTo(0, 8);
      ctx.lineTo(-8, 0);
      ctx.closePath();
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(0, 0, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Cluster centroid telemetry label
      ctx.font = '600 10px monospace';
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'center';
      const pct = Math.round(c.pi * 100);
      ctx.fillText(`μ${k + 1} (π=${pct}%)`, 0, -14);

      ctx.restore();
    }

    // 3. Colored Data Points with Responsibility Blending
    for (const p of state.points) {
      ctx.save();
      ctx.translate(p.x, p.y);

      ctx.beginPath();
      ctx.arc(0, 0, 4.5, 0, Math.PI * 2);

      // Blended RGB fill from cluster responsibilities
      ctx.fillStyle = `rgb(${p.blendedR}, ${p.blendedG}, ${p.blendedB})`;
      ctx.fill();

      // Subtle border ring
      ctx.strokeStyle = p.secondaryWeight > 0.25 ? '#f59e0b' : 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.restore();
    }

    // 4. Log-Likelihood Evolution HUD Sparkline Chart
    const hudW = 155;
    const hudH = 58;
    const hudX = width / 2 - hudW - 20;
    const hudY = height / 2 - hudH - 20;

    ctx.save();
    ctx.translate(hudX, hudY);

    // Glass panel
    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.7)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(0, 0, hudW, hudH, 6);
    ctx.fill();
    ctx.stroke();

    // Title
    ctx.font = '600 9px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('LOG-LIKELIHOOD log L', 8, 14);

    // Current value
    ctx.font = '700 11px monospace';
    ctx.fillStyle = '#f59e0b';
    ctx.fillText(state.logLikelihood.toFixed(1), hudW - 54, 14);

    // Sparkline curve
    const history = state.logLikelihoodHistory;
    if (history.length > 2) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const val of history) {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
      const span = maxVal - minVal || 1;

      const chartX = 8;
      const chartY = 22;
      const chartW = hudW - 16;
      const chartH = hudH - 30;

      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const xPos = chartX + (i / (history.length - 1)) * chartW;
        const normY = (history[i] - minVal) / span;
        const yPos = chartY + chartH - normY * chartH;
        if (i === 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
      }
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.8;
      ctx.stroke();

      // Current point dot
      const lastX = chartX + chartW;
      const lastNormY = (history[history.length - 1] - minVal) / span;
      const lastY = chartY + chartH - lastNormY * chartH;
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    ctx.restore(); // Centered viewport restore

    // 5. On-Canvas Interactive Instruction Header
    ctx.save();
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(
      'Click / Drag Centroids μ_k to perturb EM | Click empty canvas to inject new data points',
      16,
      24
    );
    ctx.restore();
  },
};

export default GMMClusteringPlugin;
