// FILE: src/modules/perception-particle-filter/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface Particle {
  x: number;
  y: number;
  theta: number;
  weight: number;
}

export interface Landmark {
  id: number;
  x: number;
  y: number;
  label: string;
}

export interface ParticleFilterState {
  timeStep: number;
  // True robot pose
  trueX: number;
  trueY: number;
  trueTheta: number;
  // Offset applied when robot is kidnapped so it continues moving from its new location
  offsetX: number;
  offsetY: number;
  // Estimated pose (weighted mean of belief)
  estX: number;
  estY: number;
  estTheta: number;
  // Particle distribution
  particles: Particle[];
  // Landmarks in the arena
  landmarks: Landmark[];
  // Sensor rays from true robot to landmarks
  sensorRays: {
    landmarkId: number;
    distance: number;
    bearing: number;
    lx: number;
    ly: number;
  }[];
  // Telemetry
  effectiveParticles: number;
  estimationError: number;
  isKidnapped: boolean;
  trailTrue: { x: number; y: number }[];
  trailEst: { x: number; y: number }[];
}

export interface ParticleFilterParams {
  particleCount: number;
  motionNoise: number;
  sensorNoise: number;
  resampleRatio: number;
  showSensorBeams: boolean;
}

function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function normalizeAngle(angle: number): number {
  let a = angle;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

const DEFAULT_LANDMARKS: Landmark[] = [
  { id: 1, x: -190, y: -120, label: 'L1' },
  { id: 2, x: 0, y: -140, label: 'L2' },
  { id: 3, x: 190, y: -110, label: 'L3' },
  { id: 4, x: 180, y: 120, label: 'L4' },
  { id: 5, x: 0, y: 140, label: 'L5' },
  { id: 6, x: -180, y: 110, label: 'L6' }
];

const ARENA_HALF_W = 230;
const ARENA_HALF_H = 165;

function figureEightOffset(step: number): { x: number; y: number } {
  const t = step * 0.04;
  return {
    x: 125 * Math.sin(t),
    y: 75 * Math.sin(2 * t) * 0.5
  };
}

export const ParticleFilterPlugin: AlgorithmPlugin<ParticleFilterState, ParticleFilterParams> = {
  id: 'perception-particle-filter',
  title: 'Monte Carlo Localization (MCL / Particle Filter)',
  shortDescription: 'Global Bayesian localization using weighted hypothesis tracking and sensor likelihood updates.',
  category: 'perception-slam',

  controls: [
    {
      id: 'particleCount',
      label: 'Particle Count (N)',
      type: 'slider',
      min: 100,
      max: 600,
      step: 50,
      defaultValue: 300,
      description: 'Number of parallel hypotheses tracked in state space.'
    },
    {
      id: 'motionNoise',
      label: 'Motion Noise (sigma_u)',
      type: 'slider',
      min: 0.5,
      max: 4.0,
      step: 0.5,
      defaultValue: 1.5,
      unit: 'px'
    },
    {
      id: 'sensorNoise',
      label: 'Sensor Range Noise (sigma_z)',
      type: 'slider',
      min: 4,
      max: 30,
      step: 2,
      defaultValue: 12,
      unit: 'px'
    },
    {
      id: 'resampleRatio',
      label: 'Resampling Threshold (N_eff / N)',
      type: 'slider',
      min: 0.2,
      max: 0.9,
      step: 0.05,
      defaultValue: 0.65
    },
    {
      id: 'showSensorBeams',
      label: 'Show Landmark Sensor Beams',
      type: 'toggle',
      defaultValue: true
    }
  ],

  documentation: {
    intuition:
      'Unlike Kalman filters which assume a unimodal Gaussian belief, Monte Carlo Localization (MCL) represents arbitrary, multi-modal probability densities using a cloud of weighted particles. Particles propagate through noisy motion dynamics, receive weights proportional to landmark measurement likelihoods, and are resampled to focus computational mass around true hypotheses.',
    governingEquationLatex:
      '\\textcolor{#0ea5e9}{x_t^{[i]}} \\sim p(\\textcolor{#0ea5e9}{x_t} \\mid \\textcolor{#38bdf8}{x_{t-1}^{[i]}}, u_t), \\quad \\textcolor{#f59e0b}{w_t^{[i]}} \\propto \\prod_{k=1}^6 \\exp\\left(-\\frac{(\\textcolor{#10b981}{z_t^k} - \\textcolor{#0ea5e9}{\\hat{z}_t^{k,[i]}})^2}{2\\textcolor{#f59e0b}{\\sigma_z^2}}\\right)',
    pseudocode: [
      '1. Motion Update: Propagate particles x_i ~ p(x_t | x_{t-1}, u_t) + N(0, sigma_u)',
      '2. Read Sensor: Measure range z_k from true robot to all 6 landmarks',
      '3. Weight Particles: Evaluate likelihood w_i = exp(-sum((z_k - z_pred_k)^2) / 2*sigma^2)',
      '4. Normalize Weights: Scale weights so that sum(w_i) = 1.0',
      '5. Estimate Belief: Compute weighted mean pose (x_est, y_est, theta_est)',
      '6. Evaluate N_eff: Calculate effective sample size N_eff = 1 / sum(w_i^2)',
      '7. Low-Variance Resample: If N_eff < threshold, draw next generation of particles'
    ]
  },

  createInitialState(params: ParticleFilterParams): ParticleFilterState {
    const N = params.particleCount || 300;
    const initialParticles: Particle[] = [];
    const uniformWeight = 1.0 / N;

    for (let i = 0; i < N; i++) {
      initialParticles.push({
        x: (Math.random() * 2 - 1) * ARENA_HALF_W,
        y: (Math.random() * 2 - 1) * ARENA_HALF_H,
        theta: Math.random() * 2 * Math.PI - Math.PI,
        weight: uniformWeight
      });
    }

    const startCurve = figureEightOffset(0);
    const startX = startCurve.x;
    const startY = startCurve.y;
    const startTheta = 0;

    return {
      timeStep: 0,
      trueX: startX,
      trueY: startY,
      trueTheta: startTheta,
      offsetX: 0,
      offsetY: 0,
      estX: 0,
      estY: 0,
      estTheta: 0,
      particles: initialParticles,
      landmarks: DEFAULT_LANDMARKS,
      sensorRays: [],
      effectiveParticles: N,
      estimationError: Math.hypot(startX, startY),
      isKidnapped: false,
      trailTrue: [{ x: startX, y: startY }],
      trailEst: [{ x: 0, y: 0 }]
    };
  },

  step(currentState: ParticleFilterState, params: ParticleFilterParams) {
    const nextStep = currentState.timeStep + 1;
    const N = currentState.particles.length;

    // 1. Advance True Robot along smooth patrol trajectory relative to current offset
    const prevCurve = figureEightOffset(currentState.timeStep);
    const nextCurve = figureEightOffset(nextStep);
    const rawDx = nextCurve.x - prevCurve.x;
    const rawDy = nextCurve.y - prevCurve.y;

    const nextTrueX = Math.max(
      -ARENA_HALF_W + 10,
      Math.min(ARENA_HALF_W - 10, currentState.trueX + rawDx)
    );
    const nextTrueY = Math.max(
      -ARENA_HALF_H + 10,
      Math.min(ARENA_HALF_H - 10, currentState.trueY + rawDy)
    );

    const deltaX = nextTrueX - currentState.trueX;
    const deltaY = nextTrueY - currentState.trueY;
    const nextTrueTheta =
      Math.hypot(deltaX, deltaY) > 0.01
        ? Math.atan2(deltaY, deltaX)
        : currentState.trueTheta;
    const deltaTheta = normalizeAngle(nextTrueTheta - currentState.trueTheta);

    // 2. Predict / Propagate particles with motion noise
    const motionNoise = params.motionNoise;
    const movedParticles: Particle[] = currentState.particles.map((p) => {
      const noisyDx = deltaX + gaussianRandom() * motionNoise;
      const noisyDy = deltaY + gaussianRandom() * motionNoise;
      const noisyDTheta = deltaTheta + gaussianRandom() * (motionNoise * 0.02);

      let px = p.x + noisyDx;
      let py = p.y + noisyDy;
      if (px < -ARENA_HALF_W) px = -ARENA_HALF_W + Math.random() * 5;
      if (px > ARENA_HALF_W) px = ARENA_HALF_W - Math.random() * 5;
      if (py < -ARENA_HALF_H) py = -ARENA_HALF_H + Math.random() * 5;
      if (py > ARENA_HALF_H) py = ARENA_HALF_H - Math.random() * 5;

      return {
        x: px,
        y: py,
        theta: normalizeAngle(p.theta + noisyDTheta),
        weight: p.weight
      };
    });

    // 3. Sensor measurement from true robot to all 6 landmarks
    const sensorNoise = params.sensorNoise;
    const sensorRays = currentState.landmarks.map((lm) => {
      const exactDist = Math.hypot(nextTrueX - lm.x, nextTrueY - lm.y);
      const measuredDist = Math.max(5, exactDist + gaussianRandom() * sensorNoise);
      const bearing = Math.atan2(lm.y - nextTrueY, lm.x - nextTrueX) - nextTrueTheta;
      return {
        landmarkId: lm.id,
        distance: measuredDist,
        bearing: normalizeAngle(bearing),
        lx: lm.x,
        ly: lm.y
      };
    });

    // 4. Update Particle Weights via Gaussian Sensor Likelihood
    const twoSigmaSq = 2 * sensorNoise * sensorNoise;
    const rawWeights: number[] = new Array(N);
    let totalWeight = 0;

    for (let i = 0; i < N; i++) {
      const p = movedParticles[i];
      let logLikelihood = 0;

      for (let k = 0; k < sensorRays.length; k++) {
        const ray = sensorRays[k];
        const predDist = Math.hypot(p.x - ray.lx, p.y - ray.ly);
        const diff = ray.distance - predDist;
        logLikelihood -= (diff * diff) / twoSigmaSq;
      }

      const weight = Math.exp(Math.max(-45, logLikelihood));
      rawWeights[i] = weight;
      totalWeight += weight;
    }

    // Detect measurement likelihood collapse (e.g., after kidnapping)
    const avgLikelihood = totalWeight / N;
    const likelihoodCollapsed = avgLikelihood < 1e-4 || currentState.isKidnapped;

    // 5. Normalize weights
    const normalizedParticles: Particle[] = new Array(N);
    let sumSqWeights = 0;

    if (totalWeight > 1e-15) {
      for (let i = 0; i < N; i++) {
        const nw = rawWeights[i] / totalWeight;
        normalizedParticles[i] = {
          ...movedParticles[i],
          weight: nw
        };
        sumSqWeights += nw * nw;
      }
    } else {
      const uniformW = 1.0 / N;
      for (let i = 0; i < N; i++) {
        normalizedParticles[i] = {
          ...movedParticles[i],
          weight: uniformW
        };
      }
      sumSqWeights = 1.0 / N;
    }

    const nEff = sumSqWeights > 0 ? 1.0 / sumSqWeights : N;

    // 6. Compute Weighted Mean Estimate
    let estX = 0;
    let estY = 0;
    let cosSum = 0;
    let sinSum = 0;

    for (let i = 0; i < N; i++) {
      const p = normalizedParticles[i];
      estX += p.x * p.weight;
      estY += p.y * p.weight;
      cosSum += Math.cos(p.theta) * p.weight;
      sinSum += Math.sin(p.theta) * p.weight;
    }
    const estTheta = Math.atan2(sinSum, cosSum);
    const estimationError = Math.hypot(estX - nextTrueX, estY - nextTrueY);

    // 7. Low-Variance Resampling + Augmented MCL Global Recovery
    let resampledParticles: Particle[] = normalizedParticles;
    const resampleThreshold = params.resampleRatio * N;
    let activeLine = nextStep % 6;

    if (nEff < resampleThreshold || likelihoodCollapsed) {
      activeLine = 6;
      resampledParticles = new Array(N);
      const randomExplorerRatio = likelihoodCollapsed ? 0.2 : 0.0;
      const numRandom = Math.floor(N * randomExplorerRatio);
      const numResampled = N - numRandom;

      const stepSize = 1.0 / numResampled;
      let r = Math.random() * stepSize;
      let cumulative = normalizedParticles[0].weight;
      let idx = 0;

      for (let m = 0; m < numResampled; m++) {
        const u = r + m * stepSize;
        while (u > cumulative && idx < N - 1) {
          idx++;
          cumulative += normalizedParticles[idx].weight;
        }

        const base = normalizedParticles[idx];
        const jitter = likelihoodCollapsed ? 4.5 : 1.5;
        resampledParticles[m] = {
          x: Math.max(-ARENA_HALF_W, Math.min(ARENA_HALF_W, base.x + gaussianRandom() * jitter)),
          y: Math.max(-ARENA_HALF_H, Math.min(ARENA_HALF_H, base.y + gaussianRandom() * jitter)),
          theta: normalizeAngle(base.theta + gaussianRandom() * 0.06),
          weight: 1.0 / N
        };
      }

      // Augmented MCL: inject uniform global explorer particles when kidnapped
      for (let m = numResampled; m < N; m++) {
        resampledParticles[m] = {
          x: (Math.random() * 2 - 1) * ARENA_HALF_W,
          y: (Math.random() * 2 - 1) * ARENA_HALF_H,
          theta: Math.random() * 2 * Math.PI - Math.PI,
          weight: 1.0 / N
        };
      }
    }

    const maxTrail = 80;
    const trailTrue = [...currentState.trailTrue, { x: nextTrueX, y: nextTrueY }].slice(-maxTrail);
    const trailEst = [...currentState.trailEst, { x: estX, y: estY }].slice(-maxTrail);

    return {
      nextState: {
        timeStep: nextStep,
        trueX: nextTrueX,
        trueY: nextTrueY,
        trueTheta: nextTrueTheta,
        offsetX: currentState.offsetX,
        offsetY: currentState.offsetY,
        estX,
        estY,
        estTheta,
        particles: resampledParticles,
        landmarks: currentState.landmarks,
        sensorRays,
        effectiveParticles: nEff,
        estimationError,
        isKidnapped: estimationError > 25,
        trailTrue,
        trailEst
      },
      isComplete: false,
      activePseudocodeLine: activeLine
    };
  },

  onPointerInteraction(
    currentState: ParticleFilterState,
    event: CanvasPointerEvent
  ): ParticleFilterState {
    // Kidnapped Robot Interaction:
    // Teleports the ground truth robot and injects global exploratory hypotheses
    const clampedX = Math.max(-ARENA_HALF_W + 10, Math.min(ARENA_HALF_W - 10, event.worldX));
    const clampedY = Math.max(-ARENA_HALF_H + 10, Math.min(ARENA_HALF_H - 10, event.worldY));

    const curve = figureEightOffset(currentState.timeStep);
    const N = currentState.particles.length;

    // Keep 70% of particles at old belief, scatter 30% globally to initiate relocalization search
    const updatedParticles = currentState.particles.map((p, idx) => {
      if (idx < N * 0.7) return { ...p };
      return {
        x: (Math.random() * 2 - 1) * ARENA_HALF_W,
        y: (Math.random() * 2 - 1) * ARENA_HALF_H,
        theta: Math.random() * 2 * Math.PI - Math.PI,
        weight: 1.0 / N
      };
    });

    return {
      ...currentState,
      trueX: clampedX,
      trueY: clampedY,
      offsetX: clampedX - curve.x,
      offsetY: clampedY - curve.y,
      particles: updatedParticles,
      isKidnapped: true,
      estimationError: Math.hypot(currentState.estX - clampedX, currentState.estY - clampedY),
      trailTrue: [{ x: clampedX, y: clampedY }]
    };
  },

  getMetrics(currentState: ParticleFilterState, params: ParticleFilterParams): MetricReadout[] {
    const errorColor =
      currentState.estimationError < 15
        ? 'emerald'
        : currentState.estimationError < 40
        ? 'amber'
        : 'rose';

    const nEffRatio = (currentState.effectiveParticles / params.particleCount) * 100;

    return [
      {
        id: 'estError',
        label: 'Position Error',
        value: currentState.estimationError.toFixed(1),
        unit: ' px',
        color: errorColor
      },
      {
        id: 'nEff',
        label: 'Effective Sample Size (N_eff)',
        value: `${Math.round(currentState.effectiveParticles)} (${nEffRatio.toFixed(0)}%)`,
        color: 'sky'
      },
      {
        id: 'status',
        label: 'Status',
        value: currentState.isKidnapped ? 'KIDNAPPED! Relocalizing...' : 'Tracking Locked',
        color: currentState.isKidnapped ? 'rose' : 'emerald'
      }
    ];
  },

  render(ctx, currentState, params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    // 1. Draw Arena Boundary
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
    ctx.beginPath();
    ctx.roundRect(
      -ARENA_HALF_W - 15,
      -ARENA_HALF_H - 15,
      (ARENA_HALF_W + 15) * 2,
      (ARENA_HALF_H + 15) * 2,
      12
    );
    ctx.fill();
    ctx.stroke();

    // 2. Draw Landmark Sensor Beams (Amber #f59e0b)
    if (params.showSensorBeams && currentState.sensorRays.length > 0) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
      ctx.setLineDash([4, 4]);
      for (const ray of currentState.sensorRays) {
        ctx.beginPath();
        ctx.moveTo(currentState.trueX, currentState.trueY);
        ctx.lineTo(ray.lx, ray.ly);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // 3. Draw Landmarks (Emerald #10b981)
    for (const lm of currentState.landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x, lm.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.font = '10px monospace';
      ctx.fillStyle = '#6ee7b7';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(lm.label, lm.x, lm.y - 10);
    }

    // 4. Draw Trajectories
    const drawTrail = (trail: { x: number; y: number }[], color: string) => {
      if (trail.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.moveTo(trail[0].x, trail[0].y);
      for (let i = 1; i < trail.length; i++) {
        ctx.lineTo(trail[i].x, trail[i].y);
      }
      ctx.stroke();
    };

    drawTrail(currentState.trailTrue, 'rgba(16, 185, 129, 0.5)');
    drawTrail(currentState.trailEst, 'rgba(14, 165, 233, 0.7)');

    // 5. Draw Particles (Sky Blue #0ea5e9 / #38bdf8)
    const maxWeight = Math.max(...currentState.particles.map((p) => p.weight), 1e-6);
    for (const p of currentState.particles) {
      const alpha = Math.min(1.0, 0.2 + (p.weight / maxWeight) * 0.8);
      ctx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
      ctx.fill();

      const headingLen = 6;
      ctx.strokeStyle = `rgba(14, 165, 233, ${alpha * 0.8})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + Math.cos(p.theta) * headingLen, p.y + Math.sin(p.theta) * headingLen);
      ctx.stroke();
    }

    // 6. Draw Estimated Robot Belief (Sky Blue #0ea5e9)
    ctx.beginPath();
    ctx.arc(currentState.estX, currentState.estY, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(currentState.estX, currentState.estY);
    ctx.lineTo(
      currentState.estX + Math.cos(currentState.estTheta) * 16,
      currentState.estY + Math.sin(currentState.estTheta) * 16
    );
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 7. Draw Ground Truth Robot (Emerald #10b981)
    ctx.beginPath();
    ctx.arc(currentState.trueX, currentState.trueY, 11, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(currentState.trueX, currentState.trueY);
    ctx.lineTo(
      currentState.trueX + Math.cos(currentState.trueTheta) * 18,
      currentState.trueY + Math.sin(currentState.trueTheta) * 18
    );
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.stroke();

    // 8. If Kidnapped / High Error: Draw Rose Error Line (#f43f5e)
    if (currentState.estimationError > 20) {
      ctx.beginPath();
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.moveTo(currentState.trueX, currentState.trueY);
      ctx.lineTo(currentState.estX, currentState.estY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const midX = (currentState.trueX + currentState.estX) / 2;
      const midY = (currentState.trueY + currentState.estY) / 2;
      ctx.fillText(`Error: ${currentState.estimationError.toFixed(0)}px`, midX, midY - 12);
    }

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Click / Drag anywhere to kidnap true robot & trigger global relocalization', 0, ARENA_HALF_H + 30);

    ctx.restore();
  }
};
