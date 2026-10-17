// src/modules/perception-kalman/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface KalmanState {
  timeStep: number;
  // Ground truth position
  trueX: number;
  trueY: number;
  // Odometry-only (dead reckoning) position (drifts over time)
  odomX: number;
  odomY: number;
  // Kalman Filter Estimated Belief (mu_x, mu_y)
  estX: number;
  estY: number;
  // Latest noisy sensor measurement (z_x, z_y)
  measX: number;
  measY: number;
  // 2D Uncertainty Variance (sigma_x^2, sigma_y^2)
  varX: number;
  varY: number;
  kalmanGain: number;
  trailTrue: { x: number; y: number }[];
  trailEst: { x: number; y: number }[];
  trailOdom: { x: number; y: number }[];
}

export interface KalmanParams {
  processNoiseQ: number;     // Wheel slip / motion uncertainty
  measurementNoiseR: number; // GPS / LiDAR sensor noise
  showOdometryDrift: boolean;
}

function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export const KalmanFilterPlugin: AlgorithmPlugin<KalmanState, KalmanParams> = {
  id: 'perception-2d-kalman',
  title: '2D Kalman Filter Sensor Fusion',
  shortDescription: 'Real-time Bayesian fusion of drifting wheel odometry with noisy GPS/LiDAR measurements.',
  category: 'perception-slam',

  controls: [
    {
      id: 'processNoiseQ',
      label: 'Motion Noise / Wheel Slip (Q)',
      type: 'slider',
      min: 0.5,
      max: 15.0,
      step: 0.5,
      defaultValue: 3.0
    },
    {
      id: 'measurementNoiseR',
      label: 'Sensor / GPS Noise (R)',
      type: 'slider',
      min: 2.0,
      max: 60.0,
      step: 2.0,
      defaultValue: 22.0
    },
    {
      id: 'showOdometryDrift',
      label: 'Show Unfiltered Odometry Drift',
      type: 'toggle',
      defaultValue: true
    }
  ],

  documentation: {
    intuition:
      'Dead reckoning (wheel odometry) drifts endlessly, while raw GPS sensors jump erratically. The Kalman Filter computes an optimal Kalman Gain (K) based on the ratio of motion uncertainty (P) to sensor noise (R), shrinking the uncertainty ellipse every update.',
    governingEquationLatex:
      '\\hat{x}_t = \\textcolor{#0ea5e9}{\\hat{x}_{t|t-1}} + K_t (\\textcolor{#f59e0b}{z_t} - \\textcolor{#0ea5e9}{\\hat{x}_{t|t-1}}), \\quad K_t = \\frac{P}{P + R}',
    pseudocode: [
      '1. Predict State: x_pred = x_est + u_motion',
      '2. Grow Uncertainty: P_pred = P_est + Q (Process Noise)',
      '3. Read Noisy Sensor: z = x_true + N(0, R)',
      '4. Compute Kalman Gain: K = P_pred / (P_pred + R)',
      '5. Fuse Estimate: x_est = x_pred + K · (z - x_pred)',
      '6. Shrink Covariance: P_est = (1 - K) · P_pred'
    ]
  },

  createInitialState(): KalmanState {
    const r = 130;
    return {
      timeStep: 0,
      trueX: r,
      trueY: 0,
      odomX: r,
      odomY: 0,
      estX: r,
      estY: 0,
      measX: r,
      measY: 0,
      varX: 25,
      varY: 25,
      kalmanGain: 0.5,
      trailTrue: [{ x: r, y: 0 }],
      trailEst: [{ x: r, y: 0 }],
      trailOdom: [{ x: r, y: 0 }]
    };
  },

  step(state: KalmanState, params: KalmanParams) {
    const nextStep = state.timeStep + 1;
    const radius = 130;
    const omega = 0.06;

    // 1. Move Ground Truth along circular trajectory
    const nextTrueX = radius * Math.cos(nextStep * omega);
    const nextTrueY = radius * Math.sin(nextStep * omega);
    const dx = nextTrueX - state.trueX;
    const dy = nextTrueY - state.trueY;

    // 2. Simulate noisy motion command (wheel slip)
    const slipX = gaussianRandom() * params.processNoiseQ;
    const slipY = gaussianRandom() * params.processNoiseQ;
    const nextOdomX = state.odomX + dx + slipX;
    const nextOdomY = state.odomY + dy + slipY;

    // 3. Kalman Prediction Step
    const predX = state.estX + dx + slipX;
    const predY = state.estY + dy + slipY;
    const predVarX = state.varX + params.processNoiseQ * params.processNoiseQ;
    const predVarY = state.varY + params.processNoiseQ * params.processNoiseQ;

    // 4. Simulate Noisy GPS Measurement z_t
    const measX = nextTrueX + gaussianRandom() * params.measurementNoiseR;
    const measY = nextTrueY + gaussianRandom() * params.measurementNoiseR;

    // 5. Kalman Update Step
    const RVar = params.measurementNoiseR * params.measurementNoiseR;
    const Kx = predVarX / (predVarX + RVar);
    const Ky = predVarY / (predVarY + RVar);

    const estX = predX + Kx * (measX - predX);
    const estY = predY + Ky * (measY - predY);
    const varX = (1 - Kx) * predVarX;
    const varY = (1 - Ky) * predVarY;

    const maxTrail = 90;
    const trailTrue = [...state.trailTrue, { x: nextTrueX, y: nextTrueY }].slice(-maxTrail);
    const trailEst = [...state.trailEst, { x: estX, y: estY }].slice(-maxTrail);
    const trailOdom = [...state.trailOdom, { x: nextOdomX, y: nextOdomY }].slice(-maxTrail);

    return {
      nextState: {
        timeStep: nextStep,
        trueX: nextTrueX,
        trueY: nextTrueY,
        odomX: nextOdomX,
        odomY: nextOdomY,
        estX,
        estY,
        measX,
        measY,
        varX,
        varY,
        kalmanGain: Kx,
        trailTrue,
        trailEst,
        trailOdom
      },
      isComplete: false,
      activePseudocodeLine: 4
    };
  },

  onPointerInteraction(state: KalmanState, event: CanvasPointerEvent): KalmanState {
    // Kidnapped Robot Test: Clicking teleports the true robot to test filter recovery!
    return {
      ...state,
      trueX: event.worldX,
      trueY: event.worldY,
      measX: event.worldX,
      measY: event.worldY,
      varX: 800,
      varY: 800
    };
  },

  getMetrics(state: KalmanState): MetricReadout[] {
    const kfError = Math.hypot(state.estX - state.trueX, state.estY - state.trueY).toFixed(1);
    const odomError = Math.hypot(state.odomX - state.trueX, state.odomY - state.trueY).toFixed(1);
    return [
      { id: 'gain', label: 'Kalman Gain (K)', value: state.kalmanGain.toFixed(2), color: 'sky' },
      { id: 'kferr', label: 'KF Error', value: `${kfError} px`, color: 'emerald' },
      { id: 'odomerr', label: 'Odometry Drift', value: `${odomError} px`, color: 'rose' }
    ];
  },

  render(ctx, state, params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    const drawPath = (pts: { x: number; y: number }[], color: string, dashed = false) => {
      if (pts.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dashed ? [4, 4] : []);
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // 1. Draw Trajectories
    if (params.showOdometryDrift) {
      drawPath(state.trailOdom, 'rgba(244, 63, 94, 0.5)', true); // Rose: Drifting Odometry
    }
    drawPath(state.trailTrue, 'rgba(16, 185, 129, 0.6)'); // Emerald: Ground Truth
    drawPath(state.trailEst, '#0ea5e9'); // Sky: Kalman Estimate

    // 2. Draw 95% Confidence Covariance Ellipse around Estimate
    const rx = Math.max(6, Math.sqrt(state.varX) * 2.45);
    const ry = Math.max(6, Math.sqrt(state.varY) * 2.45);
    ctx.beginPath();
    ctx.ellipse(state.estX, state.estY, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(14, 165, 233, 0.18)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 3. Draw Noisy Sensor Ping (Amber)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(state.measX - 6, state.measY - 6);
    ctx.lineTo(state.measX + 6, state.measY + 6);
    ctx.moveTo(state.measX + 6, state.measY - 6);
    ctx.lineTo(state.measX - 6, state.measY + 6);
    ctx.stroke();

    // 4. Draw Ground Truth Robot (Emerald) & Kalman Belief (Sky)
    ctx.beginPath();
    ctx.arc(state.trueX, state.trueY, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(state.estX, state.estY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.restore();
  }
};
