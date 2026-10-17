// FILE: src/modules/control-pure-pursuit/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface PurePursuitState {
  x: number;
  y: number;
  theta: number;
  delta: number;
  alpha: number;
  path: { x: number; y: number }[];
  lookaheadPoint: { x: number; y: number };
  closestPoint: { x: number; y: number; idx: number };
  crossTrackError: number;
  curvature: number;
  trail: { x: number; y: number }[];
  iteration: number;
}

export interface PurePursuitParams {
  lookaheadDistance: number;
  vehicleSpeed: number;
  wheelbase: number;
  pathType: string;
}

function buildReferencePath(shape: string): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  const N = 160;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 2;
    if (shape === 'oval') {
      pts.push({
        x: 215 * Math.cos(t),
        y: 115 * Math.sin(t),
      });
    } else if (shape === 'scurve') {
      // Closed-loop twisted S-chicane race circuit
      pts.push({
        x: 215 * Math.cos(t),
        y: 95 * Math.sin(t) + 42 * Math.sin(3 * t),
      });
    } else {
      pts.push({
        x: 225 * Math.sin(t),
        y: 120 * Math.sin(2 * t),
      });
    }
  }
  return pts;
}

function findTrackingPoints(
  x: number,
  y: number,
  path: { x: number; y: number }[],
  Ld: number
): {
  closestPoint: { x: number; y: number; idx: number };
  lookaheadPoint: { x: number; y: number };
  crossTrackError: number;
} {
  let bestDist = Infinity;
  let bestIdx = 0;
  for (let i = 0; i < path.length; i++) {
    const d = Math.hypot(path[i].x - x, path[i].y - y);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }

  let lookIdx = bestIdx;
  for (let step = 1; step < path.length; step++) {
    const idx = (bestIdx + step) % path.length;
    const d = Math.hypot(path[idx].x - x, path[idx].y - y);
    if (d >= Ld) {
      lookIdx = idx;
      break;
    }
  }

  return {
    closestPoint: { x: path[bestIdx].x, y: path[bestIdx].y, idx: bestIdx },
    lookaheadPoint: { x: path[lookIdx].x, y: path[lookIdx].y },
    crossTrackError: bestDist,
  };
}

export const PurePursuitPlugin: AlgorithmPlugin<PurePursuitState, PurePursuitParams> = {
  id: 'control-pure-pursuit',
  title: 'Pure Pursuit Autonomous Path Tracking',
  shortDescription:
    'Kinematic Ackermann bicycle vehicle tracking a continuous reference spline using a dynamic lookahead circle (Ld), constant-curvature pursuit arc, and steering law δ = atan(2L sin(α) / Ld).',
  category: 'kinematics-control',

  documentation: {
    intuition:
      'Just like human highway driving: if you stare right at the hood of your car (tiny Ld), you overcorrect and fishtail wildly. If you look too far ahead (huge Ld), you cut corners on sharp turns. Pure Pursuit formalizes this exact trade-off with a single geometric lookahead circle.',
    governingEquationLatex:
      '\\textcolor{#38bdf8}{\\delta} = \\arctan\\!\\left(\\frac{2 L \\sin(\\textcolor{#f59e0b}{\\alpha})}{\\textcolor{#f59e0b}{L_d}}\\right), \\quad \\kappa = \\frac{2 \\sin(\\textcolor{#f59e0b}{\\alpha})}{\\textcolor{#f59e0b}{L_d}}, \\quad \\textcolor{#f43f5e}{e_{\\text{cte}}} = \\|\\textcolor{#0ea5e9}{\\mathbf{p}_{\\text{rear}}} - \\textcolor{#10b981}{\\mathbf{p}_{\\text{ref}}}\\|',
    pseudocode: [
      'function PurePursuit_Step(pose (x, y, θ), path, Ld, wheelbase L):',
      '  p_closest = findClosestWaypoint(pose, path)',
      '  e_cte = distance(pose, p_closest)',
      '  p_target = findLookaheadIntersection(path, p_closest, Ld)',
      '  α = normalizeAngle(atan2(p_target.y - y, p_target.x - x) - θ)',
      '  δ = clamp(atan2(2 * L * sin(α), Ld), -δ_max, δ_max)',
      '  integrateAckermannBicycle(x, y, θ, speed v, δ, dt)',
    ],
  },

  controls: [
    {
      id: 'lookaheadDistance',
      label: 'Lookahead Distance (Ld)',
      type: 'slider',
      min: 18,
      max: 130,
      step: 4,
      defaultValue: 52,
      unit: 'px',
      description: 'Small Ld causes oscillatory fishtailing; large Ld cuts corners.',
    },
    {
      id: 'vehicleSpeed',
      label: 'Vehicle Speed (v)',
      type: 'slider',
      min: 30,
      max: 130,
      step: 5,
      defaultValue: 75,
      unit: 'px/s',
      description: 'Forward velocity of the Ackermann vehicle.',
    },
    {
      id: 'wheelbase',
      label: 'Wheelbase Length (L)',
      type: 'slider',
      min: 20,
      max: 50,
      step: 2,
      defaultValue: 32,
      unit: 'px',
      description: 'Distance between rear and front axles.',
    },
    {
      id: 'pathType',
      label: 'Reference Track Topology',
      type: 'select',
      options: [
        { label: 'Figure-8 Lissajous', value: 'figure8' },
        { label: 'Elliptical Speedway', value: 'oval' },
        { label: 'Twisted S-Chicane Circuit', value: 'scurve' },
      ],
      defaultValue: 'figure8',
    },
  ],

  createInitialState(params: PurePursuitParams): PurePursuitState {
    const path = buildReferencePath(params.pathType);
    const x = path[0].x;
    const y = path[0].y + 32;
    const theta = Math.atan2(path[1].y - path[0].y, path[1].x - path[0].x);
    const track = findTrackingPoints(x, y, path, params.lookaheadDistance);

    return {
      x,
      y,
      theta,
      delta: 0,
      alpha: 0,
      path,
      lookaheadPoint: track.lookaheadPoint,
      closestPoint: track.closestPoint,
      crossTrackError: track.crossTrackError,
      curvature: 0,
      trail: [{ x, y }],
      iteration: 0,
    };
  },

  step(currentState: PurePursuitState, params: PurePursuitParams) {
    const path = buildReferencePath(params.pathType);
    const { closestPoint, lookaheadPoint, crossTrackError } = findTrackingPoints(
      currentState.x,
      currentState.y,
      path,
      params.lookaheadDistance
    );

    const targetAngle = Math.atan2(
      lookaheadPoint.y - currentState.y,
      lookaheadPoint.x - currentState.x
    );
    let alpha = targetAngle - currentState.theta;
    while (alpha > Math.PI) alpha -= 2 * Math.PI;
    while (alpha < -Math.PI) alpha += 2 * Math.PI;

    const Ld = Math.max(
      10,
      Math.hypot(lookaheadPoint.x - currentState.x, lookaheadPoint.y - currentState.y)
    );
    const curvature = (2 * Math.sin(alpha)) / Ld;
    const maxSteer = 0.72;
    const delta = Math.max(-maxSteer, Math.min(maxSteer, Math.atan(curvature * params.wheelbase)));

    const dt = 0.05;
    const x = currentState.x + params.vehicleSpeed * Math.cos(currentState.theta) * dt;
    const y = currentState.y + params.vehicleSpeed * Math.sin(currentState.theta) * dt;
    let theta =
      currentState.theta +
      (params.vehicleSpeed / params.wheelbase) * Math.tan(delta) * dt;
    while (theta > Math.PI) theta -= 2 * Math.PI;
    while (theta < -Math.PI) theta += 2 * Math.PI;

    const trail = [...currentState.trail, { x, y }].slice(-260);
    const activeLine = crossTrackError > 20 ? 2 : Math.abs(delta) > 0.35 ? 5 : 6;

    return {
      nextState: {
        x,
        y,
        theta,
        delta,
        alpha,
        path,
        lookaheadPoint,
        closestPoint,
        crossTrackError,
        curvature,
        trail,
        iteration: currentState.iteration + 1,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: PurePursuitState, params: PurePursuitParams): MetricReadout[] {
    return [
      { id: 'cte', label: 'Cross-Track Error', value: state.crossTrackError.toFixed(1), unit: 'px', color: state.crossTrackError > 15 ? 'rose' : 'emerald' },
      { id: 'steer', label: 'Steer Angle δ', value: ((state.delta * 180) / Math.PI).toFixed(1), unit: '°', color: 'sky' },
      { id: 'alpha', label: 'Heading Error α', value: ((state.alpha * 180) / Math.PI).toFixed(1), unit: '°', color: 'amber' },
      { id: 'ld', label: 'Lookahead Radius Ld', value: params.lookaheadDistance, unit: 'px', color: 'amber' },
    ];
  },

  onPointerInteraction(
    state: PurePursuitState,
    event: CanvasPointerEvent,
    params: PurePursuitParams
  ): PurePursuitState {
    const x = Math.max(-270, Math.min(270, event.worldX));
    const y = Math.max(-165, Math.min(165, event.worldY));
    const track = findTrackingPoints(x, y, state.path, params.lookaheadDistance);

    return {
      ...state,
      x,
      y,
      closestPoint: track.closestPoint,
      lookaheadPoint: track.lookaheadPoint,
      crossTrackError: track.crossTrackError,
      trail: [{ x, y }],
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: PurePursuitState,
    params: PurePursuitParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Reference Path Spline (Emerald)
    if (state.path.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.path[0].x, state.path[0].y);
      for (let i = 1; i < state.path.length; i++) {
        ctx.lineTo(state.path[i].x, state.path[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.55)';
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // 2. Driven Vehicle Trail (Sky Blue)
    if (state.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.trail[0].x, state.trail[0].y);
      for (let i = 1; i < state.trail.length; i++) {
        ctx.lineTo(state.trail[i].x, state.trail[i].y);
      }
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 3. Cross-Track Error Vector e_cte (Rose)
    ctx.beginPath();
    ctx.moveTo(state.x, state.y);
    ctx.lineTo(state.closestPoint.x, state.closestPoint.y);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 4. Lookahead Circle (Amber Dashed)
    ctx.beginPath();
    ctx.arc(state.x, state.y, params.lookaheadDistance, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Constant-Curvature Circular Pursuit Arc (Amber Solid) & Chord
    ctx.beginPath();
    ctx.moveTo(state.x, state.y);
    ctx.lineTo(state.lookaheadPoint.x, state.lookaheadPoint.y);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const chordLen = Math.hypot(
      state.lookaheadPoint.x - state.x,
      state.lookaheadPoint.y - state.y
    );
    ctx.beginPath();
    ctx.moveTo(state.x, state.y);
    let arcX = state.x;
    let arcY = state.y;
    let arcTheta = state.theta;
    const arcSteps = 16;
    const ds = chordLen / arcSteps;
    for (let s = 0; s < arcSteps; s++) {
      arcX += ds * Math.cos(arcTheta);
      arcY += ds * Math.sin(arcTheta);
      arcTheta += ds * state.curvature;
      ctx.lineTo(arcX, arcY);
    }
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Lookahead "Carrot" Target Dot (Amber)
    ctx.beginPath();
    ctx.arc(state.lookaheadPoint.x, state.lookaheadPoint.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.fill();
    ctx.strokeStyle = '#fef3c7';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 6. Ackermann Vehicle Chassis & Steered Front Wheels (Sky Blue)
    ctx.save();
    ctx.translate(state.x, state.y);
    ctx.rotate(state.theta);

    const L = params.wheelbase;
    ctx.fillStyle = 'rgba(14, 165, 233, 0.28)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-8, -11, L + 16, 22, 5);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(-5, -14, 10, 4);
    ctx.fillRect(-5, 10, 10, 4);

    for (const wy of [-12, 12]) {
      ctx.save();
      ctx.translate(L, wy);
      ctx.rotate(state.delta);
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(-6, -2, 12, 4);
      ctx.restore();
    }

    ctx.restore();
    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click/Drag anywhere to displace the vehicle off-track and test Cross-Track recovery',
      16,
      24
    );
    ctx.restore();
  },
};

export default PurePursuitPlugin;
