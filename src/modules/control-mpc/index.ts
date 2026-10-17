// FILE: src/modules/control-mpc/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface MPCObstacle {
  x: number;
  y: number;
  r: number;
}

export interface MPCHorizonTrajectory {
  points: { x: number; y: number; theta: number }[];
  controls: number[];
  cost: number;
  trackingCost: number;
  controlCost: number;
  obstacleCost: number;
  collides: boolean;
}

export interface MPCState {
  vehicle: {
    x: number;
    y: number;
    theta: number;
    delta: number;
    v: number;
  };
  referencePath: { x: number; y: number; theta: number }[];
  obstacles: MPCObstacle[];
  candidateHorizons: MPCHorizonTrajectory[];
  bestHorizon: MPCHorizonTrajectory | null;
  crossTrackError: number;
  minObstacleDist: number;
  trail: { x: number; y: number }[];
  draggedObstacleIndex: number | null;
  iteration: number;
}

export interface MPCParams {
  horizonSteps: number;
  trackingWeight: number;
  controlWeight: number;
  obstacleWeight: number;
}

const WHEELBASE = 28;
const VEHICLE_SPEED = 64;
const VEHICLE_COLLISION_RADIUS = 12;
const MAX_STEER = 0.65; // ~37 degrees
const DT = 0.08;

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function buildReferencePath(): { x: number; y: number; theta: number }[] {
  const pts: { x: number; y: number; theta: number }[] = [];
  const N = 200;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 2;
    // Chicane racetrack closed loop
    const x = 220 * Math.cos(t);
    const y = 110 * Math.sin(t) + 38 * Math.sin(2 * t);
    
    // Numerical tangent vector for track heading
    const dtSample = 0.001;
    const tNext = t + dtSample;
    const xNext = 220 * Math.cos(tNext);
    const yNext = 110 * Math.sin(tNext) + 38 * Math.sin(2 * tNext);
    const theta = Math.atan2(yNext - y, xNext - x);

    pts.push({ x, y, theta });
  }
  return pts;
}

function findClosestReference(
  x: number,
  y: number,
  path: { x: number; y: number; theta: number }[]
): { dist: number; refX: number; refY: number; refTheta: number; index: number } {
  let bestDistSq = Infinity;
  let bestIdx = 0;
  for (let i = 0; i < path.length; i++) {
    const dx = path[i].x - x;
    const dy = path[i].y - y;
    const dSq = dx * dx + dy * dy;
    if (dSq < bestDistSq) {
      bestDistSq = dSq;
      bestIdx = i;
    }
  }
  const pt = path[bestIdx];
  return {
    dist: Math.sqrt(bestDistSq),
    refX: pt.x,
    refY: pt.y,
    refTheta: pt.theta,
    index: bestIdx,
  };
}

function evaluateMPCHorizon(
  vehicle: { x: number; y: number; theta: number; delta: number; v: number },
  refPath: { x: number; y: number; theta: number }[],
  obstacles: MPCObstacle[],
  params: MPCParams
): { candidateHorizons: MPCHorizonTrajectory[]; bestHorizon: MPCHorizonTrajectory } {
  const horizonN = Math.max(6, Math.min(24, Math.round(params.horizonSteps)));
  const Q = params.trackingWeight;
  const R = params.controlWeight;
  const W_obs = params.obstacleWeight;

  const candidateHorizons: MPCHorizonTrajectory[] = [];
  let bestScore = Infinity;
  let bestHorizon: MPCHorizonTrajectory | null = null;

  // Generate a rich set of multi-stage receding-horizon steering profiles
  const steer0Values = [-0.60, -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45, 0.60];
  const steerMidValues = [-0.55, -0.25, 0.0, 0.25, 0.55];

  const controlSequences: number[][] = [];

  // 1. Parameterized steering spline sequences (delta_0 -> delta_mid)
  for (const d0 of steer0Values) {
    for (const dMid of steerMidValues) {
      const seq: number[] = new Array(horizonN);
      for (let k = 0; k < horizonN; k++) {
        const tau = k / (horizonN - 1);
        const steer = d0 + (dMid - d0) * Math.sin((tau * Math.PI) / 2);
        seq[k] = Math.max(-MAX_STEER, Math.min(MAX_STEER, steer));
      }
      controlSequences.push(seq);
    }
  }

  // 2. Pure constant arc profiles
  for (const cSteer of [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]) {
    const seq = new Array(horizonN).fill(cSteer);
    controlSequences.push(seq);
  }

  // Forward simulate and score each candidate horizon
  for (const controls of controlSequences) {
    const points: { x: number; y: number; theta: number }[] = [{
      x: vehicle.x,
      y: vehicle.y,
      theta: vehicle.theta,
    }];

    let simX = vehicle.x;
    let simY = vehicle.y;
    let simTheta = vehicle.theta;
    let collides = false;

    let trackingCost = 0;
    let controlCost = 0;
    let obstacleCost = 0;

    for (let k = 0; k < horizonN; k++) {
      const delta_k = controls[k];
      
      // Integrate Ackermann bicycle kinematics
      simX += VEHICLE_SPEED * Math.cos(simTheta) * DT;
      simY += VEHICLE_SPEED * Math.sin(simTheta) * DT;
      simTheta = normalizeAngle(simTheta + (VEHICLE_SPEED / WHEELBASE) * Math.tan(delta_k) * DT);
      points.push({ x: simX, y: simY, theta: simTheta });

      // State cost: tracking deviation from reference track
      const ref = findClosestReference(simX, simY, refPath);
      const headingDiff = Math.abs(normalizeAngle(simTheta - ref.refTheta));
      trackingCost += Q * (ref.dist * ref.dist + 50 * headingDiff * headingDiff);

      // Control effort cost: penalize aggressive steering and rapid steering rate
      const steerNormalized = delta_k / MAX_STEER;
      const prevSteer = k === 0 ? vehicle.delta : controls[k - 1];
      const steerRate = (delta_k - prevSteer) / DT;
      controlCost += R * (steerNormalized * steerNormalized * 30 + 0.08 * steerRate * steerRate);

      // Proactive obstacle barrier cost
      for (const obs of obstacles) {
        const dCenter = Math.hypot(simX - obs.x, simY - obs.y);
        const clearance = dCenter - obs.r - VEHICLE_COLLISION_RADIUS;
        if (clearance <= 0) {
          collides = true;
          obstacleCost += 50000 + 3000 * Math.abs(clearance);
        } else if (clearance < 45) {
          // Smooth repulsive barrier function
          const barrierRatio = (45 - clearance) / (clearance + 3);
          obstacleCost += W_obs * barrierRatio * barrierRatio * 45;
        }
      }
    }

    const totalCost = trackingCost + controlCost + obstacleCost;
    const trajectory: MPCHorizonTrajectory = {
      points,
      controls,
      cost: totalCost,
      trackingCost,
      controlCost,
      obstacleCost,
      collides,
    };

    candidateHorizons.push(trajectory);

    if (totalCost < bestScore) {
      bestScore = totalCost;
      bestHorizon = trajectory;
    }
  }

  // Fallback to first candidate if none found
  if (!bestHorizon && candidateHorizons.length > 0) {
    bestHorizon = candidateHorizons[0];
  }

  return { candidateHorizons, bestHorizon: bestHorizon! };
}

export const MPCControlPlugin: AlgorithmPlugin<MPCState, MPCParams> = {
  id: 'control-mpc',
  title: 'Model Predictive Control (MPC) Horizon Planner',
  shortDescription:
    'Receding-horizon trajectory optimization over N future steps balancing reference tracking, steering effort, and proactive obstacle avoidance.',
  category: 'kinematics-control',

  controls: [
    {
      id: 'horizonSteps',
      label: 'Prediction Horizon (N)',
      type: 'slider',
      min: 6,
      max: 24,
      step: 2,
      defaultValue: 14,
      description: 'Prediction horizon N steps into the future.',
    },
    {
      id: 'trackingWeight',
      label: 'Tracking Cost Weight (Q)',
      type: 'slider',
      min: 0.5,
      max: 4.0,
      step: 0.25,
      defaultValue: 2.0,
      description: 'State cost weight Q penalizing deviation from reference track.',
    },
    {
      id: 'controlWeight',
      label: 'Control Effort Weight (R)',
      type: 'slider',
      min: 0.1,
      max: 3.0,
      step: 0.1,
      defaultValue: 0.8,
      description: 'Control effort weight R penalizing aggressive steering angles.',
    },
    {
      id: 'obstacleWeight',
      label: 'Obstacle Barrier Weight',
      type: 'slider',
      min: 0.5,
      max: 5.0,
      step: 0.5,
      defaultValue: 2.5,
      description: 'Repulsive cost weight penalizing proximity to track obstacles.',
    },
  ],

  documentation: {
    intuition:
      'Unlike purely reactive controllers that only react to instantaneous cross-track error, Model Predictive Control (MPC) computes a lookahead horizon N steps into the future. By forward-simulating Ackermann bicycle kinematics over candidate steering profiles, it anticipates upcoming sharp corners and proactively swerves around obstacles with smooth, effort-penalized steering before executing only the very first optimal control action u_0* in true receding-horizon fashion.',
    governingEquationLatex:
      'J(\\textcolor{#38bdf8}{\\mathbf{u}_{0:N-1}}) = \\sum_{k=0}^{N-1} \\Big( \\|\\textcolor{#38bdf8}{\\mathbf{x}_k} - \\textcolor{#10b981}{\\mathbf{x}_{\\text{ref},k}}\\|_Q^2 + \\|\\textcolor{#f59e0b}{\\mathbf{u}_k}\\|_R^2 + \\textcolor{#f43f5e}{c_{\\text{obs}}(\\mathbf{x}_k)} \\Big)',
    pseudocode: [
      'function MPC_Step(x_0, refPath, N, obstacles, Q, R, W_obs):',
      '  for each candidate sequence u_{0:N-1} in U_admissible:',
      '    x_{0:N} = rolloutBicycleDynamics(x_0, u_{0:N-1}, dt)',
      '    J = sum_{k=0}^{N-1} [ Q·dist(x_k, ref)^2 + R·u_k^2 + W_obs·barrier(x_k, obs) ]',
      '  u*_{0:N-1} = argmin_{u} J(u_{0:N-1})   // Optimal horizon sequence',
      '  applyControl(u*_0)                       // Receding horizon: execute first action',
      '  shiftHorizonAndReplan()                  // Re-solve at next control cycle',
    ],
  },

  createInitialState(params: MPCParams): MPCState {
    const referencePath = buildReferencePath();
    const startPt = referencePath[0];
    const vehicle = {
      x: startPt.x,
      y: startPt.y,
      theta: startPt.theta,
      delta: 0,
      v: VEHICLE_SPEED,
    };

    // 2 Rose obstacles placed directly near the reference curve
    const obstacles: MPCObstacle[] = [
      { x: 105, y: 76, r: 24 },
      { x: -105, y: -64, r: 24 },
    ];

    const { candidateHorizons, bestHorizon } = evaluateMPCHorizon(
      vehicle,
      referencePath,
      obstacles,
      params
    );

    const refInfo = findClosestReference(vehicle.x, vehicle.y, referencePath);
    let minObs = Infinity;
    for (const obs of obstacles) {
      const d = Math.hypot(vehicle.x - obs.x, vehicle.y - obs.y) - obs.r - VEHICLE_COLLISION_RADIUS;
      if (d < minObs) minObs = d;
    }

    return {
      vehicle,
      referencePath,
      obstacles,
      candidateHorizons,
      bestHorizon,
      crossTrackError: refInfo.dist,
      minObstacleDist: minObs,
      trail: [{ x: vehicle.x, y: vehicle.y }],
      draggedObstacleIndex: null,
      iteration: 0,
    };
  },

  step(currentState: MPCState, params: MPCParams) {
    const vehicle = { ...currentState.vehicle };
    const referencePath = currentState.referencePath;
    const obstacles = currentState.obstacles.map((o) => ({ ...o }));

    // Evaluate candidate horizons using current vehicle state
    const { candidateHorizons, bestHorizon } = evaluateMPCHorizon(
      vehicle,
      referencePath,
      obstacles,
      params
    );

    // Apply the first control action u_0* from the optimal horizon
    const u0 = bestHorizon ? bestHorizon.controls[0] : 0;
    vehicle.delta = u0;
    vehicle.x += VEHICLE_SPEED * Math.cos(vehicle.theta) * DT;
    vehicle.y += VEHICLE_SPEED * Math.sin(vehicle.theta) * DT;
    vehicle.theta = normalizeAngle(vehicle.theta + (VEHICLE_SPEED / WHEELBASE) * Math.tan(u0) * DT);

    // Compute metrics
    const refInfo = findClosestReference(vehicle.x, vehicle.y, referencePath);
    let minObs = Infinity;
    for (const obs of obstacles) {
      const d = Math.hypot(vehicle.x - obs.x, vehicle.y - obs.y) - obs.r - VEHICLE_COLLISION_RADIUS;
      if (d < minObs) minObs = d;
    }

    const trail = [...currentState.trail, { x: vehicle.x, y: vehicle.y }].slice(-240);
    const nextIter = currentState.iteration + 1;
    const activePseudocodeLine = minObs < 30 ? 4 : Math.abs(vehicle.delta) > 0.3 ? 5 : 6;

    return {
      nextState: {
        vehicle,
        referencePath,
        obstacles,
        candidateHorizons,
        bestHorizon,
        crossTrackError: refInfo.dist,
        minObstacleDist: minObs,
        trail,
        draggedObstacleIndex: currentState.draggedObstacleIndex,
        iteration: nextIter,
      },
      activePseudocodeLine,
      isComplete: false,
    };
  },

  getMetrics(state: MPCState): MetricReadout[] {
    const steerDeg = ((state.vehicle.delta * 180) / Math.PI).toFixed(1);
    const costVal = state.bestHorizon ? Math.round(state.bestHorizon.cost).toLocaleString() : '0';
    return [
      {
        id: 'cte',
        label: 'Track Deviation',
        value: state.crossTrackError.toFixed(1),
        unit: 'px',
        color: state.crossTrackError > 18 ? 'amber' : 'emerald',
      },
      {
        id: 'steer',
        label: 'Optimal Steer u₀*',
        value: steerDeg,
        unit: '°',
        color: 'sky',
      },
      {
        id: 'obsDist',
        label: 'Obs Clearance',
        value: Math.max(0, state.minObstacleDist).toFixed(1),
        unit: 'px',
        color: state.minObstacleDist < 15 ? 'rose' : state.minObstacleDist < 40 ? 'amber' : 'emerald',
      },
      {
        id: 'cost',
        label: 'Horizon Cost J*',
        value: costVal,
        color: 'amber',
      },
    ];
  },

  onPointerInteraction(
    state: MPCState,
    event: CanvasPointerEvent,
    params: MPCParams
  ): MPCState {
    const nextObstacles = state.obstacles.map((o) => ({ ...o }));
    let draggedIdx = state.draggedObstacleIndex;

    const clampedX = Math.max(-240, Math.min(240, event.worldX));
    const clampedY = Math.max(-150, Math.min(150, event.worldY));

    if (!event.isDragging) {
      // Find nearest obstacle to click
      let bestDist = Infinity;
      let hitIdx = 0;
      for (let i = 0; i < nextObstacles.length; i++) {
        const d = Math.hypot(event.worldX - nextObstacles[i].x, event.worldY - nextObstacles[i].y);
        if (d < bestDist) {
          bestDist = d;
          hitIdx = i;
        }
      }
      draggedIdx = hitIdx;
    }

    if (draggedIdx !== null && nextObstacles[draggedIdx]) {
      nextObstacles[draggedIdx] = {
        ...nextObstacles[draggedIdx],
        x: clampedX,
        y: clampedY,
      };
    }

    // Immediately re-plan horizon with updated obstacle positions
    const { candidateHorizons, bestHorizon } = evaluateMPCHorizon(
      state.vehicle,
      state.referencePath,
      nextObstacles,
      params
    );

    let minObs = Infinity;
    for (const obs of nextObstacles) {
      const d = Math.hypot(state.vehicle.x - obs.x, state.vehicle.y - obs.y) - obs.r - VEHICLE_COLLISION_RADIUS;
      if (d < minObs) minObs = d;
    }

    return {
      ...state,
      obstacles: nextObstacles,
      candidateHorizons,
      bestHorizon,
      minObstacleDist: minObs,
      draggedObstacleIndex: draggedIdx,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: MPCState,
    _params: MPCParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Reference Track Spline (Emerald #10b981)
    if (state.referencePath.length > 1) {
      // Road boundary width outline
      ctx.beginPath();
      ctx.moveTo(state.referencePath[0].x, state.referencePath[0].y);
      for (let i = 1; i < state.referencePath.length; i++) {
        ctx.lineTo(state.referencePath[i].x, state.referencePath[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.55)';
      ctx.lineWidth = 3.5;
      ctx.stroke();

      // Subtle dashed inner center guide
      ctx.beginPath();
      ctx.moveTo(state.referencePath[0].x, state.referencePath[0].y);
      for (let i = 1; i < state.referencePath.length; i++) {
        ctx.lineTo(state.referencePath[i].x, state.referencePath[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.25)';
      ctx.setLineDash([4, 6]);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. Circular Obstacles (Rose #f43f5e) + Repulsive Proximity Barrier Halo
    for (let i = 0; i < state.obstacles.length; i++) {
      const obs = state.obstacles[i];

      // Proactive barrier threshold ring
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r + 35, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.2)';
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.setLineDash([]);

      // Solid obstacle body
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Drag affordance crosshair / ring
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#f43f5e';
      ctx.fill();

      ctx.save();
      ctx.font = '600 10px monospace';
      ctx.fillStyle = '#fecdd3';
      ctx.textAlign = 'center';
      ctx.fillText(`Obs ${i + 1}`, obs.x, obs.y - obs.r - 6);
      ctx.restore();
    }

    // 3. Driven Vehicle Trail (Sky Blue)
    if (state.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.trail[0].x, state.trail[0].y);
      for (let i = 1; i < state.trail.length; i++) {
        ctx.lineTo(state.trail[i].x, state.trail[i].y);
      }
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 4. Candidate Horizon Rollout Fan (Amber #f59e0b)
    for (const traj of state.candidateHorizons) {
      if (traj === state.bestHorizon || traj.points.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(traj.points[0].x, traj.points[0].y);
      for (let j = 1; j < traj.points.length; j++) {
        ctx.lineTo(traj.points[j].x, traj.points[j].y);
      }

      if (traj.collides) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.18)';
        ctx.lineWidth = 1;
      } else {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.22)';
        ctx.lineWidth = 1.2;
      }
      ctx.stroke();
    }

    // 5. Optimal Horizon Rollout x*_{0:N} (Glowing Emerald #10b981)
    if (state.bestHorizon && state.bestHorizon.points.length > 1) {
      const best = state.bestHorizon;

      ctx.save();
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(best.points[0].x, best.points[0].y);
      for (let j = 1; j < best.points.length; j++) {
        ctx.lineTo(best.points[j].x, best.points[j].y);
      }
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.restore();

      // Rollout discrete time steps dots along the winning horizon
      for (let j = 1; j < best.points.length; j++) {
        ctx.beginPath();
        ctx.arc(best.points[j].x, best.points[j].y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#6ee7b7';
        ctx.fill();
      }
    }

    // 6. Sky Blue Vehicle Chassis with Steered Front Wheels
    ctx.save();
    ctx.translate(state.vehicle.x, state.vehicle.y);
    ctx.rotate(state.vehicle.theta);

    const L = WHEELBASE;
    // Main body
    ctx.fillStyle = 'rgba(14, 165, 233, 0.32)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-8, -11, L + 16, 22, 5);
    ctx.fill();
    ctx.stroke();

    // Rear axle wheels
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(-5, -14, 10, 4);
    ctx.fillRect(-5, 10, 10, 4);

    // Steered front wheels (Amber)
    for (const wy of [-12, 12]) {
      ctx.save();
      ctx.translate(L, wy);
      ctx.rotate(state.vehicle.delta);
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(-6, -2, 12, 4);
      ctx.restore();
    }

    // Heading direction indicator
    ctx.beginPath();
    ctx.moveTo(L + 6, 0);
    ctx.lineTo(L + 14, 0);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();
    ctx.restore();

    // Canvas HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click / Drag Rose Obstacles to test proactive MPC horizon obstacle avoidance',
      16,
      24
    );
    ctx.restore();
  },
};

export default MPCControlPlugin;
