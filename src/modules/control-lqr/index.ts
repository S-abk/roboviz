// FILE: src/modules/control-lqr/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface SpacecraftAgent {
  x: number;
  y: number;
  vx: number;
  vy: number;
  uX: number;
  uY: number;
  trail: { x: number; y: number }[];
  color: string;
  name: string;
}

export interface LQRState {
  primary: SpacecraftAgent;
  ghosts: SpacecraftAgent[];
  goal: { x: number; y: number };
  draggedTarget: 'none' | 'goal' | 'primary';
  P: number[][]; // 4x4 discrete cost-to-go matrix
  K: number[][]; // 2x4 optimal state feedback gain matrix
  P_pos: {
    a: number;
    b: number;
    d: number;
    lambda1: number;
    lambda2: number;
    angle: number;
  };
  kPosNorm: number;
  kVelNorm: number;
  instantCost: number;
  costToGo: number;
  iteration: number;
}

export interface LQRParams {
  qPosition: number;
  qVelocity: number;
  rEffort: number;
  crossCoupling: number;
}

const DT = 0.05;
const MAX_THRUST = 360;

// Analytical 2x2 matrix inversion
function inv22(M: number[][]): number[][] {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const safeDet = Math.abs(det) < 1e-9 ? 1e-9 : det;
  return [
    [M[1][1] / safeDet, -M[0][1] / safeDet],
    [-M[1][0] / safeDet, M[0][0] / safeDet],
  ];
}

// Matrix multiplication helper
function mulMat(A: number[][], B: number[][]): number[][] {
  const rA = A.length;
  const cA = A[0].length;
  const cB = B[0].length;
  const out = Array.from({ length: rA }, () => new Array(cB).fill(0));
  for (let i = 0; i < rA; i++) {
    for (let k = 0; k < cA; k++) {
      const aik = A[i][k];
      for (let j = 0; j < cB; j++) {
        out[i][j] += aik * B[k][j];
      }
    }
  }
  return out;
}

function transposeMat(A: number[][]): number[][] {
  const r = A.length;
  const c = A[0].length;
  const out = Array.from({ length: c }, () => new Array(r).fill(0));
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[j][i] = A[i][j];
    }
  }
  return out;
}

// Solves the Discrete Algebraic Riccati Equation (DARE)
// P = Q + A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A
// K = (R + B^T P B)^(-1) B^T P A
function solveDARE(
  A: number[][],
  B: number[][],
  Q: number[][],
  R: number[][],
  maxIter = 75,
  eps = 1e-6
): { P: number[][]; K: number[][] } {
  let P = Q.map((row) => [...row]);
  const AT = transposeMat(A);
  const BT = transposeMat(B);
  let K: number[][] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];

  for (let iter = 0; iter < maxIter; iter++) {
    const BTP = mulMat(BT, P);
    const BTPB = mulMat(BTP, B);
    const S = [
      [BTPB[0][0] + R[0][0], BTPB[0][1] + R[0][1]],
      [BTPB[1][0] + R[1][0], BTPB[1][1] + R[1][1]],
    ];
    const Sinv = inv22(S);
    const BTPA = mulMat(BTP, A);
    K = mulMat(Sinv, BTPA);

    const ATP = mulMat(AT, P);
    const ATPA = mulMat(ATP, A);
    const ATPB = mulMat(ATP, B);
    const ATPBK = mulMat(ATPB, K);

    const Pnext = Array.from({ length: 4 }, () => new Array(4).fill(0));
    let maxDiff = 0;

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        Pnext[i][j] = ATPA[i][j] - ATPBK[i][j] + Q[i][j];
        maxDiff = Math.max(maxDiff, Math.abs(Pnext[i][j] - P[i][j]));
      }
    }

    // Preserve symmetry
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        const avg = 0.5 * (Pnext[i][j] + Pnext[j][i]);
        Pnext[i][j] = avg;
        Pnext[j][i] = avg;
      }
    }

    P = Pnext;
    if (maxDiff < eps) break;
  }

  return { P, K };
}

function computeLQRMatrices(params: LQRParams): {
  P: number[][];
  K: number[][];
  P_pos: {
    a: number;
    b: number;
    d: number;
    lambda1: number;
    lambda2: number;
    angle: number;
  };
  kPosNorm: number;
  kVelNorm: number;
} {
  const omega = params.crossCoupling;
  const qP = params.qPosition;
  const qV = params.qVelocity;
  const rE = params.rEffort;

  // Discrete system dynamics with Coriolis / swirl cross-coupling
  const A = [
    [1, omega * DT * 0.7, DT, 0],
    [-omega * DT * 0.7, 1, 0, DT],
    [0, 0, 1, 2 * omega * DT],
    [0, 0, -2 * omega * DT, 1],
  ];

  const B = [
    [0.5 * DT * DT, 0],
    [0, 0.5 * DT * DT],
    [DT, 0],
    [0, DT],
  ];

  // Anisotropic state penalty in orbital frame:
  const Q = [
    [qP * 1.5, 0, 0, 0],
    [0, qP * 0.85, 0, 0],
    [0, 0, qV * 1.25, 0],
    [0, 0, 0, qV * 0.8],
  ];

  const R = [
    [rE, 0],
    [0, rE],
  ];

  const { P, K } = solveDARE(A, B, Q, R);

  // 2D position sub-block of cost-to-go matrix P
  const a = P[0][0];
  const b = P[0][1];
  const d = P[1][1];

  const trace = a + d;
  const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const lambda1 = Math.max(0.1, (trace + disc) / 2);
  const lambda2 = Math.max(0.1, (trace - disc) / 2);
  const angle = 0.5 * Math.atan2(2 * b, a - d);

  const kPosNorm = Math.hypot(K[0][0], K[0][1], K[1][0], K[1][1]);
  const kVelNorm = Math.hypot(K[0][2], K[0][3], K[1][2], K[1][3]);

  return {
    P,
    K,
    P_pos: { a, b, d, lambda1, lambda2, angle },
    kPosNorm,
    kVelNorm,
  };
}

export const LQRControlPlugin: AlgorithmPlugin<LQRState, LQRParams> = {
  id: 'control-lqr',
  title: 'Linear Quadratic Regulator (LQR) & Riccati Control',
  shortDescription:
    'Infinite-horizon optimal state feedback u = -Kx solved via the Discrete Algebraic Riccati Equation (DARE) with quadratic cost-to-go contours.',
  category: 'kinematics-control',

  controls: [
    {
      id: 'qPosition',
      label: 'Position Penalty (Q_pos)',
      type: 'slider',
      min: 0.2,
      max: 5.0,
      step: 0.2,
      defaultValue: 1.5,
      description: 'State position error penalty Q_pos in quadratic cost.',
    },
    {
      id: 'qVelocity',
      label: 'Velocity Damping (Q_vel)',
      type: 'slider',
      min: 0.1,
      max: 4.0,
      step: 0.1,
      defaultValue: 0.8,
      description: 'State velocity damping penalty Q_vel in quadratic cost.',
    },
    {
      id: 'rEffort',
      label: 'Control Effort Penalty (R)',
      type: 'slider',
      min: 0.1,
      max: 5.0,
      step: 0.1,
      defaultValue: 1.0,
      description: 'Control actuation effort penalty R (higher R = gentler thruster force).',
    },
    {
      id: 'crossCoupling',
      label: 'Coriolis / Swirl Coupling (ω)',
      type: 'slider',
      min: 0.0,
      max: 0.8,
      step: 0.1,
      defaultValue: 0.25,
      description: 'Coriolis/swirl cross-coupling in system dynamics A.',
    },
  ],

  documentation: {
    intuition:
      'The Linear Quadratic Regulator (LQR) solves for the mathematically optimal trade-off between tracking precision and actuator effort over an infinite future horizon. By solving the Discrete Algebraic Riccati Equation (DARE), it derives a steady-state constant feedback matrix K. The quadratic cost-to-go level sets P warp into tilted ellipses as Coriolis cross-coupling introduces gyroscopic swirling, guiding the spacecraft along energy-minimal approach corridors.',
    governingEquationLatex:
      '\\textcolor{#f59e0b}{\\mathbf{u}_t^*} = -\\mathbf{K}(\\textcolor{#38bdf8}{\\mathbf{x}_t} - \\textcolor{#10b981}{\\mathbf{x}_{\\text{goal}}}), \\quad \\mathbf{P} = \\mathbf{Q} + \\mathbf{A}^T\\mathbf{P}\\mathbf{A} - \\mathbf{A}^T\\mathbf{P}\\mathbf{B}(\\textcolor{#f43f5e}{\\mathbf{R}} + \\mathbf{B}^T\\mathbf{P}\\mathbf{B})^{-1}\\mathbf{B}^T\\mathbf{P}\\mathbf{A}',
    pseudocode: [
      'function LQR_DARE_Solve(A, B, Q, R):',
      '  P = Q // Initialize cost-to-go matrix',
      '  repeat: P_next = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A',
      '  until ||P_next - P|| < epsilon // Convergence to infinite horizon',
      '  K = (R + B^T P B)^{-1} B^T P A // Optimal feedback gain',
      '  u* = -K (x - x_goal) // Closed-loop optimal control law',
      '  x_{t+1} = A x_t + B u* // Propagate coupled dynamics',
    ],
  },

  createInitialState(params: LQRParams): LQRState {
    const { P, K, P_pos, kPosNorm, kVelNorm } = computeLQRMatrices(params);

    const goal = { x: 0, y: 0 };

    const primary: SpacecraftAgent = {
      x: -185,
      y: -115,
      vx: 24,
      vy: -18,
      uX: 0,
      uY: 0,
      trail: [{ x: -185, y: -115 }],
      color: '#38bdf8', // Sky Blue
      name: 'Primary Spacecraft',
    };

    const ghosts: SpacecraftAgent[] = [
      {
        x: 185,
        y: -115,
        vx: -18,
        vy: -15,
        uX: 0,
        uY: 0,
        trail: [{ x: 185, y: -115 }],
        color: '#a855f7', // Violet
        name: 'Ghost Q1',
      },
      {
        x: 175,
        y: 125,
        vx: -15,
        vy: 18,
        uX: 0,
        uY: 0,
        trail: [{ x: 175, y: 125 }],
        color: '#06b6d4', // Cyan
        name: 'Ghost Q4',
      },
      {
        x: -175,
        y: 125,
        vx: 18,
        vy: 12,
        uX: 0,
        uY: 0,
        trail: [{ x: -175, y: 125 }],
        color: '#f43f5e', // Rose
        name: 'Ghost Q3',
      },
    ];

    const dx = primary.x - goal.x;
    const dy = primary.y - goal.y;
    const e = [dx, dy, primary.vx, primary.vy];
    let instantCost = params.qPosition * (dx * dx + dy * dy) + params.qVelocity * (primary.vx * primary.vx + primary.vy * primary.vy);
    let costToGo = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        costToGo += e[i] * P[i][j] * e[j];
      }
    }

    return {
      primary,
      ghosts,
      goal,
      draggedTarget: 'none',
      P,
      K,
      P_pos,
      kPosNorm,
      kVelNorm,
      instantCost,
      costToGo,
      iteration: 0,
    };
  },

  step(currentState: LQRState, params: LQRParams) {
    const { P, K, P_pos, kPosNorm, kVelNorm } = computeLQRMatrices(params);
    const goal = currentState.goal;
    const omega = params.crossCoupling;

    function simulateAgent(agent: SpacecraftAgent): SpacecraftAgent {
      const ex = agent.x - goal.x;
      const ey = agent.y - goal.y;
      const evx = agent.vx;
      const evy = agent.vy;

      // Optimal state feedback: u* = -K * e
      let ux = -(K[0][0] * ex + K[0][1] * ey + K[0][2] * evx + K[0][3] * evy);
      let uy = -(K[1][0] * ex + K[1][1] * ey + K[1][2] * evx + K[1][3] * evy);

      // Actuator saturation limit
      const uNorm = Math.hypot(ux, uy);
      if (uNorm > MAX_THRUST) {
        ux = (ux / uNorm) * MAX_THRUST;
        uy = (uy / uNorm) * MAX_THRUST;
      }

      // Propagate coupled planar dynamics
      const nextX = agent.x + agent.vx * DT + 0.5 * ux * DT * DT;
      const nextY = agent.y + agent.vy * DT + 0.5 * uy * DT * DT;
      const nextVx = agent.vx + (omega * agent.vy + ux) * DT;
      const nextVy = agent.vy + (-omega * agent.vx + uy) * DT;

      const nextTrail = [...agent.trail, { x: nextX, y: nextY }].slice(-160);

      return {
        ...agent,
        x: nextX,
        y: nextY,
        vx: nextVx,
        vy: nextVy,
        uX: ux,
        uY: uy,
        trail: nextTrail,
      };
    }

    const nextPrimaryRaw = simulateAgent(currentState.primary);
    let nextGhosts = currentState.ghosts.map((g) => simulateAgent(g));

    // Periodic repositioning of spacecraft once docked so user always sees active orbital convergence
    const pDist = Math.hypot(nextPrimaryRaw.x - goal.x, nextPrimaryRaw.y - goal.y);
    const pSpeed = Math.hypot(nextPrimaryRaw.vx, nextPrimaryRaw.vy);

    const nextPrimary: SpacecraftAgent =
      pDist < 6 && pSpeed < 3 && currentState.iteration > 0 && currentState.iteration % 180 === 0
        ? {
            ...nextPrimaryRaw,
            x: -185,
            y: -115,
            vx: 24,
            vy: -18,
            trail: [{ x: -185, y: -115 }],
          }
        : nextPrimaryRaw;

    // If ghost is settled inside target for long, relaunch from initial quadrant
    const ghostOrigins = [
      { x: 185, y: -115, vx: -18, vy: -15 },
      { x: 175, y: 125, vx: -15, vy: 18 },
      { x: -175, y: 125, vx: 18, vy: 12 },
    ];

    nextGhosts = nextGhosts.map((g, idx) => {
      const gDist = Math.hypot(g.x - goal.x, g.y - goal.y);
      const gSpeed = Math.hypot(g.vx, g.vy);
      if (gDist < 6 && gSpeed < 3 && currentState.iteration % 180 === 0) {
        const orig = ghostOrigins[idx];
        return {
          ...g,
          x: orig.x,
          y: orig.y,
          vx: orig.vx,
          vy: orig.vy,
          trail: [{ x: orig.x, y: orig.y }],
        };
      }
      return g;
    });

    // Primary agent state metrics
    const ex = nextPrimary.x - goal.x;
    const ey = nextPrimary.y - goal.y;
    const e = [ex, ey, nextPrimary.vx, nextPrimary.vy];

    const stateCost =
      params.qPosition * (ex * ex + ey * ey) +
      params.qVelocity * (nextPrimary.vx * nextPrimary.vx + nextPrimary.vy * nextPrimary.vy);
    const controlCost = params.rEffort * (nextPrimary.uX * nextPrimary.uX + nextPrimary.uY * nextPrimary.uY) * 0.005;
    const instantCost = stateCost + controlCost;

    let costToGo = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        costToGo += e[i] * P[i][j] * e[j];
      }
    }

    const nextIter = currentState.iteration + 1;
    const activePseudocodeLine = pDist > 40 ? 5 : pSpeed > 10 ? 6 : 4;

    return {
      nextState: {
        primary: nextPrimary,
        ghosts: nextGhosts,
        goal,
        draggedTarget: currentState.draggedTarget,
        P,
        K,
        P_pos,
        kPosNorm,
        kVelNorm,
        instantCost,
        costToGo,
        iteration: nextIter,
      },
      activePseudocodeLine,
      isComplete: false,
    };
  },

  getMetrics(state: LQRState, params: LQRParams): MetricReadout[] {
    const lqr = computeLQRMatrices(params);
    const primaryDist = Math.hypot(state.primary.x - state.goal.x, state.primary.y - state.goal.y);
    const primarySpeed = Math.hypot(state.primary.vx, state.primary.vy);
    const thrustNorm = Math.hypot(state.primary.uX, state.primary.uY);

    return [
      {
        id: 'dist',
        label: 'Docking Offset',
        value: primaryDist.toFixed(1),
        unit: 'px',
        color: primaryDist < 12 ? 'emerald' : primaryDist < 40 ? 'amber' : 'sky',
      },
      {
        id: 'speed',
        label: 'Speed ||v||',
        value: primarySpeed.toFixed(1),
        unit: 'px/s',
        color: primarySpeed < 5 ? 'emerald' : 'sky',
      },
      {
        id: 'thrust',
        label: 'Actuation ||u*||',
        value: Math.round(thrustNorm),
        unit: 'N',
        color: 'amber',
      },
      {
        id: 'kpos',
        label: 'Gain ||K_pos||',
        value: lqr.kPosNorm.toFixed(2),
        color: 'sky',
      },
      {
        id: 'kvel',
        label: 'Gain ||K_vel||',
        value: lqr.kVelNorm.toFixed(2),
        color: 'sky',
      },
      {
        id: 'costToGo',
        label: 'Cost-to-Go J*',
        value: Math.round(state.costToGo).toLocaleString(),
        color: 'rose',
      },
    ];
  },

  onPointerInteraction(state: LQRState, event: CanvasPointerEvent, _params: LQRParams): LQRState {
    const distToGoal = Math.hypot(event.worldX - state.goal.x, event.worldY - state.goal.y);
    const distToShip = Math.hypot(event.worldX - state.primary.x, event.worldY - state.primary.y);

    let nextTarget = state.draggedTarget;
    if (!event.isDragging) {
      if (distToGoal < 32) {
        nextTarget = 'goal';
      } else if (distToShip < 32) {
        nextTarget = 'primary';
      } else {
        nextTarget = 'primary';
      }
    }

    if (nextTarget === 'goal') {
      return {
        ...state,
        goal: { x: event.worldX, y: event.worldY },
        draggedTarget: nextTarget,
      };
    }

    // Teleport primary spacecraft & inject tangential orbital velocity
    const targetX = event.worldX;
    const targetY = event.worldY;
    const dx = targetX - state.goal.x;
    const dy = targetY - state.goal.y;
    const injectedVx = -dy * 0.28;
    const injectedVy = dx * 0.28;

    const updatedPrimary: SpacecraftAgent = {
      ...state.primary,
      x: targetX,
      y: targetY,
      vx: injectedVx,
      vy: injectedVy,
      trail: [{ x: targetX, y: targetY }],
    };

    return {
      ...state,
      primary: updatedPrimary,
      draggedTarget: nextTarget,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: LQRState,
    params: LQRParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    const liveLQR = computeLQRMatrices(params);

    ctx.save();
    ctx.translate(cx, cy);

    const goalX = state.goal.x;
    const goalY = state.goal.y;

    // 1. Background Optimal Feedback Vector Field (Amber / Slate arrows)
    const gridSpacing = 48;
    const xRange = Math.min(280, Math.floor(width / 2 - 20));
    const yRange = Math.min(180, Math.floor(height / 2 - 20));

    ctx.lineWidth = 1;
    for (let gx = -xRange; gx <= xRange; gx += gridSpacing) {
      for (let gy = -yRange; gy <= yRange; gy += gridSpacing) {
        const ex = gx - goalX;
        const ey = gy - goalY;
        const dist = Math.hypot(ex, ey);
        if (dist < 18) continue;

        // Feedback thrust at zero velocity
        const ux = -(liveLQR.K[0][0] * ex + liveLQR.K[0][1] * ey);
        const uy = -(liveLQR.K[1][0] * ex + liveLQR.K[1][1] * ey);
        const uMag = Math.hypot(ux, uy);
        if (uMag < 1e-4) continue;

        const arrowLen = Math.min(16, Math.max(6, Math.sqrt(uMag) * 0.9));
        const dirX = (ux / uMag) * arrowLen;
        const dirY = (uy / uMag) * arrowLen;

        ctx.strokeStyle = 'rgba(245, 158, 11, 0.16)';
        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(gx + dirX, gy + dirY);
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(dirY, dirX);
        const headLen = 3.5;
        ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
        ctx.beginPath();
        ctx.moveTo(gx + dirX, gy + dirY);
        ctx.lineTo(
          gx + dirX - headLen * Math.cos(angle - Math.PI / 6),
          gy + dirY - headLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          gx + dirX - headLen * Math.cos(angle + Math.PI / 6),
          gy + dirY - headLen * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      }
    }

    // 2. Nested Quadratic Cost-to-Go Level-Set Ellipses around Emerald Goal
    // (x - x_goal)^T P_pos (x - x_goal) = c
    const { lambda1, lambda2, angle: tiltAngle } = liveLQR.P_pos;
    const baseLevels = [500, 2200, 6000, 13000, 24000];

    for (let i = 0; i < baseLevels.length; i++) {
      const c = baseLevels[i];
      const rx = Math.sqrt(c / lambda1);
      const ry = Math.sqrt(c / lambda2);

      if (rx < 4 || ry < 4) continue;

      ctx.save();
      ctx.translate(goalX, goalY);
      ctx.rotate(tiltAngle);

      // Ellipse fill
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(16, 185, 129, ${0.035 - i * 0.005})`;
      ctx.fill();

      // Ellipse stroke
      ctx.strokeStyle = `rgba(16, 185, 129, ${0.45 - i * 0.07})`;
      ctx.lineWidth = i === 1 ? 1.8 : 1.0;
      ctx.setLineDash(i % 2 === 0 ? [] : [4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label on the ellipse boundary
      if (i === 2 || i === 4) {
        ctx.font = '600 9px monospace';
        ctx.fillStyle = 'rgba(110, 231, 183, 0.55)';
        ctx.fillText(`J*=${Math.round(c / 100)}`, rx + 3, 0);
      }

      ctx.restore();
    }

    // 3. Ghost Spacecraft Trajectories (Comparison Rollouts)
    for (const ghost of state.ghosts) {
      if (ghost.trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(ghost.trail[0].x, ghost.trail[0].y);
        for (let i = 1; i < ghost.trail.length; i++) {
          ctx.lineTo(ghost.trail[i].x, ghost.trail[i].y);
        }
        ctx.strokeStyle = ghost.color;
        ctx.globalAlpha = 0.38;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }

      // Small ghost marker
      ctx.save();
      ctx.translate(ghost.x, ghost.y);
      ctx.fillStyle = ghost.color;
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      ctx.arc(0, 0, 4, 0, Math.PI * 2);
      ctx.fill();

      // Ghost thruster vector
      const gThrust = Math.hypot(ghost.uX, ghost.uY);
      if (gThrust > 8) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo((-ghost.uX / gThrust) * 12, (-ghost.uY / gThrust) * 12);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 4. Primary Spacecraft Past Trajectory Trail (Sky Blue)
    if (state.primary.trail.length > 1) {
      ctx.save();
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(state.primary.trail[0].x, state.primary.trail[0].y);
      for (let i = 1; i < state.primary.trail.length; i++) {
        ctx.lineTo(state.primary.trail[i].x, state.primary.trail[i].y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.restore();
    }

    // 5. Emerald Docking Target Station (Goal)
    ctx.save();
    ctx.translate(goalX, goalY);

    // Pulsing outer halo ring
    const pulseR = 20 + 3 * Math.sin(state.iteration * 0.08);
    ctx.beginPath();
    ctx.arc(0, 0, pulseR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Solid inner docking ring
    ctx.beginPath();
    ctx.arc(0, 0, 9, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Crosshairs
    ctx.strokeStyle = '#6ee7b7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-14, 0);
    ctx.lineTo(14, 0);
    ctx.moveTo(0, -14);
    ctx.lineTo(0, 14);
    ctx.stroke();

    ctx.font = '600 10px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#6ee7b7';
    ctx.textAlign = 'center';
    ctx.fillText('Docking Target (x_goal)', 0, -26);
    ctx.restore();

    // 6. Primary Spacecraft Vehicle Body & Thruster Plume
    const ship = state.primary;
    const heading = Math.atan2(ship.vy, ship.vx);
    const thrustNorm = Math.hypot(ship.uX, ship.uY);

    ctx.save();
    ctx.translate(ship.x, ship.y);

    // Thruster exhaust flame plume (pointing opposite to thrust vector)
    if (thrustNorm > 15) {
      const tAngle = Math.atan2(ship.uY, ship.uX);
      const flameLen = Math.min(26, Math.max(8, thrustNorm * 0.08));
      const flicker = 1 + 0.2 * Math.sin(state.iteration * 1.5);

      ctx.save();
      ctx.rotate(tAngle + Math.PI); // plume fires backwards from thruster

      const grad = ctx.createRadialGradient(0, 0, 1, flameLen * flicker, 0, flameLen * flicker);
      grad.addColorStop(0, '#fef08a');
      grad.addColorStop(0.4, '#f59e0b');
      grad.addColorStop(1, 'rgba(239, 68, 68, 0)');

      ctx.beginPath();
      ctx.moveTo(-2, -5);
      ctx.lineTo(flameLen * flicker, 0);
      ctx.lineTo(-2, 5);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.restore();

      // Optimal control force vector arrow (Amber)
      ctx.save();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      const arrowX = (ship.uX / thrustNorm) * 28;
      const arrowY = (ship.uY / thrustNorm) * 28;
      ctx.lineTo(arrowX, arrowY);
      ctx.stroke();

      ctx.fillStyle = '#f59e0b';
      ctx.font = '600 9px monospace';
      ctx.fillText('u*', arrowX + 4, arrowY + 4);
      ctx.restore();
    }

    // Spacecraft chassis (Sky Blue delta-lander)
    ctx.save();
    ctx.rotate(heading);

    ctx.beginPath();
    ctx.moveTo(14, 0); // Nose cone
    ctx.lineTo(-10, -9); // Left wing
    ctx.lineTo(-6, 0); // Engine socket
    ctx.lineTo(-10, 9); // Right wing
    ctx.closePath();

    ctx.fillStyle = 'rgba(14, 165, 233, 0.4)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Center cockpit canopy
    ctx.beginPath();
    ctx.arc(2, 0, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#e0f2fe';
    ctx.fill();

    ctx.restore(); // Heading rotate
    ctx.restore(); // Ship translate

    ctx.restore(); // Viewport translate

    // 7. On-Canvas Interactive HUD Hint
    ctx.save();
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(
      'Click / Drag Canvas to launch spacecraft with impulse | Drag Emerald Ring to relocate Target',
      16,
      24
    );
    ctx.restore();
  },
};

export default LQRControlPlugin;
