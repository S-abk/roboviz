import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface IKState {
  targetX: number;
  targetY: number;
  currentTheta1: number;
  currentTheta2: number;
  goalTheta1: number;
  goalTheta2: number;
  isReachable: boolean;
  distanceToTarget: number;
}

export interface IKParams {
  L1: number;
  L2: number;
  elbowUp: boolean;
  smoothingRate: number;
}

function solveAnalyticalIK(x: number, y: number, L1: number, L2: number, elbowUp: boolean) {
  const distSq = x * x + y * y;
  const dist = Math.sqrt(distSq);
  const maxReach = L1 + L2;
  const minReach = Math.abs(L1 - L2);
  const isReachable = dist <= maxReach && dist >= minReach;

  let clampedX = x;
  let clampedY = y;
  if (dist > maxReach && dist > 0) {
    clampedX = (x / dist) * (maxReach - 0.01);
    clampedY = (y / dist) * (maxReach - 0.01);
  } else if (dist < minReach && dist > 0) {
    clampedX = (x / dist) * (minReach + 0.01);
    clampedY = (y / dist) * (minReach + 0.01);
  }

  const c2 = Math.max(
    -1,
    Math.min(1, (clampedX * clampedX + clampedY * clampedY - L1 * L1 - L2 * L2) / (2 * L1 * L2))
  );
  const s2 = (elbowUp ? -1 : 1) * Math.sqrt(Math.max(0, 1 - c2 * c2));
  const theta2 = Math.atan2(s2, c2);

  const k1 = L1 + L2 * c2;
  const k2 = L2 * s2;
  const theta1 = Math.atan2(clampedY, clampedX) - Math.atan2(k2, k1);

  return { theta1, theta2, isReachable, dist };
}

export const KinematicsIKPlugin: AlgorithmPlugin<IKState, IKParams> = {
  id: 'kinematics-2d-ik',
  title: '2-DOF Planar Inverse Kinematics',
  shortDescription: 'Analytical Law-of-Cosines IK solver with workspace boundaries & elbow posture switching.',
  category: 'kinematics-control',

  controls: [
    { id: 'L1', label: 'Link 1 Length (L1)', type: 'slider', min: 60, max: 200, step: 5, defaultValue: 130, unit: 'px' },
    { id: 'L2', label: 'Link 2 Length (L2)', type: 'slider', min: 50, max: 180, step: 5, defaultValue: 100, unit: 'px' },
    { id: 'elbowUp', label: 'Elbow-Up Posture', type: 'toggle', defaultValue: true },
    { id: 'smoothingRate', label: 'Servo Tracking Gain', type: 'slider', min: 0.05, max: 1.0, step: 0.05, defaultValue: 0.25 }
  ],

  documentation: {
    intuition:
      'Inverse Kinematics (IK) solves for the joint angles (θ₁, θ₂) needed to place a robotic end-effector at a target coordinate (x, y). Using the Law of Cosines on the triangle formed by the shoulder, elbow, and wrist yields two valid geometric postures: Elbow-Up and Elbow-Down.',
    governingEquationLatex:
      '\\cos(\\textcolor{#0ea5e9}{\\theta_2}) = \\frac{\\textcolor{#10b981}{x^2 + y^2} - L_1^2 - L_2^2}{2 L_1 L_2}',
    pseudocode: [
      '1. Measure target distance d = sqrt(x² + y²)',
      '2. Check reachability: |L₁ - L₂| ≤ d ≤ L₁ + L₂',
      '3. Clamp (x, y) to workspace boundary if unreachable',
      '4. Compute cos(θ₂) = (x² + y² - L₁² - L₂²) / (2·L₁·L₂)',
      '5. Select sin(θ₂) = ±sqrt(1 - cos²(θ₂)) based on Elbow posture',
      '6. Solve θ₁ = atan2(y, x) - atan2(L₂·sin(θ₂), L₁ + L₂·cos(θ₂))',
      '7. Interpolate joint servos toward (θ₁, θ₂)'
    ]
  },

  createInitialState(params: IKParams): IKState {
    const targetX = 140;
    const targetY = -80;
    const sol = solveAnalyticalIK(targetX, targetY, params.L1, params.L2, params.elbowUp);
    return {
      targetX,
      targetY,
      currentTheta1: sol.theta1,
      currentTheta2: sol.theta2,
      goalTheta1: sol.theta1,
      goalTheta2: sol.theta2,
      isReachable: sol.isReachable,
      distanceToTarget: sol.dist
    };
  },

  step(state: IKState, params: IKParams) {
    const sol = solveAnalyticalIK(state.targetX, state.targetY, params.L1, params.L2, params.elbowUp);
    const d1 = sol.theta1 - state.currentTheta1;
    const d2 = sol.theta2 - state.currentTheta2;

    const nextTheta1 = state.currentTheta1 + d1 * params.smoothingRate;
    const nextTheta2 = state.currentTheta2 + d2 * params.smoothingRate;
    const converged = Math.abs(d1) < 0.002 && Math.abs(d2) < 0.002;

    return {
      nextState: {
        ...state,
        currentTheta1: nextTheta1,
        currentTheta2: nextTheta2,
        goalTheta1: sol.theta1,
        goalTheta2: sol.theta2,
        isReachable: sol.isReachable,
        distanceToTarget: sol.dist
      },
      isComplete: converged,
      activePseudocodeLine: converged ? 5 : 6
    };
  },

  onPointerInteraction(state: IKState, event: CanvasPointerEvent, params: IKParams): IKState {
    const sol = solveAnalyticalIK(event.worldX, event.worldY, params.L1, params.L2, params.elbowUp);
    return {
      ...state,
      targetX: event.worldX,
      targetY: event.worldY,
      currentTheta1: sol.theta1,
      currentTheta2: sol.theta2,
      goalTheta1: sol.theta1,
      goalTheta2: sol.theta2,
      isReachable: sol.isReachable,
      distanceToTarget: sol.dist
    };
  },

  getMetrics(state: IKState): MetricReadout[] {
    const deg1 = ((state.currentTheta1 * 180) / Math.PI).toFixed(1);
    const deg2 = ((state.currentTheta2 * 180) / Math.PI).toFixed(1);
    return [
      { id: 'theta1', label: 'Shoulder θ₁', value: `${deg1}°`, color: 'sky' },
      { id: 'theta2', label: 'Elbow θ₂', value: `${deg2}°`, color: 'sky' },
      {
        id: 'status',
        label: 'Workspace Status',
        value: state.isReachable ? 'Reachable' : 'Out of Bounds',
        color: state.isReachable ? 'emerald' : 'rose'
      }
    ];
  },

  render(ctx, state, params, viewport) {
    const cx = viewport.width / 2;
    const cy = viewport.height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    const maxR = params.L1 + params.L2;
    const minR = Math.abs(params.L1 - params.L2);

    ctx.beginPath();
    ctx.arc(0, 0, maxR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.25)';
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (minR > 0) {
      ctx.beginPath();
      ctx.arc(0, 0, minR, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const elbowX = params.L1 * Math.cos(state.currentTheta1);
    const elbowY = params.L1 * Math.sin(state.currentTheta1);
    const wristX = elbowX + params.L2 * Math.cos(state.currentTheta1 + state.currentTheta2);
    const wristY = elbowY + params.L2 * Math.sin(state.currentTheta1 + state.currentTheta2);

    ctx.lineCap = 'round';
    ctx.lineWidth = 10;
    ctx.strokeStyle = '#0ea5e9';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(elbowX, elbowY);
    ctx.stroke();

    ctx.lineWidth = 8;
    ctx.strokeStyle = '#38bdf8';
    ctx.beginPath();
    ctx.moveTo(elbowX, elbowY);
    ctx.lineTo(wristX, wristY);
    ctx.stroke();

    [
      { x: 0, y: 0, r: 9, color: '#f8fafc' },
      { x: elbowX, y: elbowY, r: 7, color: '#0f172a' },
      { x: wristX, y: wristY, r: 6, color: '#10b981' }
    ].forEach((joint) => {
      ctx.beginPath();
      ctx.arc(joint.x, joint.y, joint.r, 0, Math.PI * 2);
      ctx.fillStyle = joint.color;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#0ea5e9';
      ctx.stroke();
    });

    const targetColor = state.isReachable ? '#10b981' : '#f43f5e';
    ctx.strokeStyle = targetColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(state.targetX, state.targetY, 10, 0, Math.PI * 2);
    ctx.moveTo(state.targetX - 15, state.targetY);
    ctx.lineTo(state.targetX + 15, state.targetY);
    ctx.moveTo(state.targetX, state.targetY - 15);
    ctx.lineTo(state.targetX, state.targetY + 15);
    ctx.stroke();

    ctx.restore();
  }
};
