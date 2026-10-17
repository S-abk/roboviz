// FILE: src/modules/kinematics-jacobian-ik/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface IKState {
  angles: number[];
  target: { x: number; y: number };
  activeJointIdx: number;
  lastDeltaTheta: number;
  endEffectorTrail: { x: number; y: number }[];
  iteration: number;
}

export interface IKParams {
  numLinks: number;
  linkLength: number;
  dampingAlpha: number;
  ccdSweepsPerStep: number;
}

function computeForwardKinematics(
  angles: number[],
  linkLength: number
): { x: number; y: number; cumAngle: number }[] {
  const joints: { x: number; y: number; cumAngle: number }[] = [{ x: 0, y: 0, cumAngle: 0 }];
  let x = 0;
  let y = 0;
  let cumAngle = 0;

  for (let i = 0; i < angles.length; i++) {
    cumAngle += angles[i];
    x += linkLength * Math.cos(cumAngle);
    y += linkLength * Math.sin(cumAngle);
    joints.push({ x, y, cumAngle });
  }
  return joints;
}

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function syncAnglesCount(angles: number[], numLinks: number): number[] {
  const res = [...angles];
  while (res.length < numLinks) res.push(0.2);
  if (res.length > numLinks) res.length = numLinks;
  return res;
}

export const HyperRedundantIKPlugin: AlgorithmPlugin<IKState, IKParams> = {
  id: 'kinematics-jacobian-ik',
  title: 'N-Link Hyper-Redundant Arm (CCD & Jacobian IK)',
  shortDescription:
    'Interactive multi-segment robotic manipulator (3 to 8 DOF) that smoothly curls around target coordinates using Cyclic Coordinate Descent (CCD) / iterative gradient steps.',
  category: 'kinematics-control',

  documentation: {
    intuition:
      'Starting from the wrist and working down to the shoulder, each joint i asks a simple geometric question: "What angle Δθᵢ should I rotate so that the current end-effector vector v_e swings directly onto the target vector v_t?" Repeating this sweep creates organic, tentacle-like motion.',
    governingEquationLatex:
      '\\Delta \\textcolor{#f59e0b}{\\theta_i} = \\alpha \\, \\text{atan2}\\!\\left( \\textcolor{#38bdf8}{\\mathbf{v}_e} \\times \\textcolor{#10b981}{\\mathbf{v}_t}, \\; \\textcolor{#38bdf8}{\\mathbf{v}_e} \\cdot \\textcolor{#10b981}{\\mathbf{v}_t} \\right), \\quad \\textcolor{#f43f5e}{\\mathbf{e}} = \\textcolor{#10b981}{\\mathbf{x}^*} - \\textcolor{#0ea5e9}{\\mathbf{x}_e(\\boldsymbol{\\theta})}',
    pseudocode: [
      'function CCD_InverseKinematics_Step(θ[1..N], target x*):',
      '  joints = forwardKinematics(θ)  // Compute joint positions p_i',
      '  if ||x* - p_end|| < ε_tol: return CONVERGED',
      '  for i = N down to 1:           // Tip-to-base coordinate descent',
      '    v_e = normalize(p_end - p_i)',
      '    v_t = normalize(x* - p_i)',
      '    θ[i] += α * atan2(v_e × v_t, v_e · v_t)',
    ],
  },

  controls: [
    {
      id: 'numLinks',
      label: 'Number of Links (N)',
      type: 'slider',
      min: 3,
      max: 8,
      step: 1,
      defaultValue: 6,
      description: 'Degrees of freedom (joint segments) in the hyper-redundant arm.',
    },
    {
      id: 'linkLength',
      label: 'Segment Length (L)',
      type: 'slider',
      min: 22,
      max: 48,
      step: 2,
      defaultValue: 32,
      unit: 'px',
      description: 'Length of each rigid robotic link in pixels.',
    },
    {
      id: 'dampingAlpha',
      label: 'Step Gain / Damping (α)',
      type: 'slider',
      min: 0.05,
      max: 0.65,
      step: 0.05,
      defaultValue: 0.25,
      description: 'Fraction of the angular error corrected per joint sweep.',
    },
    {
      id: 'ccdSweepsPerStep',
      label: 'Joint Updates / Step',
      type: 'slider',
      min: 1,
      max: 6,
      step: 1,
      defaultValue: 2,
      description: 'Number of individual joints updated per simulation step.',
    },
  ],

  createInitialState(params: IKParams): IKState {
    const angles = Array.from({ length: params.numLinks }, (_, i) =>
      i === 0 ? -0.6 : 0.28
    );
    const fk = computeForwardKinematics(angles, params.linkLength);
    const tip = fk[fk.length - 1];

    return {
      angles,
      target: { x: 115, y: -75 },
      activeJointIdx: params.numLinks - 1,
      lastDeltaTheta: 0,
      endEffectorTrail: [{ x: tip.x, y: tip.y }],
      iteration: 0,
    };
  },

  step(currentState: IKState, params: IKParams) {
    const angles = syncAnglesCount(currentState.angles, params.numLinks);

    let activeJointIdx = currentState.activeJointIdx % params.numLinks;
    if (activeJointIdx < 0) activeJointIdx = params.numLinks - 1;
    let lastDeltaTheta = 0;

    for (let s = 0; s < params.ccdSweepsPerStep; s++) {
      const fk = computeForwardKinematics(angles, params.linkLength);
      const endEffector = fk[fk.length - 1];
      const jointPos = fk[activeJointIdx];

      const vEx = endEffector.x - jointPos.x;
      const vEy = endEffector.y - jointPos.y;
      const vTx = currentState.target.x - jointPos.x;
      const vTy = currentState.target.y - jointPos.y;

      const cross = vEx * vTy - vEy * vTx;
      const dot = vEx * vTx + vEy * vTy;
      const rawDelta = Math.atan2(cross, dot);
      lastDeltaTheta = params.dampingAlpha * rawDelta;

      angles[activeJointIdx] = normalizeAngle(angles[activeJointIdx] + lastDeltaTheta);

      activeJointIdx = activeJointIdx - 1;
      if (activeJointIdx < 0) activeJointIdx = params.numLinks - 1;
    }

    const finalFk = computeForwardKinematics(angles, params.linkLength);
    const tip = finalFk[finalFk.length - 1];
    const posError = Math.hypot(currentState.target.x - tip.x, currentState.target.y - tip.y);

    const endEffectorTrail = [
      ...currentState.endEffectorTrail,
      { x: tip.x, y: tip.y },
    ].slice(-120);

    const activeLine = posError < 3.5 ? 2 : 6;

    return {
      nextState: {
        angles,
        target: { ...currentState.target },
        activeJointIdx,
        lastDeltaTheta,
        endEffectorTrail,
        iteration: currentState.iteration + 1,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: IKState, params: IKParams): MetricReadout[] {
    const angles = syncAnglesCount(state.angles, params.numLinks);
    const fk = computeForwardKinematics(angles, params.linkLength);
    const tip = fk[fk.length - 1];
    const posError = Math.hypot(state.target.x - tip.x, state.target.y - tip.y);
    const maxReach = params.numLinks * params.linkLength;
    const targetDist = Math.hypot(state.target.x, state.target.y);

    return [
      { id: 'err', label: 'End-Effector Error ||e||', value: posError.toFixed(1), unit: 'px', color: posError < 5 ? 'emerald' : 'rose' },
      { id: 'joint', label: 'Active CCD Joint', value: `J${state.activeJointIdx + 1}/${params.numLinks}`, color: 'amber' },
      { id: 'reach', label: 'Workspace Status', value: targetDist <= maxReach ? 'Reachable' : 'Out of Reach', color: targetDist <= maxReach ? 'emerald' : 'amber' },
      { id: 'dof', label: 'Manipulator DOFs', value: params.numLinks, color: 'sky' },
    ];
  },

  onPointerInteraction(
    state: IKState,
    event: CanvasPointerEvent,
    _params: IKParams
  ): IKState {
    return {
      ...state,
      target: {
        x: Math.max(-270, Math.min(270, event.worldX)),
        y: Math.max(-170, Math.min(170, event.worldY)),
      },
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: IKState,
    params: IKParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    const angles = syncAnglesCount(state.angles, params.numLinks);
    const joints = computeForwardKinematics(angles, params.linkLength);
    const tip = joints[joints.length - 1];
    const maxReach = params.numLinks * params.linkLength;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Reachable Workspace Boundary Circle
    ctx.beginPath();
    ctx.arc(0, 0, maxReach, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. End-Effector Trajectory Trail (Emerald)
    if (state.endEffectorTrail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.endEffectorTrail[0].x, state.endEffectorTrail[0].y);
      for (let i = 1; i < state.endEffectorTrail.length; i++) {
        ctx.lineTo(state.endEffectorTrail[i].x, state.endEffectorTrail[i].y);
      }
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 3. Active Joint CCD Geometric Vectors: v_e (Sky Blue) and v_t (Emerald)
    const activeIdx = Math.min(Math.max(0, state.activeJointIdx), joints.length - 2);
    const activeJoint = joints[activeIdx];
    if (activeJoint) {
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(activeJoint.x, activeJoint.y);
      ctx.lineTo(state.target.x, state.target.y);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(activeJoint.x, activeJoint.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 4. Residual Error Vector e = x* - x_e (Rose Dashed)
    const errDist = Math.hypot(state.target.x - tip.x, state.target.y - tip.y);
    if (errDist > 3) {
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(state.target.x, state.target.y);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 5. Kinematic Links (Sky Blue / Active Link Amber)
    for (let i = 0; i < joints.length - 1; i++) {
      const p1 = joints[i];
      const p2 = joints[i + 1];
      const thickness = Math.max(4, 11 - i * 0.9);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = i === activeIdx ? '#f59e0b' : '#0ea5e9';
      ctx.lineWidth = thickness;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // 6. Revolute Joint Pivots
    for (let i = 0; i < joints.length - 1; i++) {
      const j = joints[i];
      const isAct = i === activeIdx;
      ctx.beginPath();
      ctx.arc(j.x, j.y, isAct ? 8 : 5.5, 0, Math.PI * 2);
      ctx.fillStyle = isAct ? '#f59e0b' : '#0f172a';
      ctx.fill();
      ctx.strokeStyle = isAct ? '#fef3c7' : '#38bdf8';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 7. End-Effector Gripper
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 8. Target Coordinate Crosshair (Emerald)
    ctx.beginPath();
    ctx.arc(state.target.x, state.target.y, 12, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(state.target.x - 16, state.target.y);
    ctx.lineTo(state.target.x + 16, state.target.y);
    ctx.moveTo(state.target.x, state.target.y - 16);
    ctx.lineTo(state.target.x, state.target.y + 16);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click/Drag Emerald Target anywhere inside or outside the workspace ring',
      16,
      24
    );
    ctx.restore();
  },
};

export default HyperRedundantIKPlugin;
