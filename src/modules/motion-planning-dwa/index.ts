// FILE: src/modules/motion-planning-dwa/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface DWAObstacle {
  x: number;
  y: number;
  r: number;
}

export interface DWACandidateArc {
  v: number;
  omega: number;
  points: { x: number; y: number }[];
  collides: boolean;
  clearanceScore: number;
  headingScore: number;
  velocityScore: number;
  totalScore: number;
}

export interface DWAState {
  robot: { x: number; y: number; theta: number; v: number; omega: number };
  goal: { x: number; y: number };
  obstacles: DWAObstacle[];
  candidateArcs: DWACandidateArc[];
  bestArcIndex: number;
  trail: { x: number; y: number }[];
  dragTarget: 'goal' | 'robot' | number | null;
  iteration: number;
}

export interface DWAParams {
  maxSpeed: number;
  maxYawRate: number;
  predictTime: number;
  headingWeight: number;
  clearanceWeight: number;
  velocityWeight: number;
}

const ROBOT_RADIUS = 13;

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function evaluateDynamicWindow(
  robot: { x: number; y: number; theta: number; v: number; omega: number },
  goal: { x: number; y: number },
  obstacles: DWAObstacle[],
  params: DWAParams
): { candidateArcs: DWACandidateArc[]; bestArcIndex: number } {
  const vSamples = 5;
  const omegaSamples = 13;
  const dt = 0.1;
  const simSteps = Math.max(5, Math.round(params.predictTime / dt));

  const candidateArcs: DWACandidateArc[] = [];
  let bestScore = -Infinity;
  let bestArcIndex = -1;

  for (let vi = 1; vi <= vSamples; vi++) {
    const v = (vi / vSamples) * params.maxSpeed;
    for (let wi = 0; wi < omegaSamples; wi++) {
      const frac = (wi / (omegaSamples - 1)) * 2 - 1;
      const omega = frac * params.maxYawRate;

      const points: { x: number; y: number }[] = [{ x: robot.x, y: robot.y }];
      let simX = robot.x;
      let simY = robot.y;
      let simTheta = robot.theta;
      let minClearance = 150;
      let collides = false;

      for (let s = 0; s < simSteps; s++) {
        simX += v * Math.cos(simTheta) * dt;
        simY += v * Math.sin(simTheta) * dt;
        simTheta += omega * dt;
        points.push({ x: simX, y: simY });

        for (const obs of obstacles) {
          const dist = Math.hypot(simX - obs.x, simY - obs.y) - obs.r - ROBOT_RADIUS;
          if (dist < minClearance) minClearance = dist;
          if (dist <= 0) {
            collides = true;
            break;
          }
        }
        // Truncate rollout immediately on collision so arc stops at the obstacle surface
        if (collides) break;
      }

      const goalHeading = Math.atan2(goal.y - simY, goal.x - simX);
      const headingDiff = Math.abs(normalizeAngle(goalHeading - simTheta));
      const headingScore = 1 - headingDiff / Math.PI;
      const clearanceScore = collides ? 0 : Math.min(1, Math.max(0, minClearance / 95));
      const velocityScore = v / Math.max(1, params.maxSpeed);

      const totalScore = collides
        ? -1000
        : params.headingWeight * headingScore +
          params.clearanceWeight * clearanceScore +
          params.velocityWeight * velocityScore;

      const idx = candidateArcs.length;
      candidateArcs.push({
        v,
        omega,
        points,
        collides,
        clearanceScore,
        headingScore,
        velocityScore,
        totalScore,
      });

      if (!collides && totalScore > bestScore) {
        bestScore = totalScore;
        bestArcIndex = idx;
      }
    }
  }

  return { candidateArcs, bestArcIndex };
}

export const DWAPlugin: AlgorithmPlugin<DWAState, DWAParams> = {
  id: 'motion-planning-dwa',
  title: 'Dynamic Window Approach (DWA) Local Planner',
  shortDescription:
    'Samples admissible velocity pairs (v, ω), forward-simulates circular arc rollouts over horizon T, and selects the collision-free arc maximizing goal heading, obstacle clearance, and forward speed.',
  category: 'motion-planning',

  documentation: {
    intuition:
      'Each constant velocity pair (v, ω) produces a circular arc trajectory over a short lookahead time T. DWA fans out dozens of candidate arcs like whiskers: arcs that hit obstacles are pruned and truncated at impact in Rose, safe arcs are scored in Amber, and the highest-scoring trajectory is highlighted in glowing Emerald.',
    governingEquationLatex:
      'G(\\textcolor{#38bdf8}{v}, \\textcolor{#38bdf8}{\\omega}) = \\sigma\\!\\left( \\alpha \\cdot \\textcolor{#10b981}{\\text{heading}(v,\\omega)} + \\beta \\cdot \\textcolor{#f59e0b}{\\text{dist}(v,\\omega)} + \\gamma \\cdot \\textcolor{#38bdf8}{\\text{vel}(v,\\omega)} - \\textcolor{#f43f5e}{\\text{collision}} \\right)',
    pseudocode: [
      'function DWA_Control(pose, v_curr, ω_curr, goal, obstacles):',
      '  V_d = computeDynamicWindow(v_curr, ω_curr, limits)',
      '  for each (v, ω) sampled in V_d:',
      '    arc = simulateForwardTrajectory(pose, v, ω, T_predict)',
      '    if minObstacleClearance(arc, obstacles) <= R_robot: markCollision(arc)',
      '    score(arc) = α·heading(arc, goal) + β·clearance(arc) + γ·velocity(v)',
      '  (v*, ω*) = argmax_{(v, ω)} score(arc)  // Execute winning arc',
    ],
  },

  controls: [
    {
      id: 'predictTime',
      label: 'Rollout Horizon (T)',
      type: 'slider',
      min: 0.8,
      max: 3.0,
      step: 0.2,
      defaultValue: 1.8,
      unit: 's',
      description: 'Forward simulation time horizon for candidate arcs.',
    },
    {
      id: 'maxSpeed',
      label: 'Max Linear Speed (v)',
      type: 'slider',
      min: 30,
      max: 110,
      step: 5,
      defaultValue: 70,
      unit: 'px/s',
      description: 'Maximum forward velocity of the robot.',
    },
    {
      id: 'maxYawRate',
      label: 'Max Yaw Rate (ω)',
      type: 'slider',
      min: 0.8,
      max: 2.6,
      step: 0.2,
      defaultValue: 1.6,
      unit: 'rad/s',
      description: 'Maximum angular turning rate in rad/s.',
    },
    {
      id: 'headingWeight',
      label: 'Heading Weight (α)',
      type: 'slider',
      min: 0.2,
      max: 3.0,
      step: 0.2,
      defaultValue: 1.6,
      description: 'Preference for pointing directly toward the goal.',
    },
    {
      id: 'clearanceWeight',
      label: 'Clearance Weight (β)',
      type: 'slider',
      min: 0.2,
      max: 3.0,
      step: 0.2,
      defaultValue: 1.4,
      description: 'Preference for keeping wide buffer distance from obstacles.',
    },
    {
      id: 'velocityWeight',
      label: 'Velocity Weight (γ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      defaultValue: 0.8,
      description: 'Preference for fast forward progress.',
    },
  ],

  createInitialState(params: DWAParams): DWAState {
    const robot = { x: -235, y: 110, theta: -0.35, v: 0, omega: 0 };
    const goal = { x: 225, y: -95 };
    const obstacles: DWAObstacle[] = [
      { x: -115, y: 55, r: 30 },
      { x: -25, y: -35, r: 34 },
      { x: 10, y: 80, r: 28 },
      { x: 95, y: -5, r: 32 },
      { x: 115, y: -110, r: 26 },
    ];
    const evalRes = evaluateDynamicWindow(robot, goal, obstacles, params);

    return {
      robot,
      goal,
      obstacles,
      candidateArcs: evalRes.candidateArcs,
      bestArcIndex: evalRes.bestArcIndex,
      trail: [{ x: robot.x, y: robot.y }],
      dragTarget: null,
      iteration: 0,
    };
  },

  step(currentState: DWAState, params: DWAParams) {
    const robot = { ...currentState.robot };
    const goal = { ...currentState.goal };
    const obstacles = currentState.obstacles.map((o) => ({ ...o }));

    const distToGoal = Math.hypot(goal.x - robot.x, goal.y - robot.y);
    const { candidateArcs, bestArcIndex } = evaluateDynamicWindow(robot, goal, obstacles, params);

    const controlDt = 0.065;
    if (distToGoal > 16) {
      if (bestArcIndex >= 0) {
        const best = candidateArcs[bestArcIndex];
        robot.v = best.v;
        robot.omega = best.omega;
        robot.x += robot.v * Math.cos(robot.theta) * controlDt;
        robot.y += robot.v * Math.sin(robot.theta) * controlDt;
        robot.theta = normalizeAngle(robot.theta + robot.omega * controlDt);
      } else {
        robot.v = 0;
        robot.omega = params.maxYawRate * 0.8;
        robot.theta = normalizeAngle(robot.theta + robot.omega * controlDt);
      }
    } else {
      robot.v = 0;
      robot.omega = 0;
    }

    const trail = [...currentState.trail, { x: robot.x, y: robot.y }].slice(-220);
    const nextIter = currentState.iteration + 1;
    const anyCollision = candidateArcs.some((a) => a.collides);
    const activeLine = bestArcIndex >= 0 ? (nextIter % 2 === 0 ? 6 : 5) : anyCollision ? 4 : 3;

    return {
      nextState: {
        robot,
        goal,
        obstacles,
        candidateArcs,
        bestArcIndex,
        trail,
        dragTarget: currentState.dragTarget,
        iteration: nextIter,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: DWAState): MetricReadout[] {
    const distToGoal = Math.hypot(state.goal.x - state.robot.x, state.goal.y - state.robot.y);
    const validCount = state.candidateArcs.filter((a) => !a.collides).length;
    return [
      { id: 'v', label: 'Winning Speed v*', value: state.robot.v.toFixed(1), unit: 'px/s', color: 'sky' },
      { id: 'omega', label: 'Winning Yaw ω*', value: state.robot.omega.toFixed(2), unit: 'rad/s', color: 'sky' },
      { id: 'arcs', label: 'Safe Rollout Arcs', value: `${validCount}/${state.candidateArcs.length}`, color: 'amber' },
      { id: 'dist', label: 'Goal Distance', value: distToGoal.toFixed(1), unit: 'px', color: 'emerald' },
    ];
  },

  onPointerInteraction(
    state: DWAState,
    event: CanvasPointerEvent,
    params: DWAParams
  ): DWAState {
    const nextObstacles = state.obstacles.map((o) => ({ ...o }));
    let nextGoal = { ...state.goal };
    let nextRobot = { ...state.robot };
    let nextTrail = [...state.trail];
    let dragTarget = state.dragTarget;

    if (!event.isDragging) {
      if (Math.hypot(event.worldX - nextGoal.x, event.worldY - nextGoal.y) < 26) {
        dragTarget = 'goal';
      } else if (Math.hypot(event.worldX - nextRobot.x, event.worldY - nextRobot.y) < 24) {
        dragTarget = 'robot';
      } else {
        const hitIdx = nextObstacles.findIndex(
          (o) => Math.hypot(event.worldX - o.x, event.worldY - o.y) <= o.r + 10
        );
        if (hitIdx >= 0) {
          dragTarget = hitIdx;
        } else {
          nextGoal = { x: event.worldX, y: event.worldY };
          dragTarget = 'goal';
        }
      }
    }

    if (dragTarget === 'goal') {
      nextGoal = {
        x: Math.max(-270, Math.min(270, event.worldX)),
        y: Math.max(-165, Math.min(165, event.worldY)),
      };
    } else if (dragTarget === 'robot') {
      nextRobot = {
        ...nextRobot,
        x: Math.max(-270, Math.min(270, event.worldX)),
        y: Math.max(-165, Math.min(165, event.worldY)),
      };
      nextTrail = [{ x: nextRobot.x, y: nextRobot.y }];
    } else if (typeof dragTarget === 'number' && nextObstacles[dragTarget]) {
      nextObstacles[dragTarget] = {
        ...nextObstacles[dragTarget],
        x: Math.max(-250, Math.min(250, event.worldX)),
        y: Math.max(-155, Math.min(155, event.worldY)),
      };
    }

    const evalRes = evaluateDynamicWindow(nextRobot, nextGoal, nextObstacles, params);

    return {
      ...state,
      robot: nextRobot,
      goal: nextGoal,
      obstacles: nextObstacles,
      candidateArcs: evalRes.candidateArcs,
      bestArcIndex: evalRes.bestArcIndex,
      trail: nextTrail,
      dragTarget,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: DWAState,
    _params: DWAParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Obstacles (Rose) + safety inflation ring
    for (const obs of state.obstacles) {
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r + ROBOT_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 2. Executed Trail (Sky Blue)
    if (state.trail.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.trail[0].x, state.trail[0].y);
      for (let i = 1; i < state.trail.length; i++) {
        ctx.lineTo(state.trail[i].x, state.trail[i].y);
      }
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 3. Candidate Rollout Arcs (Rose = Collision truncated at impact, Amber = Feasible)
    for (let i = 0; i < state.candidateArcs.length; i++) {
      if (i === state.bestArcIndex) continue;
      const arc = state.candidateArcs[i];
      if (arc.points.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(arc.points[0].x, arc.points[0].y);
      for (let j = 1; j < arc.points.length; j++) {
        ctx.lineTo(arc.points[j].x, arc.points[j].y);
      }

      if (arc.collides) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.34)';
        ctx.lineWidth = 1;
        ctx.stroke();
        const impact = arc.points[arc.points.length - 1];
        ctx.beginPath();
        ctx.arc(impact.x, impact.y, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.75)';
        ctx.fill();
      } else {
        const alpha = 0.14 + 0.35 * Math.max(0, Math.min(1, arc.totalScore / 3.5));
        ctx.strokeStyle = `rgba(245, 158, 11, ${alpha.toFixed(2)})`;
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }
    }

    // 4. Winning Rollout Arc (Glowing Emerald)
    if (state.bestArcIndex >= 0 && state.candidateArcs[state.bestArcIndex]) {
      const best = state.candidateArcs[state.bestArcIndex];
      ctx.save();
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(best.points[0].x, best.points[0].y);
      for (let j = 1; j < best.points.length; j++) {
        ctx.lineTo(best.points[j].x, best.points[j].y);
      }
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.restore();
    }

    // 5. Goal Beacon (Emerald)
    ctx.beginPath();
    ctx.arc(state.goal.x, state.goal.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 6. Robot Pose (Sky Blue)
    ctx.save();
    ctx.translate(state.robot.x, state.robot.y);
    ctx.rotate(state.robot.theta);
    ctx.beginPath();
    ctx.arc(0, 0, ROBOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(ROBOT_RADIUS + 6, 0);
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();

    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Drag Rose Obstacles, Sky Blue Robot, or Emerald Goal to reshape DWA rollouts',
      16,
      24
    );
    ctx.restore();
  },
};

export default DWAPlugin;
