// FILE: src/modules/motion-planning-potential-fields/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface Point2D {
  x: number;
  y: number;
}

export interface ObstacleSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PotentialFieldsState {
  timeStep: number;
  robot: Point2D;
  goal: Point2D;
  obstacles: ObstacleSegment[];
  drawAnchor: Point2D | null;
  forceAtt: Point2D;
  forceRep: Point2D;
  forceNet: Point2D;
  isTrapped: boolean;
  reachedGoal: boolean;
  trail: Point2D[];
  scenario: string;
}

export interface PotentialFieldsParams {
  kAtt: number;
  kRep: number;
  repDistance: number;
  scenario: string;
  showForceQuivers: boolean;
}

function dist(p1: Point2D, p2: Point2D): number {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function closestPointOnSegment(p: Point2D, seg: ObstacleSegment): { pt: Point2D; d: number } {
  const dx = seg.x2 - seg.x1;
  const dy = seg.y2 - seg.y1;
  const l2 = dx * dx + dy * dy;

  if (l2 === 0) {
    const d = dist(p, { x: seg.x1, y: seg.y1 });
    return { pt: { x: seg.x1, y: seg.y1 }, d };
  }

  let t = ((p.x - seg.x1) * dx + (p.y - seg.y1) * dy) / l2;
  t = Math.max(0, Math.min(1, t));

  const cx = seg.x1 + t * dx;
  const cy = seg.y1 + t * dy;
  const d = dist(p, { x: cx, y: cy });

  return { pt: { x: cx, y: cy }, d };
}

function computeForces(
  pos: Point2D,
  goal: Point2D,
  obstacles: ObstacleSegment[],
  kAtt: number,
  kRep: number,
  d0: number
): { fAtt: Point2D; fRep: Point2D; fNet: Point2D } {
  const dGoal = dist(pos, goal);
  const toGoalX = goal.x - pos.x;
  const toGoalY = goal.y - pos.y;

  let fAttX = 0;
  let fAttY = 0;

  if (dGoal > 0) {
    fAttX = (toGoalX / dGoal) * kAtt * Math.min(dGoal, 100);
    fAttY = (toGoalY / dGoal) * kAtt * Math.min(dGoal, 100);
  }

  let fRepX = 0;
  let fRepY = 0;

  for (const seg of obstacles) {
    const { pt, d } = closestPointOnSegment(pos, seg);
    if (d < d0 && d > 1.0) {
      const mag = Math.min(250, kRep * (1.0 / d - 1.0 / d0) * (1.0 / (d * d)) * 400);
      const nx = (pos.x - pt.x) / d;
      const ny = (pos.y - pt.y) / d;
      fRepX += nx * mag;
      fRepY += ny * mag;
    }
  }

  return {
    fAtt: { x: fAttX, y: fAttY },
    fRep: { x: fRepX, y: fRepY },
    fNet: { x: fAttX + fRepX, y: fAttY + fRepY }
  };
}

function createScenarioObstacles(type: string): ObstacleSegment[] {
  const segs: ObstacleSegment[] = [];

  if (type === 'gate') {
    segs.push({ x1: -20, y1: -180, x2: -20, y2: -38 });
    segs.push({ x1: -20, y1: 38, x2: -20, y2: 180 });
  } else if (type === 'cluttered') {
    segs.push({ x1: -90, y1: -110, x2: -90, y2: 20 });
    segs.push({ x1: 45, y1: -20, x2: 45, y2: 120 });
    segs.push({ x1: -30, y1: -85, x2: 95, y2: -85 });
  } else {
    // Classic U-shaped box canyon open to the left
    segs.push({ x1: -80, y1: -75, x2: 40, y2: -75 });
    segs.push({ x1: 40, y1: -75, x2: 40, y2: 75 });
    segs.push({ x1: 40, y1: 75, x2: -80, y2: 75 });
  }

  return segs;
}

export const PotentialFieldsPlugin: AlgorithmPlugin<PotentialFieldsState, PotentialFieldsParams> = {
  id: 'motion-planning-potential-fields',
  title: 'Artificial Potential Fields & Local Minima',
  shortDescription: 'Continuous gradient descent navigating attractive goal and repulsive obstacle vector fields, exposing local minima traps.',
  category: 'motion-planning',

  controls: [
    {
      id: 'kAtt',
      label: 'Attractive Gain (xi)',
      type: 'slider',
      min: 0.2,
      max: 2.5,
      step: 0.1,
      defaultValue: 0.8,
      description: 'Strengthens pull toward goal.'
    },
    {
      id: 'kRep',
      label: 'Repulsive Gain (eta)',
      type: 'slider',
      min: 200,
      max: 3000,
      step: 200,
      defaultValue: 1400,
      description: 'Strengthens push away from obstacle surfaces.'
    },
    {
      id: 'repDistance',
      label: 'Influence Distance (d_0)',
      type: 'slider',
      min: 30,
      max: 130,
      step: 5,
      defaultValue: 75,
      unit: 'px',
      description: 'Obstacle repulsion acts only within this distance.'
    },
    {
      id: 'scenario',
      label: 'Trap Geometry',
      type: 'select',
      options: [
        { label: 'U-Shaped Trap (Local Minimum)', value: 'u-trap' },
        { label: 'Narrow Gate Pass', value: 'gate' },
        { label: 'Cluttered Barriers', value: 'cluttered' }
      ],
      defaultValue: 'u-trap'
    },
    {
      id: 'showForceQuivers',
      label: 'Show 2D Force Quiver Field',
      type: 'toggle',
      defaultValue: true
    }
  ],

  documentation: {
    intuition:
      'Artificial Potential Fields treat the robot as a particle sliding down a composite potential energy landscape. The goal creates a deep attractive parabolic basin, while obstacles erect steep repulsive hills. When repulsive vectors exactly cancel the attractive pull before reaching the goal, the resultant force drops to zero (F_net = 0), trapping the robot in a local minimum.',
    governingEquationLatex:
      '\\mathbf{F}_{\\text{net}}(\\textcolor{#0ea5e9}{q}) = -\\textcolor{#38bdf8}{\\nabla U_{\\text{att}}(\\textcolor{#0ea5e9}{q})} - \\sum_{i} \\textcolor{#f43f5e}{\\nabla U_{\\text{rep},i}(\\textcolor{#0ea5e9}{q})}, \\quad \\textcolor{#f59e0b}{\\mathbf{F}_{\\text{net}}} = \\mathbf{0} \\implies \\text{Local Minima}',
    pseudocode: [
      '1. Attractive Pull: F_att = -k_att * (q - q_goal)',
      '2. Surface Query: For each obstacle segment, find closest distance d_i',
      '3. Influence Test: If d_i < d_0, evaluate repulsive gradient:',
      '4. Repulsive Push: F_rep,i = k_rep * (1/d_i - 1/d_0) * (1/d_i^2) * normal_vector',
      '5. Vector Superposition: F_net = F_att + sum(F_rep,i)',
      '6. Equilibrium Check: If ||F_net|| < threshold and dist(q, goal) > tol: Trapped!',
      '7. Euler Step: q_next = q + clamp(F_net * dt, max_velocity)'
    ]
  },

  createInitialState(params: PotentialFieldsParams): PotentialFieldsState {
    const scenario = params.scenario || 'u-trap';
    const obstacles = createScenarioObstacles(scenario);
    const robot = scenario === 'u-trap' ? { x: -160, y: 0 } : { x: -190, y: -25 };
    const goal = { x: 190, y: 0 };

    const { fAtt, fRep, fNet } = computeForces(
      robot,
      goal,
      obstacles,
      params.kAtt,
      params.kRep,
      params.repDistance
    );

    return {
      timeStep: 0,
      robot,
      goal,
      obstacles,
      drawAnchor: null,
      forceAtt: fAtt,
      forceRep: fRep,
      forceNet: fNet,
      isTrapped: false,
      reachedGoal: false,
      trail: [robot],
      scenario
    };
  },

  step(currentState: PotentialFieldsState, params: PotentialFieldsParams) {
    if (currentState.reachedGoal) {
      return {
        nextState: currentState,
        isComplete: true,
        activePseudocodeLine: 6
      };
    }

    const nextStep = currentState.timeStep + 1;
    const { fAtt, fRep, fNet } = computeForces(
      currentState.robot,
      currentState.goal,
      currentState.obstacles,
      params.kAtt,
      params.kRep,
      params.repDistance
    );

    const fMag = Math.hypot(fNet.x, fNet.y);
    const dGoal = dist(currentState.robot, currentState.goal);

    if (dGoal < 16) {
      return {
        nextState: {
          ...currentState,
          timeStep: nextStep,
          reachedGoal: true,
          isTrapped: false,
          forceAtt: { x: 0, y: 0 },
          forceRep: { x: 0, y: 0 },
          forceNet: { x: 0, y: 0 }
        },
        isComplete: true,
        activePseudocodeLine: 6
      };
    }

    const maxSpeed = 3.5;
    const speed = Math.min(maxSpeed, fMag * 0.08);
    const moveX = fMag > 0 ? (fNet.x / fMag) * speed : 0;
    const moveY = fMag > 0 ? (fNet.y / fMag) * speed : 0;

    const nextPos = {
      x: currentState.robot.x + moveX,
      y: currentState.robot.y + moveY
    };

    const maxTrail = 120;
    const trail = [...currentState.trail, nextPos].slice(-maxTrail);

    // Detect equilibrium or limit-cycle stall inside local minimum
    const isStalled =
      trail.length >= 12 &&
      dist(nextPos, trail[trail.length - 10]) < 2.2 &&
      dGoal > 30;
    const isTrapped = (fMag < 1.2 && dGoal > 30) || isStalled;

    return {
      nextState: {
        timeStep: nextStep,
        robot: nextPos,
        goal: currentState.goal,
        obstacles: currentState.obstacles,
        drawAnchor: null,
        forceAtt: fAtt,
        forceRep: fRep,
        forceNet: fNet,
        isTrapped,
        reachedGoal: false,
        trail,
        scenario: currentState.scenario
      },
      isComplete: isTrapped,
      activePseudocodeLine: isTrapped ? 5 : nextStep % 5
    };
  },

  onPointerInteraction(
    currentState: PotentialFieldsState,
    event: CanvasPointerEvent,
    params: PotentialFieldsParams
  ): PotentialFieldsState {
    const wx = event.worldX;
    const wy = event.worldY;

    // Right-click & drag: Construct new obstacle wall segment!
    if (event.button === 'right') {
      if (!event.isDragging || !currentState.drawAnchor) {
        const anchor = { x: wx, y: wy };
        const nextObstacles = [
          ...currentState.obstacles,
          { x1: wx, y1: wy, x2: wx + 1, y2: wy + 1 }
        ];
        return {
          ...currentState,
          drawAnchor: anchor,
          obstacles: nextObstacles,
          isTrapped: false,
          reachedGoal: false
        };
      } else {
        const nextObstacles = currentState.obstacles.map((seg, idx) =>
          idx === currentState.obstacles.length - 1
            ? { x1: currentState.drawAnchor!.x, y1: currentState.drawAnchor!.y, x2: wx, y2: wy }
            : { ...seg }
        );
        return {
          ...currentState,
          obstacles: nextObstacles,
          isTrapped: false,
          reachedGoal: false
        };
      }
    }

    if (event.shiftKey) {
      const nextGoal = { x: wx, y: wy };
      const { fAtt, fRep, fNet } = computeForces(
        currentState.robot,
        nextGoal,
        currentState.obstacles,
        params.kAtt,
        params.kRep,
        params.repDistance
      );
      return {
        ...currentState,
        goal: nextGoal,
        forceAtt: fAtt,
        forceRep: fRep,
        forceNet: fNet,
        reachedGoal: false,
        isTrapped: false,
        trail: [currentState.robot]
      };
    }

    const nextPos = { x: wx, y: wy };
    const { fAtt, fRep, fNet } = computeForces(
      nextPos,
      currentState.goal,
      currentState.obstacles,
      params.kAtt,
      params.kRep,
      params.repDistance
    );

    return {
      ...currentState,
      robot: nextPos,
      forceAtt: fAtt,
      forceRep: fRep,
      forceNet: fNet,
      reachedGoal: false,
      isTrapped: false,
      trail: [nextPos]
    };
  },

  getMetrics(currentState: PotentialFieldsState): MetricReadout[] {
    const fNetMag = Math.hypot(currentState.forceNet.x, currentState.forceNet.y).toFixed(1);
    const dGoal = dist(currentState.robot, currentState.goal).toFixed(0);

    return [
      {
        id: 'fnet',
        label: 'Net Force (|F_net|)',
        value: `${fNetMag} N`,
        color: currentState.isTrapped ? 'rose' : 'amber'
      },
      {
        id: 'distGoal',
        label: 'Distance to Goal',
        value: `${dGoal} px`,
        color: currentState.reachedGoal ? 'emerald' : 'sky'
      },
      {
        id: 'status',
        label: 'Dynamic Status',
        value: currentState.isTrapped
          ? 'TRAPPED IN LOCAL MINIMUM!'
          : currentState.reachedGoal
          ? 'Goal Basin Reached'
          : 'Navigating Gradient',
        color: currentState.isTrapped
          ? 'rose'
          : currentState.reachedGoal
          ? 'emerald'
          : 'sky'
      }
    ];
  },

  render(ctx, currentState, params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    if (params.showForceQuivers) {
      const step = 42;
      const xMin = -280;
      const xMax = 280;
      const yMin = -170;
      const yMax = 170;

      for (let x = xMin; x <= xMax; x += step) {
        for (let y = yMin; y <= yMax; y += step) {
          const sample = { x, y };
          const { fNet } = computeForces(
            sample,
            currentState.goal,
            currentState.obstacles,
            params.kAtt,
            params.kRep,
            params.repDistance
          );
          const mag = Math.hypot(fNet.x, fNet.y);
          if (mag > 0.05) {
            const arrowLen = Math.min(18, Math.max(5, mag * 0.3));
            const nx = fNet.x / mag;
            const ny = fNet.y / mag;

            ctx.strokeStyle = 'rgba(56, 189, 248, 0.16)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + nx * arrowLen, y + ny * arrowLen);
            ctx.stroke();
          }
        }
      }
    }

    for (const seg of currentState.obstacles) {
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.12)';
      ctx.lineWidth = params.repDistance * 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.stroke();

      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(seg.x1, seg.y1);
      ctx.lineTo(seg.x2, seg.y2);
      ctx.stroke();
      ctx.lineCap = 'butt';
    }

    if (currentState.trail.length > 1) {
      ctx.strokeStyle = 'rgba(14, 165, 233, 0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(currentState.trail[0].x, currentState.trail[0].y);
      for (let i = 1; i < currentState.trail.length; i++) {
        ctx.lineTo(currentState.trail[i].x, currentState.trail[i].y);
      }
      ctx.stroke();
    }

    const rx = currentState.robot.x;
    const ry = currentState.robot.y;

    // F_att (Sky Blue Vector)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + currentState.forceAtt.x * 0.55, ry + currentState.forceAtt.y * 0.55);
    ctx.stroke();

    // F_rep (Rose Vector)
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + currentState.forceRep.x * 0.55, ry + currentState.forceRep.y * 0.55);
    ctx.stroke();

    // F_net (Amber Vector #f59e0b)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + currentState.forceNet.x * 0.55, ry + currentState.forceNet.y * 0.55);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(currentState.goal.x, currentState.goal.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.3)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.font = 'bold 10px sans-serif';
    ctx.fillStyle = '#6ee7b7';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('GOAL', currentState.goal.x, currentState.goal.y);

    ctx.beginPath();
    ctx.arc(rx, ry, 9, 0, Math.PI * 2);
    ctx.fillStyle = currentState.isTrapped ? '#f43f5e' : '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    if (currentState.isTrapped) {
      ctx.beginPath();
      ctx.arc(rx, ry, 22, 0, Math.PI * 2);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = 'bold 11px sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.textAlign = 'center';
      ctx.fillText('LOCAL MINIMUM (F_net ≈ 0)', rx, ry - 26);
    }

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Drag to move robot; Shift-drag to move Goal; Right-drag to draw wall traps', 0, 195);

    ctx.restore();
  }
};
