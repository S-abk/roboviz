// FILE: src/modules/motion-planning-rrt-star/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface RRTNode {
  id: number;
  x: number;
  y: number;
  parentId: number | null;
  cost: number;
}

export interface Obstacle {
  x: number;
  y: number;
  radius: number;
}

export interface RRTStarState {
  timeStep: number;
  nodes: RRTNode[];
  start: { x: number; y: number };
  goal: { x: number; y: number; radius: number };
  obstacles: Obstacle[];
  lastSample: { x: number; y: number } | null;
  lastNewNodeId: number | null;
  candidateNearIds: number[];
  bestGoalNodeId: number | null;
  path: { x: number; y: number }[];
  pathCost: number;
  goalReached: boolean;
  rewireCount: number;
}

export interface RRTStarParams {
  algorithm: 'rrt-star' | 'rrt';
  stepSize: number;
  goalBias: number;
  rewireRadius: number;
  maxNodes: number;
}

const ARENA_W = 600;
const ARENA_H = 400;
const HALF_W = ARENA_W / 2;
const HALF_H = ARENA_H / 2;

const DEFAULT_OBSTACLES: Obstacle[] = [
  { x: -110, y: -40, radius: 48 },
  { x: 30, y: 50, radius: 55 },
  { x: -20, y: -90, radius: 42 },
  { x: 120, y: -60, radius: 46 },
  { x: -130, y: 100, radius: 40 }
];

function dist(x1: number, y1: number, x2: number, y2: number): number {
  return Math.hypot(x1 - x2, y1 - y2);
}

function lineIntersectsCircle(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  cx: number,
  cy: number,
  r: number
): boolean {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;

  if (lenSq === 0) {
    return dist(x1, y1, cx, cy) <= r;
  }

  let t = ((cx - x1) * dx + (cy - y1) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));

  const projX = x1 + t * dx;
  const projY = y1 + t * dy;

  return dist(projX, projY, cx, cy) <= r;
}

function isSegmentInCollision(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  obstacles: Obstacle[]
): boolean {
  for (const obs of obstacles) {
    if (lineIntersectsCircle(x1, y1, x2, y2, obs.x, obs.y, obs.radius)) {
      return true;
    }
  }
  return false;
}

function propagateSubtreeCosts(nodes: RRTNode[], rootId: number, costDelta: number): void {
  const queue: number[] = [rootId];
  while (queue.length > 0) {
    const parentId = queue.shift()!;
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].parentId === parentId) {
        nodes[i].cost -= costDelta;
        queue.push(nodes[i].id);
      }
    }
  }
}

function reconstructPath(
  nodes: RRTNode[],
  endNodeId: number,
  goal: { x: number; y: number }
): { x: number; y: number }[] {
  const path: { x: number; y: number }[] = [{ x: goal.x, y: goal.y }];
  let curr: RRTNode | undefined = nodes.find((n) => n.id === endNodeId);
  while (curr) {
    path.push({ x: curr.x, y: curr.y });
    if (curr.parentId === null) break;
    curr = nodes.find((n) => n.id === curr!.parentId);
  }
  return path.reverse();
}

export const RRTStarPlugin: AlgorithmPlugin<RRTStarState, RRTStarParams> = {
  id: 'motion-planning-rrt-star',
  title: 'RRT & RRT* Continuous Motion Planner',
  shortDescription: 'Sampling-based continuous path planning with goal biasing, obstacle collision detection, and asymptotic rewiring.',
  category: 'motion-planning',

  controls: [
    {
      id: 'algorithm',
      label: 'Planner Mode',
      type: 'select',
      options: [
        { label: 'RRT* (Asymptotically Optimal Rewiring)', value: 'rrt-star' },
        { label: 'Standard RRT (No Rewiring)', value: 'rrt' }
      ],
      defaultValue: 'rrt-star'
    },
    {
      id: 'stepSize',
      label: 'Extension Step Size (delta_q)',
      type: 'slider',
      min: 15,
      max: 45,
      step: 5,
      defaultValue: 25,
      unit: 'px'
    },
    {
      id: 'goalBias',
      label: 'Goal Sampling Bias (P_goal)',
      type: 'slider',
      min: 0.05,
      max: 0.35,
      step: 0.05,
      defaultValue: 0.15,
      description: 'Probability of setting sample point directly to the goal.'
    },
    {
      id: 'rewireRadius',
      label: 'Neighborhood Rewire Radius (r_rewire)',
      type: 'slider',
      min: 35,
      max: 110,
      step: 5,
      defaultValue: 70,
      unit: 'px'
    },
    {
      id: 'maxNodes',
      label: 'Maximum Node Budget',
      type: 'slider',
      min: 150,
      max: 800,
      step: 50,
      defaultValue: 450
    }
  ],

  documentation: {
    intuition:
      'Standard RRT rapidly explores free continuous space by growing a random tree toward unbiased and goal-directed samples. However, standard RRT paths are jagged and sub-optimal. RRT* introduces two key steps: choosing the parent in a local neighborhood that minimizes total start-to-node path cost, and rewiring neighboring branches whenever routing through the new node yields a cheaper path.',
    governingEquationLatex:
      'c(\\textcolor{#0ea5e9}{x_{\\text{new}}}) = \\min_{\\textcolor{#0ea5e9}{x_{\\text{near}}} \\in \\textcolor{#f59e0b}{X_{\\text{near}}}} \\{c(\\textcolor{#0ea5e9}{x_{\\text{near}}}) + \\|\\textcolor{#0ea5e9}{x_{\\text{near}}} - \\textcolor{#0ea5e9}{x_{\\text{new}}}\\|\\}, \\quad \\textcolor{#10b981}{\\text{Cost}(\\pi^*)} \\to c^*',
    pseudocode: [
      '1. Sample: x_rand ~ Uniform(X_free) with goalBias probability toward x_goal',
      '2. Nearest: Find x_nearest in tree minimizing Euclidean distance to x_rand',
      '3. Steer: Extend from x_nearest toward x_rand by stepSize to obtain x_new',
      '4. Collision Test: Verify straight-line edge (x_nearest, x_new) is obstacle-free',
      '5. Choose Parent: Search X_near within r_rewire for parent with lowest total cost',
      '6. Insert Node: Add x_new into tree connected to best parent',
      '7. Rewire Tree: For each x_near, rewire if cost(x_new) + dist < cost(x_near)'
    ]
  },

  createInitialState(): RRTStarState {
    const start = { x: -220, y: 130 };
    const goal = { x: 210, y: -120, radius: 28 };

    return {
      timeStep: 0,
      nodes: [
        {
          id: 0,
          x: start.x,
          y: start.y,
          parentId: null,
          cost: 0
        }
      ],
      start,
      goal,
      obstacles: DEFAULT_OBSTACLES.map((o) => ({ ...o })),
      lastSample: null,
      lastNewNodeId: null,
      candidateNearIds: [],
      bestGoalNodeId: null,
      path: [],
      pathCost: Infinity,
      goalReached: false,
      rewireCount: 0
    };
  },

  step(currentState: RRTStarState, params: RRTStarParams) {
    if (currentState.nodes.length >= params.maxNodes) {
      return {
        nextState: currentState,
        isComplete: true,
        activePseudocodeLine: 6
      };
    }

    const nextStep = currentState.timeStep + 1;
    const isStar = params.algorithm === 'rrt-star';

    let sampleX: number;
    let sampleY: number;

    if (Math.random() < params.goalBias) {
      sampleX = currentState.goal.x;
      sampleY = currentState.goal.y;
    } else {
      sampleX = (Math.random() * 2 - 1) * (HALF_W - 20);
      sampleY = (Math.random() * 2 - 1) * (HALF_H - 20);
    }

    let nearestNode = currentState.nodes[0];
    let minDist = dist(nearestNode.x, nearestNode.y, sampleX, sampleY);

    for (let i = 1; i < currentState.nodes.length; i++) {
      const n = currentState.nodes[i];
      const d = dist(n.x, n.y, sampleX, sampleY);
      if (d < minDist) {
        minDist = d;
        nearestNode = n;
      }
    }

    let newX = sampleX;
    let newY = sampleY;

    if (minDist > params.stepSize) {
      const angle = Math.atan2(sampleY - nearestNode.y, sampleX - nearestNode.x);
      newX = nearestNode.x + Math.cos(angle) * params.stepSize;
      newY = nearestNode.y + Math.sin(angle) * params.stepSize;
    }

    newX = Math.max(-HALF_W + 15, Math.min(HALF_W - 15, newX));
    newY = Math.max(-HALF_H + 15, Math.min(HALF_H - 15, newY));

    if (isSegmentInCollision(nearestNode.x, nearestNode.y, newX, newY, currentState.obstacles)) {
      return {
        nextState: {
          ...currentState,
          timeStep: nextStep,
          lastSample: { x: sampleX, y: sampleY },
          candidateNearIds: []
        },
        isComplete: false,
        activePseudocodeLine: 3
      };
    }

    let bestParent = nearestNode;
    let minCost = nearestNode.cost + dist(nearestNode.x, nearestNode.y, newX, newY);
    const nearNodeIds: number[] = [];

    if (isStar) {
      for (const n of currentState.nodes) {
        const d = dist(n.x, n.y, newX, newY);
        if (d <= params.rewireRadius) {
          nearNodeIds.push(n.id);
          const candidateCost = n.cost + d;
          if (
            candidateCost < minCost &&
            !isSegmentInCollision(n.x, n.y, newX, newY, currentState.obstacles)
          ) {
            bestParent = n;
            minCost = candidateCost;
          }
        }
      }
    }

    const newNodeId = currentState.nodes.length;
    const newNode: RRTNode = {
      id: newNodeId,
      x: newX,
      y: newY,
      parentId: bestParent.id,
      cost: minCost
    };

    const nextNodes: RRTNode[] = [...currentState.nodes.map((n) => ({ ...n })), newNode];
    let rewires = currentState.rewireCount;
    let didRewire = false;

    if (isStar) {
      for (let i = 0; i < nextNodes.length; i++) {
        const n = nextNodes[i];
        if (n.id === newNodeId || n.id === 0) continue;
        const d = dist(newNode.x, newNode.y, n.x, n.y);
        if (d <= params.rewireRadius) {
          const potentialCost = newNode.cost + d;
          if (
            potentialCost + 1e-4 < n.cost &&
            !isSegmentInCollision(newNode.x, newNode.y, n.x, n.y, currentState.obstacles)
          ) {
            const costReduction = n.cost - potentialCost;
            n.parentId = newNodeId;
            n.cost = potentialCost;
            propagateSubtreeCosts(nextNodes, n.id, costReduction);
            rewires++;
            didRewire = true;
          }
        }
      }
    }

    // Re-evaluate best path to goal across all goal-connected nodes
    let bestGoalNodeId: number | null = null;
    let bestPathCost = Infinity;

    for (const n of nextNodes) {
      const dToGoal = dist(n.x, n.y, currentState.goal.x, currentState.goal.y);
      if (dToGoal <= currentState.goal.radius + params.stepSize) {
        if (!isSegmentInCollision(n.x, n.y, currentState.goal.x, currentState.goal.y, currentState.obstacles)) {
          const totalCost = n.cost + dToGoal;
          if (totalCost < bestPathCost) {
            bestPathCost = totalCost;
            bestGoalNodeId = n.id;
          }
        }
      }
    }

    const goalReached = bestGoalNodeId !== null;
    const path =
      bestGoalNodeId !== null
        ? reconstructPath(nextNodes, bestGoalNodeId, currentState.goal)
        : [];

    return {
      nextState: {
        timeStep: nextStep,
        nodes: nextNodes,
        start: currentState.start,
        goal: currentState.goal,
        obstacles: currentState.obstacles,
        lastSample: { x: sampleX, y: sampleY },
        lastNewNodeId: newNodeId,
        candidateNearIds: nearNodeIds,
        bestGoalNodeId,
        path,
        pathCost: bestPathCost,
        goalReached,
        rewireCount: rewires
      },
      isComplete: nextNodes.length >= params.maxNodes,
      activePseudocodeLine: didRewire ? 6 : isStar ? 4 : 5
    };
  },

  onPointerInteraction(
    currentState: RRTStarState,
    event: CanvasPointerEvent
  ): RRTStarState {
    const wx = Math.max(-HALF_W + 30, Math.min(HALF_W - 30, event.worldX));
    const wy = Math.max(-HALF_H + 30, Math.min(HALF_H - 30, event.worldY));

    if (event.shiftKey) {
      return {
        ...currentState,
        goal: { ...currentState.goal, x: wx, y: wy },
        nodes: [{ id: 0, x: currentState.start.x, y: currentState.start.y, parentId: null, cost: 0 }],
        path: [],
        pathCost: Infinity,
        bestGoalNodeId: null,
        goalReached: false,
        rewireCount: 0
      };
    }

    const clickedObsIdx = currentState.obstacles.findIndex(
      (o) => dist(o.x, o.y, wx, wy) < o.radius + 15
    );

    if (clickedObsIdx >= 0) {
      const nextObs = currentState.obstacles.map((o, idx) =>
        idx === clickedObsIdx ? { ...o, x: wx, y: wy } : o
      );
      return {
        ...currentState,
        obstacles: nextObs,
        nodes: [{ id: 0, x: currentState.start.x, y: currentState.start.y, parentId: null, cost: 0 }],
        path: [],
        pathCost: Infinity,
        bestGoalNodeId: null,
        goalReached: false,
        rewireCount: 0
      };
    }

    return {
      ...currentState,
      start: { x: wx, y: wy },
      nodes: [{ id: 0, x: wx, y: wy, parentId: null, cost: 0 }],
      path: [],
      pathCost: Infinity,
      bestGoalNodeId: null,
      goalReached: false,
      rewireCount: 0
    };
  },

  getMetrics(currentState: RRTStarState, params: RRTStarParams): MetricReadout[] {
    const costText =
      currentState.pathCost < Infinity
        ? `${currentState.pathCost.toFixed(0)} px`
        : 'Searching...';

    return [
      {
        id: 'nodes',
        label: 'Tree Nodes',
        value: `${currentState.nodes.length} / ${params.maxNodes}`,
        color: 'sky'
      },
      {
        id: 'cost',
        label: 'Best Path Cost',
        value: costText,
        color: currentState.goalReached ? 'emerald' : 'amber'
      },
      {
        id: 'rewires',
        label: 'Rewired Edges',
        value: currentState.rewireCount,
        color: 'amber'
      },
      {
        id: 'status',
        label: 'Status',
        value: currentState.goalReached ? 'Goal Connected' : 'Exploring Free Space',
        color: currentState.goalReached ? 'emerald' : 'rose'
      }
    ];
  },

  render(ctx, currentState, params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
    ctx.beginPath();
    ctx.roundRect(-HALF_W, -HALF_H, ARENA_W, ARENA_H, 12);
    ctx.fill();
    ctx.stroke();

    for (const obs of currentState.obstacles) {
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.25)';
      ctx.fill();
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.radius * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.fill();
    }

    ctx.strokeStyle = 'rgba(14, 165, 233, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();

    const nodeMap = new Map<number, RRTNode>();
    for (const n of currentState.nodes) nodeMap.set(n.id, n);

    for (let i = 1; i < currentState.nodes.length; i++) {
      const n = currentState.nodes[i];
      if (n.parentId !== null) {
        const p = nodeMap.get(n.parentId);
        if (p) {
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(n.x, n.y);
        }
      }
    }
    ctx.stroke();

    if (currentState.lastSample) {
      ctx.beginPath();
      ctx.arc(currentState.lastSample.x, currentState.lastSample.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
    }

    if (currentState.lastNewNodeId !== null && params.algorithm === 'rrt-star') {
      const lastNode = nodeMap.get(currentState.lastNewNodeId);
      if (lastNode) {
        ctx.beginPath();
        ctx.arc(lastNode.x, lastNode.y, params.rewireRadius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (currentState.path.length > 1) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(currentState.path[0].x, currentState.path[0].y);
      for (let i = 1; i < currentState.path.length; i++) {
        ctx.lineTo(currentState.path[i].x, currentState.path[i].y);
      }
      ctx.stroke();

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(currentState.path[0].x, currentState.path[0].y);
      for (let i = 1; i < currentState.path.length; i++) {
        ctx.lineTo(currentState.path[i].x, currentState.path[i].y);
      }
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(currentState.goal.x, currentState.goal.y, currentState.goal.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = '#6ee7b7';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('GOAL', currentState.goal.x, currentState.goal.y);

    ctx.beginPath();
    ctx.arc(currentState.start.x, currentState.start.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = 'bold 10px sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('START', currentState.start.x, currentState.start.y + 12);

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Click/drag to move Start or Obstacle; Shift-click to move Goal', 0, HALF_H + 24);

    ctx.restore();
  }
};
