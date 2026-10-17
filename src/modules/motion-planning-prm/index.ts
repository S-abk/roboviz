// FILE: src/modules/motion-planning-prm/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export type Obstacle =
  | { id: number; type: 'circle'; x: number; y: number; radius: number }
  | { id: number; type: 'rect'; x: number; y: number; width: number; height: number };

export interface PRMNode {
  id: number;
  x: number;
  y: number;
  neighbors: number[];
}

export interface PRMEdge {
  from: number;
  to: number;
  cost: number;
}

export interface PRMState {
  stepCount: number;
  nodes: PRMNode[];
  edges: PRMEdge[];
  start: { x: number; y: number };
  goal: { x: number; y: number };
  obstacles: Obstacle[];
  path: { x: number; y: number }[];
  pathCost: number;
  goalReached: boolean;
  recentRejected: { x: number; y: number; age: number }[];
  recentTestedEdges: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    accepted: boolean;
    age: number;
  }[];
  startConnections: number[];
  goalConnections: number[];
  draggedTarget: { type: 'start' | 'goal' | 'obstacle'; id?: number } | null;
  activeLine: number;
}

export interface PRMParams {
  targetNodes: number;
  connectionRadius: number;
  maxNeighbors: number;
  obstacleInflation: number;
}

const ARENA_X_MIN = -280;
const ARENA_X_MAX = 280;
const ARENA_Y_MIN = -165;
const ARENA_Y_MAX = 165;

const DEFAULT_OBSTACLES: Obstacle[] = [
  { id: 0, type: 'rect', x: 0, y: -10, width: 70, height: 120 },
  { id: 1, type: 'circle', x: -115, y: -65, radius: 44 },
  { id: 2, type: 'rect', x: -130, y: 75, width: 80, height: 60 },
  { id: 3, type: 'circle', x: 115, y: 65, radius: 46 },
  { id: 4, type: 'rect', x: 125, y: -75, width: 85, height: 55 }
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

function lineIntersectsAABB(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): boolean {
  if (x1 >= minX && x1 <= maxX && y1 >= minY && y1 <= maxY) return true;
  if (x2 >= minX && x2 <= maxX && y2 >= minY && y2 <= maxY) return true;

  let t0 = 0.0;
  let t1 = 1.0;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1 - minX, maxX - x1, y1 - minY, maxY - y1];

  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return false;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) {
        if (t > t1) return false;
        if (t > t0) t0 = t;
      } else {
        if (t < t0) return false;
        if (t < t1) t1 = t;
      }
    }
  }

  return t0 <= t1;
}

function isPointInCollision(
  x: number,
  y: number,
  obstacles: Obstacle[],
  inflation: number
): boolean {
  for (const obs of obstacles) {
    if (obs.type === 'circle') {
      if (dist(x, y, obs.x, obs.y) <= obs.radius + inflation) return true;
    } else {
      const halfW = obs.width / 2 + inflation;
      const halfH = obs.height / 2 + inflation;
      if (
        x >= obs.x - halfW &&
        x <= obs.x + halfW &&
        y >= obs.y - halfH &&
        y <= obs.y + halfH
      ) {
        return true;
      }
    }
  }
  return false;
}

function isSegmentInCollision(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  obstacles: Obstacle[],
  inflation: number
): boolean {
  for (const obs of obstacles) {
    if (obs.type === 'circle') {
      if (lineIntersectsCircle(x1, y1, x2, y2, obs.x, obs.y, obs.radius + inflation)) {
        return true;
      }
    } else {
      const halfW = obs.width / 2 + inflation;
      const halfH = obs.height / 2 + inflation;
      if (
        lineIntersectsAABB(
          x1,
          y1,
          x2,
          y2,
          obs.x - halfW,
          obs.x + halfW,
          obs.y - halfH,
          obs.y + halfH
        )
      ) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Executes multi-query A* shortest path search over the PRM roadmap.
 */
function solvePRMQuery(
  start: { x: number; y: number },
  goal: { x: number; y: number },
  nodes: PRMNode[],
  obstacles: Obstacle[],
  params: PRMParams
): {
  path: { x: number; y: number }[];
  pathCost: number;
  goalReached: boolean;
  startConnections: number[];
  goalConnections: number[];
} {
  // Check if direct straight-line connection is collision-free
  if (!isSegmentInCollision(start.x, start.y, goal.x, goal.y, obstacles, params.obstacleInflation)) {
    const directDist = dist(start.x, start.y, goal.x, goal.y);
    return {
      path: [
        { x: start.x, y: start.y },
        { x: goal.x, y: goal.y }
      ],
      pathCost: directDist,
      goalReached: true,
      startConnections: [],
      goalConnections: []
    };
  }

  if (nodes.length === 0) {
    return {
      path: [],
      pathCost: 0,
      goalReached: false,
      startConnections: [],
      goalConnections: []
    };
  }

  // Connect Start and Goal to k-nearest visible roadmap milestones
  const queryRadius = params.connectionRadius * 1.5;
  const startCandidates: { id: number; d: number }[] = [];
  const goalCandidates: { id: number; d: number }[] = [];

  for (const node of nodes) {
    if (isPointInCollision(node.x, node.y, obstacles, params.obstacleInflation)) continue;
    const dStart = dist(start.x, start.y, node.x, node.y);
    if (dStart <= queryRadius) {
      if (
        !isSegmentInCollision(
          start.x,
          start.y,
          node.x,
          node.y,
          obstacles,
          params.obstacleInflation
        )
      ) {
        startCandidates.push({ id: node.id, d: dStart });
      }
    }

    const dGoal = dist(goal.x, goal.y, node.x, node.y);
    if (dGoal <= queryRadius) {
      if (
        !isSegmentInCollision(
          goal.x,
          goal.y,
          node.x,
          node.y,
          obstacles,
          params.obstacleInflation
        )
      ) {
        goalCandidates.push({ id: node.id, d: dGoal });
      }
    }
  }

  startCandidates.sort((a, b) => a.d - b.d);
  goalCandidates.sort((a, b) => a.d - b.d);

  const startConnections = startCandidates.slice(0, params.maxNeighbors).map((c) => c.id);
  const goalConnections = goalCandidates.slice(0, params.maxNeighbors).map((c) => c.id);

  if (startConnections.length === 0 || goalConnections.length === 0) {
    return {
      path: [],
      pathCost: 0,
      goalReached: false,
      startConnections,
      goalConnections
    };
  }

  // Virtual IDs: Start = -1, Goal = -2
  const START_ID = -1;
  const GOAL_ID = -2;

  const gScore = new Map<number, number>();
  const fScore = new Map<number, number>();
  const cameFrom = new Map<number, number>();
  const openSet = new Set<number>();

  gScore.set(START_ID, 0);
  fScore.set(START_ID, dist(start.x, start.y, goal.x, goal.y));
  openSet.add(START_ID);

  function getNodePos(id: number): { x: number; y: number } {
    if (id === START_ID) return start;
    if (id === GOAL_ID) return goal;
    return nodes[id];
  }

  function getNeighbors(id: number): { neighborId: number; weight: number }[] {
    const result: { neighborId: number; weight: number }[] = [];
    if (id === START_ID) {
      for (const nid of startConnections) {
        result.push({ neighborId: nid, weight: dist(start.x, start.y, nodes[nid].x, nodes[nid].y) });
      }
    } else {
      const u = nodes[id];
      for (const nid of u.neighbors) {
        const v = nodes[nid];
        if (
          !isPointInCollision(v.x, v.y, obstacles, params.obstacleInflation) &&
          !isSegmentInCollision(u.x, u.y, v.x, v.y, obstacles, params.obstacleInflation)
        ) {
          result.push({ neighborId: nid, weight: dist(u.x, u.y, v.x, v.y) });
        }
      }
      if (goalConnections.includes(id)) {
        result.push({ neighborId: GOAL_ID, weight: dist(u.x, u.y, goal.x, goal.y) });
      }
    }
    return result;
  }

  while (openSet.size > 0) {
    let current = -999;
    let minF = Infinity;
    for (const id of openSet) {
      const f = fScore.get(id) ?? Infinity;
      if (f < minF) {
        minF = f;
        current = id;
      }
    }

    if (current === GOAL_ID) {
      // Reconstruct path
      const path: { x: number; y: number }[] = [];
      let curr = GOAL_ID;
      while (curr !== undefined) {
        path.unshift(getNodePos(curr));
        const prev = cameFrom.get(curr);
        if (curr === START_ID || prev === undefined) break;
        curr = prev;
      }
      const totalCost = gScore.get(GOAL_ID) ?? 0;
      return {
        path,
        pathCost: totalCost,
        goalReached: true,
        startConnections,
        goalConnections
      };
    }

    openSet.delete(current);
    const currentG = gScore.get(current) ?? Infinity;
    const neighbors = getNeighbors(current);

    for (const { neighborId, weight } of neighbors) {
      const tentativeG = currentG + weight;
      const neighborG = gScore.get(neighborId) ?? Infinity;

      if (tentativeG < neighborG) {
        cameFrom.set(neighborId, current);
        gScore.set(neighborId, tentativeG);
        const neighborPos = getNodePos(neighborId);
        const h = dist(neighborPos.x, neighborPos.y, goal.x, goal.y);
        fScore.set(neighborId, tentativeG + h);
        openSet.add(neighborId);
      }
    }
  }

  return {
    path: [],
    pathCost: 0,
    goalReached: false,
    startConnections,
    goalConnections
  };
}

export const PRMPlannerPlugin: AlgorithmPlugin<PRMState, PRMParams> = {
  id: 'motion-planning-prm',
  title: 'Probabilistic Roadmap (PRM) Planner',
  shortDescription:
    'Multi-query sampling-based roadmap construction connecting collision-free milestones via k-nearest visibility edges and shortest-path graph search.',
  category: 'motion-planning',

  controls: [
    {
      id: 'targetNodes',
      label: 'Target Milestones (N)',
      type: 'slider',
      min: 40,
      max: 200,
      step: 10,
      defaultValue: 100,
      description: 'Total milestone configurations N to sample in free space.'
    },
    {
      id: 'connectionRadius',
      label: 'Connection Radius (R)',
      type: 'slider',
      min: 50,
      max: 160,
      step: 10,
      defaultValue: 95,
      unit: 'px',
      description: 'Maximum radius R for connecting neighbouring roadmap milestones.'
    },
    {
      id: 'maxNeighbors',
      label: 'Max Neighbors (k)',
      type: 'slider',
      min: 3,
      max: 12,
      step: 1,
      defaultValue: 6,
      description: 'Maximum k-nearest neighbors connected per milestone.'
    },
    {
      id: 'obstacleInflation',
      label: 'Obstacle Inflation Margin',
      type: 'slider',
      min: 0,
      max: 18,
      step: 2,
      defaultValue: 6,
      unit: 'px',
      description: 'Safety clearance margin around obstacles during line-of-sight checks.'
    }
  ],

  documentation: {
    intuition:
      'The Probabilistic Roadmap (PRM) planner decouples path planning into two distinct phases: an offline Learning Phase that constructs a persistent roadmap graph G = (V, E) throughout the free configuration space C_free, and an online Query Phase that rapidly solves point-to-point queries. During construction, PRM uniformly samples configurations across the environment, discarding those colliding with obstacles. Each valid milestone connects to its k-nearest neighbors within connection radius R if the local straight-line visibility segment is collision-free. Once constructed, any start and goal configuration can be routed almost instantaneously via A* or Dijkstra graph search. This multi-query efficiency makes PRM ideal for stationary environments where a robot frequently navigates between changing pickup and delivery stations.',
    governingEquationLatex:
      '\\textcolor{#38bdf8}{G = (V, E)}, \\quad e_{ij} \\in \\textcolor{#38bdf8}{E} \\iff \\|\\mathbf{q}_i - \\mathbf{q}_j\\| \\le R \\;\\wedge\\; \\overline{\\mathbf{q}_i\\mathbf{q}_j} \\cap \\textcolor{#f43f5e}{\\mathcal{C}_{\\text{obs}}} = \\emptyset, \\quad \\textcolor{#10b981}{\\pi^*} = \\text{A}^*(\\textcolor{#38bdf8}{G}, \\mathbf{q}_{\\text{start}}, \\textcolor{#10b981}{\\mathbf{q}_{\\text{goal}}})',
    pseudocode: [
      '1. Sample Milestone: q_rand ~ Uniform(C_free); reject if q_rand ∈ C_obs',
      '2. Neighbor Search: N_q = { v ∈ V | ||v - q_rand|| ≤ R, rank ≤ k }',
      '3. Local Collision Check: For each v ∈ N_q, verify segment (q_rand, v) ∩ C_obs = ∅',
      '4. Graph Augmentation: Add milestone q_rand to V; add valid edges (q_rand, v) to E',
      '5. Query Connect: Connect q_start and q_goal to their nearest visible roadmap milestones',
      '6. Shortest Path Search: Run A* / Dijkstra on G to find optimal milestone sequence π*',
      '7. Instant Multi-Query: On query target change, re-route immediately using precomputed G'
    ]
  },

  createInitialState(_params: PRMParams): PRMState {
    const start = { x: -220, y: 100 };
    const goal = { x: 220, y: -100 };
    const obstacles = DEFAULT_OBSTACLES.map((o) => ({ ...o }));

    return {
      stepCount: 0,
      nodes: [],
      edges: [],
      start,
      goal,
      obstacles,
      path: [],
      pathCost: 0,
      goalReached: false,
      recentRejected: [],
      recentTestedEdges: [],
      startConnections: [],
      goalConnections: [],
      draggedTarget: null,
      activeLine: 0
    };
  },

  step(currentState: PRMState, params: PRMParams) {
    const step = currentState.stepCount + 1;
    const nodes = currentState.nodes.map((n) => ({ ...n, neighbors: [...n.neighbors] }));
    const edges = currentState.edges.map((e) => ({ ...e }));
    const obstacles = currentState.obstacles;

    // Age existing visual debug items
    const recentRejected = currentState.recentRejected
      .map((r) => ({ ...r, age: r.age + 1 }))
      .filter((r) => r.age <= 2);

    const recentTestedEdges = currentState.recentTestedEdges
      .map((t) => ({ ...t, age: t.age + 1 }))
      .filter((t) => t.age <= 2);

    let activeLine = 0;
    const isComplete = nodes.length >= params.targetNodes;

    if (!isComplete) {
      // Phase 1: Sample 2-3 candidate milestones per step
      const samplesPerStep = 2;

      for (let s = 0; s < samplesPerStep; s++) {
        if (nodes.length >= params.targetNodes) break;

        const candX = ARENA_X_MIN + Math.random() * (ARENA_X_MAX - ARENA_X_MIN);
        const candY = ARENA_Y_MIN + Math.random() * (ARENA_Y_MAX - ARENA_Y_MIN);

        // Collision check sample configuration
        if (isPointInCollision(candX, candY, obstacles, params.obstacleInflation)) {
          recentRejected.push({ x: candX, y: candY, age: 0 });
          activeLine = 0;
          continue;
        }

        const newId = nodes.length;
        const newNeighbors: number[] = [];

        // Find candidate neighbors within connectionRadius
        const neighborCandidates: { id: number; d: number }[] = [];
        for (const existingNode of nodes) {
          const d = dist(candX, candY, existingNode.x, existingNode.y);
          if (d <= params.connectionRadius) {
            neighborCandidates.push({ id: existingNode.id, d });
          }
        }

        neighborCandidates.sort((a, b) => a.d - b.d);
        const topNeighbors = neighborCandidates.slice(0, params.maxNeighbors);

        // Check straight line collision for each candidate neighbor
        for (const { id: otherId, d } of topNeighbors) {
          const other = nodes[otherId];
          const inCollision = isSegmentInCollision(
            candX,
            candY,
            other.x,
            other.y,
            obstacles,
            params.obstacleInflation
          );

          recentTestedEdges.push({
            x1: candX,
            y1: candY,
            x2: other.x,
            y2: other.y,
            accepted: !inCollision,
            age: 0
          });

          if (!inCollision) {
            newNeighbors.push(otherId);
            other.neighbors.push(newId);
            edges.push({ from: newId, to: otherId, cost: d });
            activeLine = 3;
          }
        }

        nodes.push({
          id: newId,
          x: candX,
          y: candY,
          neighbors: newNeighbors
        });
      }
    }

    // Phase 2: Solve multi-query shortest path
    const queryResult = solvePRMQuery(
      currentState.start,
      currentState.goal,
      nodes,
      obstacles,
      params
    );

    if (queryResult.goalReached) {
      activeLine = isComplete ? 6 : 5;
    }

    return {
      nextState: {
        stepCount: step,
        nodes,
        edges,
        start: currentState.start,
        goal: currentState.goal,
        obstacles,
        path: queryResult.path,
        pathCost: queryResult.pathCost,
        goalReached: queryResult.goalReached,
        recentRejected,
        recentTestedEdges,
        startConnections: queryResult.startConnections,
        goalConnections: queryResult.goalConnections,
        draggedTarget: currentState.draggedTarget,
        activeLine
      },
      isComplete: nodes.length >= params.targetNodes,
      activePseudocodeLine: activeLine
    };
  },

  onPointerInteraction(
    currentState: PRMState,
    event: CanvasPointerEvent,
    params: PRMParams
  ): PRMState {
    const start = { ...currentState.start };
    const goal = { ...currentState.goal };
    const obstacles = currentState.obstacles.map((o) => ({ ...o }));
    let dragged = currentState.draggedTarget;

    if (!event.isDragging) {
      const dStart = dist(event.worldX, event.worldY, start.x, start.y);
      const dGoal = dist(event.worldX, event.worldY, goal.x, goal.y);

      // Check obstacle hit
      let hitObsId: number | null = null;
      for (const obs of obstacles) {
        if (obs.type === 'circle') {
          if (dist(event.worldX, event.worldY, obs.x, obs.y) <= obs.radius + 6) {
            hitObsId = obs.id;
            break;
          }
        } else {
          const hw = obs.width / 2 + 6;
          const hh = obs.height / 2 + 6;
          if (
            event.worldX >= obs.x - hw &&
            event.worldX <= obs.x + hw &&
            event.worldY >= obs.y - hh &&
            event.worldY <= obs.y + hh
          ) {
            hitObsId = obs.id;
            break;
          }
        }
      }

      if (hitObsId !== null) {
        dragged = { type: 'obstacle', id: hitObsId };
      } else if (dStart < 26) {
        dragged = { type: 'start' };
      } else if (dGoal < 26) {
        dragged = { type: 'goal' };
      } else {
        dragged = dStart < dGoal ? { type: 'start' } : { type: 'goal' };
      }
    }

    if (dragged) {
      if (dragged.type === 'start') {
        start.x = Math.max(ARENA_X_MIN + 10, Math.min(ARENA_X_MAX - 10, event.worldX));
        start.y = Math.max(ARENA_Y_MIN + 10, Math.min(ARENA_Y_MAX - 10, event.worldY));
      } else if (dragged.type === 'goal') {
        goal.x = Math.max(ARENA_X_MIN + 10, Math.min(ARENA_X_MAX - 10, event.worldX));
        goal.y = Math.max(ARENA_Y_MIN + 10, Math.min(ARENA_Y_MAX - 10, event.worldY));
      } else if (dragged.type === 'obstacle' && dragged.id !== undefined) {
        const obs = obstacles.find((o) => o.id === dragged!.id);
        if (obs) {
          obs.x = Math.max(ARENA_X_MIN + 30, Math.min(ARENA_X_MAX - 30, event.worldX));
          obs.y = Math.max(ARENA_Y_MIN + 30, Math.min(ARENA_Y_MAX - 30, event.worldY));
        }
      }
    }

    // Instantly solve shortest path query for real-time responsiveness
    const queryResult = solvePRMQuery(
      start,
      goal,
      currentState.nodes,
      obstacles,
      params
    );

    return {
      ...currentState,
      start,
      goal,
      obstacles,
      path: queryResult.path,
      pathCost: queryResult.pathCost,
      goalReached: queryResult.goalReached,
      startConnections: queryResult.startConnections,
      goalConnections: queryResult.goalConnections,
      draggedTarget: dragged,
      activeLine: 6
    };
  },

  getMetrics(currentState: PRMState, params: PRMParams): MetricReadout[] {
    const isFinishedSampling = currentState.nodes.length >= params.targetNodes;
    const pathColor =
      currentState.goalReached
        ? 'emerald'
        : isFinishedSampling
        ? 'rose'
        : 'amber';

    return [
      {
        id: 'milestones',
        label: 'Milestones (V)',
        value: `${currentState.nodes.length} / ${params.targetNodes}`,
        color: isFinishedSampling ? 'emerald' : 'sky'
      },
      {
        id: 'edges',
        label: 'Roadmap Edges (E)',
        value: `${currentState.edges.length}`,
        color: 'sky'
      },
      {
        id: 'pathCost',
        label: 'Shortest Path Length',
        value: currentState.goalReached ? currentState.pathCost.toFixed(1) : 'Disconnected',
        unit: currentState.goalReached ? ' px' : undefined,
        color: pathColor
      },
      {
        id: 'queryStatus',
        label: 'Multi-Query Status',
        value: currentState.goalReached
          ? 'Route Connected (Instant Query)'
          : isFinishedSampling
          ? 'No Path Found'
          : 'Building Roadmap...',
        color: currentState.goalReached ? 'emerald' : isFinishedSampling ? 'rose' : 'amber'
      }
    ];
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: PRMState,
    params: PRMParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const cx = viewport.width / 2;
    const cy = viewport.height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Workspace Boundary
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      ARENA_X_MIN,
      ARENA_Y_MIN,
      ARENA_X_MAX - ARENA_X_MIN,
      ARENA_Y_MAX - ARENA_Y_MIN
    );

    // 2. Obstacles and Inflation Margins (Rose #f43f5e)
    for (const obs of state.obstacles) {
      const inflation = params.obstacleInflation;

      if (obs.type === 'circle') {
        // Inflation margin halo
        if (inflation > 0) {
          ctx.beginPath();
          ctx.arc(obs.x, obs.y, obs.radius + inflation, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(244, 63, 94, 0.1)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Obstacle body
        ctx.beginPath();
        ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2.0;
        ctx.stroke();

        ctx.font = '600 10px Inter, sans-serif';
        ctx.fillStyle = '#fda4af';
        ctx.textAlign = 'center';
        ctx.fillText(`C_obs ${obs.id + 1}`, obs.x, obs.y + 3);
      } else {
        const halfW = obs.width / 2;
        const halfH = obs.height / 2;

        // Inflation margin halo
        if (inflation > 0) {
          ctx.beginPath();
          ctx.roundRect(
            obs.x - halfW - inflation,
            obs.y - halfH - inflation,
            obs.width + 2 * inflation,
            obs.height + 2 * inflation,
            6
          );
          ctx.fillStyle = 'rgba(244, 63, 94, 0.1)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Obstacle body
        ctx.beginPath();
        ctx.roundRect(obs.x - halfW, obs.y - halfH, obs.width, obs.height, 4);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2.0;
        ctx.stroke();

        ctx.font = '600 10px Inter, sans-serif';
        ctx.fillStyle = '#fda4af';
        ctx.textAlign = 'center';
        ctx.fillText(`C_obs ${obs.id + 1}`, obs.x, obs.y + 3);
      }
    }

    // 3. Recently Rejected Samples in C_obs (Rose #f43f5e 'X')
    for (const r of state.recentRejected) {
      const alpha = Math.max(0.2, 1.0 - r.age * 0.35);
      ctx.strokeStyle = `rgba(244, 63, 94, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(r.x - 4, r.y - 4);
      ctx.lineTo(r.x + 4, r.y + 4);
      ctx.moveTo(r.x + 4, r.y - 4);
      ctx.lineTo(r.x - 4, r.y + 4);
      ctx.stroke();
    }

    // 4. Tested Edges during Construction (Amber #f59e0b)
    for (const edge of state.recentTestedEdges) {
      const alpha = Math.max(0.3, 1.0 - edge.age * 0.35);
      ctx.beginPath();
      ctx.moveTo(edge.x1, edge.y1);
      ctx.lineTo(edge.x2, edge.y2);
      ctx.strokeStyle = edge.accepted
        ? `rgba(245, 158, 11, ${alpha})`
        : `rgba(244, 63, 94, ${alpha * 0.7})`;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([2, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 5. Accepted PRM Roadmap Edges (Sky Blue #38bdf8)
    ctx.beginPath();
    for (const edge of state.edges) {
      if (edge.from < state.nodes.length && edge.to < state.nodes.length) {
        const u = state.nodes[edge.from];
        const v = state.nodes[edge.to];
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);
      }
    }
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // 6. PRM Roadmap Milestones (V)
    for (const node of state.nodes) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
    }

    // 7. Connections to Start and Goal (Dashed Lines)
    ctx.beginPath();
    for (const nid of state.startConnections) {
      if (nid < state.nodes.length) {
        ctx.moveTo(state.start.x, state.start.y);
        ctx.lineTo(state.nodes[nid].x, state.nodes[nid].y);
      }
    }
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    for (const nid of state.goalConnections) {
      if (nid < state.nodes.length) {
        ctx.moveTo(state.goal.x, state.goal.y);
        ctx.lineTo(state.nodes[nid].x, state.nodes[nid].y);
      }
    }
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 8. Shortest Multi-Query Path (Glowing Emerald #10b981)
    if (state.goalReached && state.path.length > 1) {
      ctx.save();
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 12;

      // Glow layer
      ctx.beginPath();
      ctx.moveTo(state.path[0].x, state.path[0].y);
      for (let i = 1; i < state.path.length; i++) {
        ctx.lineTo(state.path[i].x, state.path[i].y);
      }
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 6;
      ctx.stroke();

      // Sharp core line
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.8;
      ctx.stroke();

      // Waypoint nodes along shortest path
      for (let i = 1; i < state.path.length - 1; i++) {
        ctx.beginPath();
        ctx.arc(state.path[i].x, state.path[i].y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = '#10b981';
        ctx.fill();
        ctx.strokeStyle = '#a7f3d0';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.restore();
    }

    // 9. Start Node (Sky Blue #38bdf8)
    ctx.save();
    ctx.beginPath();
    ctx.arc(state.start.x, state.start.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#0284c7';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillStyle = '#7dd3fc';
    ctx.textAlign = 'center';
    ctx.fillText('Start q_start', state.start.x, state.start.y - 12);
    ctx.restore();

    // 10. Goal Node (Emerald #10b981)
    ctx.save();
    ctx.beginPath();
    ctx.arc(state.goal.x, state.goal.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#059669';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillStyle = '#a7f3d0';
    ctx.textAlign = 'center';
    ctx.fillText('Goal q_goal', state.goal.x, state.goal.y - 12);
    ctx.restore();

    // 11. Status Banner
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    if (state.goalReached) {
      ctx.fillStyle = '#10b981';
      ctx.textAlign = 'center';
      ctx.fillText(
        `✓ Shortest Path Found (${state.pathCost.toFixed(1)} px) — Drag Start, Goal, or Obstacles for instant multi-query re-routing!`,
        0,
        -146
      );
    } else if (state.nodes.length >= params.targetNodes) {
      ctx.fillStyle = '#f43f5e';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Roadmap Complete (${state.nodes.length} nodes) — Disconnected: increase connection radius R or drag endpoints`,
        0,
        -146
      );
    } else {
      ctx.fillStyle = '#38bdf8';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Constructing Roadmap: Sampling free space (${state.nodes.length}/${params.targetNodes} Milestones)`,
        0,
        -146
      );
    }

    // Helper Footer
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(
      'Sky Blue: Milestones & Edges | Rose: C_obs Obstacles & Clearance | Emerald: Shortest Query Path π* | Drag Start / Goal / Obstacles',
      0,
      156
    );

    ctx.restore();
  }
};

export default PRMPlannerPlugin;
