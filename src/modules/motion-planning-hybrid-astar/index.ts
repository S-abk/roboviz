// FILE: src/modules/motion-planning-hybrid-astar/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface CarPose {
  x: number;
  y: number;
  theta: number;
}

export interface Obstacle {
  id: number;
  x: number;
  y: number;
  radius: number;
  label?: string;
}

export interface ArcPrimitive {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  points: { x: number; y: number }[];
  gear: number; // 1 = forward (Sky Blue), -1 = reverse (Amber)
  steer: number; // -1 = left, 0 = straight, +1 = right
}

export interface HybridNode {
  id: number;
  x: number;
  y: number;
  theta: number;
  g: number;
  f: number;
  h: number;
  parentId: number | null;
  steer: number;
  gear: number;
  arcPoints: { x: number; y: number }[];
}

export interface HybridAStarState {
  start: CarPose;
  goal: CarPose;
  obstacles: Obstacle[];
  nodes: HybridNode[];
  openSet: number[]; // Binary min-heap of node IDs
  visited: Record<string, number>;
  exploredArcs: ArcPrimitive[];
  path: CarPose[];
  pathCost: number;
  foundPath: boolean;
  isComplete: boolean;
  nodesExplored: number;
  iteration: number;
  dragTarget: 'goal' | number | null;
  activeLine: number;
}

export interface HybridAStarParams {
  minTurnRadius: number;
  reversePenalty: number;
  heuristicWeight: number;
  allowReverse: boolean;
}

const CAR_LENGTH = 28;
const CAR_WIDTH = 15;
const CAR_AXLE_OFFSET = 7.5;
const COLLISION_RADIUS = 8.5;
const STEP_ARC_LEN = 20;
const MICRO_STEPS = 4;
const GRID_CELL_SIZE = 18;
const ANGLE_BINS = 16;
const EXPANSIONS_PER_STEP = 6;

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function normalizeAnglePositive(a: number): number {
  const norm = a % (2 * Math.PI);
  return norm >= 0 ? norm : norm + 2 * Math.PI;
}

function getGridKey(x: number, y: number, theta: number): string {
  const gx = Math.round(x / GRID_CELL_SIZE);
  const gy = Math.round(y / GRID_CELL_SIZE);
  const gth = Math.floor(normalizeAnglePositive(theta) / (Math.PI / 8)) % ANGLE_BINS;
  return `${gx},${gy},${gth}`;
}

function computeHeuristic(
  x: number,
  y: number,
  theta: number,
  gx: number,
  gy: number,
  gtheta: number,
  turnRadius: number
): number {
  const euclidDist = Math.hypot(gx - x, gy - y);
  const bearing = Math.atan2(gy - y, gx - x);
  const dTheta1 = Math.abs(normalizeAngle(theta - bearing));
  const dTheta2 = Math.abs(normalizeAngle(gtheta - theta));

  // Non-holonomic lower bound accounting for orientation alignment
  return euclidDist + turnRadius * (dTheta1 * 0.4 + dTheta2 * 0.35);
}

function checkVehicleCollision(
  x: number,
  y: number,
  theta: number,
  obstacles: Obstacle[]
): boolean {
  // Boundary constraints
  if (x < -275 || x > 275 || y < -165 || y > 165) {
    return true;
  }

  // 2-circle vehicle footprint check (front axle + rear axle)
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  const fx = x + CAR_AXLE_OFFSET * cosT;
  const fy = y + CAR_AXLE_OFFSET * sinT;
  const rx = x - CAR_AXLE_OFFSET * cosT;
  const ry = y - CAR_AXLE_OFFSET * sinT;

  for (const obs of obstacles) {
    const safeDist = obs.radius + COLLISION_RADIUS;
    if (Math.hypot(fx - obs.x, fy - obs.y) <= safeDist) return true;
    if (Math.hypot(rx - obs.x, ry - obs.y) <= safeDist) return true;
  }

  return false;
}

// Binary Min-Heap operations for priority queue
function heapPush(openSet: number[], nodeId: number, nodes: HybridNode[]): void {
  openSet.push(nodeId);
  let idx = openSet.length - 1;
  while (idx > 0) {
    const parentIdx = Math.floor((idx - 1) / 2);
    if (nodes[openSet[idx]].f < nodes[openSet[parentIdx]].f) {
      const temp = openSet[idx];
      openSet[idx] = openSet[parentIdx];
      openSet[parentIdx] = temp;
      idx = parentIdx;
    } else {
      break;
    }
  }
}

function heapPop(openSet: number[], nodes: HybridNode[]): number | null {
  if (openSet.length === 0) return null;
  const top = openSet[0];
  const last = openSet.pop()!;
  if (openSet.length > 0) {
    openSet[0] = last;
    let idx = 0;
    const len = openSet.length;
    while (true) {
      const left = 2 * idx + 1;
      const right = 2 * idx + 2;
      let smallest = idx;

      if (left < len && nodes[openSet[left]].f < nodes[openSet[smallest]].f) {
        smallest = left;
      }
      if (right < len && nodes[openSet[right]].f < nodes[openSet[smallest]].f) {
        smallest = right;
      }
      if (smallest !== idx) {
        const temp = openSet[idx];
        openSet[idx] = openSet[smallest];
        openSet[smallest] = temp;
        idx = smallest;
      } else {
        break;
      }
    }
  }
  return top;
}

function createFreshSearch(
  start: CarPose,
  goal: CarPose,
  obstacles: Obstacle[],
  params: HybridAStarParams
): HybridAStarState {
  const h0 = computeHeuristic(
    start.x,
    start.y,
    start.theta,
    goal.x,
    goal.y,
    goal.theta,
    params.minTurnRadius
  );
  const startNode: HybridNode = {
    id: 0,
    x: start.x,
    y: start.y,
    theta: start.theta,
    g: 0,
    h: h0,
    f: params.heuristicWeight * h0,
    parentId: null,
    steer: 0,
    gear: 1,
    arcPoints: [{ x: start.x, y: start.y }]
  };

  const visited: Record<string, number> = {};
  visited[getGridKey(start.x, start.y, start.theta)] = 0;

  const openSet: number[] = [];
  heapPush(openSet, 0, [startNode]);

  return {
    start,
    goal,
    obstacles,
    nodes: [startNode],
    openSet,
    visited,
    exploredArcs: [],
    path: [],
    pathCost: 0,
    foundPath: false,
    isComplete: false,
    nodesExplored: 0,
    iteration: 0,
    dragTarget: null,
    activeLine: 0
  };
}

export const HybridAStarPlugin: AlgorithmPlugin<HybridAStarState, HybridAStarParams> = {
  id: 'motion-planning-hybrid-astar',
  title: 'Hybrid A* Non-Holonomic Car Planner',
  shortDescription:
    'Kinematic motion-primitive search in continuous (x, y, θ) space respecting minimum turning radius and forward/reverse gear costs.',
  category: 'motion-planning',

  controls: [
    {
      id: 'minTurnRadius',
      label: 'Min Turning Radius (R_min)',
      type: 'slider',
      min: 28,
      max: 90,
      step: 4,
      defaultValue: 48,
      unit: 'px',
      description: 'Minimum Ackermann turning radius R_min = L / tan(δ_max).'
    },
    {
      id: 'reversePenalty',
      label: 'Reverse Gear Penalty',
      type: 'slider',
      min: 1.0,
      max: 4.0,
      step: 0.2,
      defaultValue: 1.6,
      description: 'Cost multiplier applied to reverse-gear motion primitives.'
    },
    {
      id: 'heuristicWeight',
      label: 'Heuristic Weight (w_h)',
      type: 'slider',
      min: 0.5,
      max: 2.5,
      step: 0.1,
      defaultValue: 1.3,
      description: 'Weight applied to non-holonomic goal heuristic.'
    },
    {
      id: 'allowReverse',
      label: 'Allow Reverse Gear',
      type: 'toggle',
      defaultValue: true,
      description: 'Allow reverse-gear primitives for tight turns and parking.'
    }
  ],

  documentation: {
    intuition:
      'Standard grid pathfinding assumes holonomic motion without orientation constraints, which fails for real vehicles that cannot move sideways and have a minimum turning radius R_min. Hybrid A* bridges continuous kinematics and discrete graph search: candidate vehicle states exist in continuous (x, y, θ) space produced by Ackermann steering arc primitives, while redundant states are pruned by hashing poses into a 3D discretization grid. Explored forward arcs fan out in Sky Blue, reverse primitives in Amber, and the winning trajectory with vehicle footprint boxes is highlighted in glowing Emerald.',
    governingEquationLatex:
      'f(\\textcolor{#38bdf8}{\\mathbf{q}}) = g(\\textcolor{#38bdf8}{x,y,\\theta}) + w_{\\text{rev}}\\,\\textcolor{#f43f5e}{c_{\\text{gear}}} + w_h\\,\\textcolor{#f59e0b}{h(\\mathbf{q}, \\textcolor{#10b981}{\\mathbf{q}_{\\text{goal}}})}',
    pseudocode: [
      '1. Initialize Priority Queue: Push start pose q_0 = (x_0, y_0, θ_0) with g=0, f=w_h·h(q_0)',
      '2. Pop Lowest Cost Node: q_curr = argmin_{q ∈ Open} f(q) from priority queue',
      '3. Goal Check: If within goal tolerance (x, y, θ), reconstruct kinematic path and finish',
      '4. Generate Primitives: Roll out Ackermann arcs for steer ∈ {-1, 0, 1} and gear ∈ {+1, -1}',
      '5. Continuous Collision Check: Test vehicle footprint along each arc against obstacle geometry',
      '6. Discrete 3D Binning: Map (x, y, θ) to grid cell (gx, gy, gθ); prune if cell has lower g',
      '7. Insert Open Set: Compute g_new = g + ds·c_gear and f_new = g_new + w_h·h; push to Open'
    ]
  },

  createInitialState(params: HybridAStarParams): HybridAStarState {
    const start: CarPose = { x: -210, y: 80, theta: 0 };
    const goal: CarPose = { x: 190, y: -70, theta: 0 };
    const obstacles: Obstacle[] = [
      { id: 0, x: -75, y: 15, radius: 40, label: 'Obs 1' },
      { id: 1, x: 45, y: -20, radius: 44, label: 'Obs 2' },
      { id: 2, x: 60, y: 80, radius: 36, label: 'Obs 3' }
    ];

    return createFreshSearch(start, goal, obstacles, params);
  },

  step(currentState: HybridAStarState, params: HybridAStarParams) {
    if (currentState.foundPath || currentState.isComplete || currentState.openSet.length === 0) {
      return {
        nextState: {
          ...currentState,
          isComplete: true
        },
        isComplete: true,
        activePseudocodeLine: currentState.foundPath ? 2 : 1
      };
    }

    const nodes = [...currentState.nodes];
    const openSet = [...currentState.openSet];
    const visited = { ...currentState.visited };
    const exploredArcs = [...currentState.exploredArcs];
    const goal = currentState.goal;
    const obstacles = currentState.obstacles;

    let foundPath = false;
    let winningNodeId: number | null = null;
    let nodesExplored = currentState.nodesExplored;
    let activeLine = 1;

    const steerAngles = [-1, 0, 1];
    const gears = params.allowReverse ? [1, -1] : [1];

    for (let exp = 0; exp < EXPANSIONS_PER_STEP; exp++) {
      const currId = heapPop(openSet, nodes);
      if (currId === null) break;

      const currNode = nodes[currId];
      nodesExplored++;
      activeLine = 1;

      // 3. Goal Tolerance Check
      const distToGoal = Math.hypot(goal.x - currNode.x, goal.y - currNode.y);
      const headingDiff = Math.abs(normalizeAngle(goal.theta - currNode.theta));

      if (distToGoal <= 26 && headingDiff <= 0.48) {
        foundPath = true;
        winningNodeId = currId;
        activeLine = 2;
        break;
      }

      // 4. Generate Kinematic Motion Primitives
      for (const gear of gears) {
        for (const steer of steerAngles) {
          const arcPoints: { x: number; y: number }[] = [{ x: currNode.x, y: currNode.y }];
          let simX = currNode.x;
          let simY = currNode.y;
          let simTheta = currNode.theta;
          let inCollision = false;

          const microDs = STEP_ARC_LEN / MICRO_STEPS;
          const kappa = steer === 0 ? 0 : steer / params.minTurnRadius;

          for (let s = 0; s < MICRO_STEPS; s++) {
            simX += gear * microDs * Math.cos(simTheta);
            simY += gear * microDs * Math.sin(simTheta);
            simTheta = normalizeAngle(simTheta + gear * microDs * kappa);

            arcPoints.push({ x: simX, y: simY });

            if (checkVehicleCollision(simX, simY, simTheta, obstacles)) {
              inCollision = true;
              break;
            }
          }

          if (inCollision) continue;

          // Compute Cost: movement + reverse penalty + gear switch + steer change
          const gearCostMult = gear < 0 ? params.reversePenalty : 1.0;
          const gearSwitchPenalty = currNode.gear !== gear ? 10.0 : 0;
          const steerSwitchPenalty = currNode.steer !== steer ? 2.0 : 0;
          const edgeCost = STEP_ARC_LEN * gearCostMult + gearSwitchPenalty + steerSwitchPenalty;
          const tentativeG = currNode.g + edgeCost;

          // 6. Discrete 3D Grid Binning
          const cellKey = getGridKey(simX, simY, simTheta);
          if (visited[cellKey] !== undefined && visited[cellKey] <= tentativeG) {
            continue; // Prune state
          }
          visited[cellKey] = tentativeG;

          // 7. Push to Open Priority Queue
          const h = computeHeuristic(
            simX,
            simY,
            simTheta,
            goal.x,
            goal.y,
            goal.theta,
            params.minTurnRadius
          );
          const f = tentativeG + params.heuristicWeight * h;

          const newNodeId = nodes.length;
          const newNode: HybridNode = {
            id: newNodeId,
            x: simX,
            y: simY,
            theta: simTheta,
            g: tentativeG,
            h,
            f,
            parentId: currId,
            steer,
            gear,
            arcPoints
          };

          nodes.push(newNode);
          heapPush(openSet, newNodeId, nodes);

          if (exploredArcs.length < 650) {
            exploredArcs.push({
              startX: currNode.x,
              startY: currNode.y,
              endX: simX,
              endY: simY,
              points: arcPoints,
              gear,
              steer
            });
          }
        }
      }
    }

    // Reconstruct trajectory if goal reached
    let path: CarPose[] = currentState.path;
    let pathCost = currentState.pathCost;

    if (foundPath && winningNodeId !== null) {
      path = [];
      let traceId: number | null = winningNodeId;
      while (traceId !== null) {
        const n: HybridNode = nodes[traceId];
        path.unshift({ x: n.x, y: n.y, theta: n.theta });
        traceId = n.parentId;
      }
      pathCost = nodes[winningNodeId].g;
    }

    const isComplete = foundPath || openSet.length === 0;

    return {
      nextState: {
        ...currentState,
        nodes,
        openSet,
        visited,
        exploredArcs,
        path,
        pathCost,
        foundPath,
        isComplete,
        nodesExplored,
        iteration: currentState.iteration + 1,
        activeLine: foundPath ? 2 : activeLine
      },
      isComplete,
      activePseudocodeLine: foundPath ? 2 : activeLine
    };
  },

  onPointerInteraction(
    currentState: HybridAStarState,
    event: CanvasPointerEvent,
    params: HybridAStarParams
  ): HybridAStarState {
    const obstacles = currentState.obstacles.map((o) => ({ ...o }));
    let goal = { ...currentState.goal };
    let dragTarget = currentState.dragTarget;

    if (!event.isDragging) {
      if (event.shiftKey) {
        goal = {
          ...goal,
          theta: normalizeAngle(goal.theta + Math.PI / 4)
        };
        return createFreshSearch(currentState.start, goal, obstacles, params);
      }

      const dGoal = Math.hypot(event.worldX - goal.x, event.worldY - goal.y);
      if (dGoal < 32) {
        dragTarget = 'goal';
      } else {
        const hitObs = obstacles.find(
          (o) => Math.hypot(event.worldX - o.x, event.worldY - o.y) <= o.radius + 14
        );
        if (hitObs !== undefined) {
          dragTarget = hitObs.id;
        } else {
          goal = {
            x: Math.max(-250, Math.min(250, event.worldX)),
            y: Math.max(-145, Math.min(145, event.worldY)),
            theta: goal.theta
          };
          dragTarget = 'goal';
          const fresh = createFreshSearch(currentState.start, goal, obstacles, params);
          return { ...fresh, dragTarget };
        }
      }
    } else {
      if (dragTarget === 'goal') {
        goal = {
          x: Math.max(-250, Math.min(250, event.worldX)),
          y: Math.max(-145, Math.min(145, event.worldY)),
          theta: goal.theta
        };
      } else if (typeof dragTarget === 'number' && obstacles[dragTarget]) {
        obstacles[dragTarget] = {
          ...obstacles[dragTarget],
          x: Math.max(-240, Math.min(240, event.worldX)),
          y: Math.max(-140, Math.min(140, event.worldY))
        };
      } else {
        goal = {
          x: Math.max(-250, Math.min(250, event.worldX)),
          y: Math.max(-145, Math.min(145, event.worldY)),
          theta: goal.theta
        };
        dragTarget = 'goal';
      }

      const fresh = createFreshSearch(currentState.start, goal, obstacles, params);
      return {
        ...fresh,
        dragTarget
      };
    }

    return {
      ...currentState,
      dragTarget
    };
  },

  getMetrics(currentState: HybridAStarState): MetricReadout[] {
    const statusLabel = currentState.foundPath
      ? 'Path Found (Parked)'
      : currentState.openSet.length === 0
      ? 'Search Exhausted'
      : 'Searching Arcs...';

    const statusColor = currentState.foundPath
      ? 'emerald'
      : currentState.openSet.length === 0
      ? 'rose'
      : 'sky';

    return [
      {
        id: 'status',
        label: 'Planner Status',
        value: statusLabel,
        color: statusColor
      },
      {
        id: 'cost',
        label: 'Trajectory Cost',
        value: currentState.foundPath ? currentState.pathCost.toFixed(1) : '--',
        unit: currentState.foundPath ? ' cost' : undefined,
        color: currentState.foundPath ? 'emerald' : 'default'
      },
      {
        id: 'explored',
        label: 'Kinematic States',
        value: currentState.nodesExplored,
        color: 'sky'
      },
      {
        id: 'open',
        label: 'Open Priority Queue',
        value: currentState.openSet.length,
        color: 'amber'
      }
    ];
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: HybridAStarState,
    _params: HybridAStarParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const cx = viewport.width / 2;
    const cy = viewport.height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Boundary Arena Limits
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(-275, -165, 550, 330);
    ctx.setLineDash([]);

    // 2. Obstacles (Rose #f43f5e) with safety inflation zone
    for (const obs of state.obstacles) {
      // Safety zone
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.radius + COLLISION_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Body
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillStyle = '#fca5a5';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(obs.label || `Obs ${obs.id + 1}`, obs.x, obs.y);
    }

    // 3. Explored Motion Primitives (Sky Blue #38bdf8 forward, Amber #f59e0b reverse)
    for (const arc of state.exploredArcs) {
      if (arc.points.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(arc.points[0].x, arc.points[0].y);
      for (let i = 1; i < arc.points.length; i++) {
        ctx.lineTo(arc.points[i].x, arc.points[i].y);
      }

      if (arc.gear < 0) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.42)'; // Reverse gear
        ctx.lineWidth = 1.1;
      } else {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)'; // Forward gear
        ctx.lineWidth = 1.1;
      }
      ctx.stroke();
    }

    // 4. Goal Parking Bay (Emerald #10b981)
    const gx = state.goal.x;
    const gy = state.goal.y;
    const gth = state.goal.theta;

    ctx.save();
    ctx.translate(gx, gy);
    ctx.rotate(gth);

    // Parking slot border
    ctx.beginPath();
    ctx.rect(-CAR_LENGTH / 2 - 4, -CAR_WIDTH / 2 - 4, CAR_LENGTH + 8, CAR_WIDTH + 8);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.65)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Parking symbol 'P'
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('P', 0, 0);

    // Goal heading indicator
    ctx.beginPath();
    ctx.moveTo(CAR_LENGTH / 2 + 4, 0);
    ctx.lineTo(CAR_LENGTH / 2 + 14, 0);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.restore();

    // 5. Start Vehicle Pose (Sky Blue #0ea5e9)
    const sx = state.start.x;
    const sy = state.start.y;
    const sth = state.start.theta;

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(sth);

    // Chassis
    ctx.beginPath();
    ctx.rect(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Wheels
    ctx.fillStyle = '#334155';
    ctx.fillRect(-CAR_AXLE_OFFSET - 3, -CAR_WIDTH / 2 - 2, 6, 2.5);
    ctx.fillRect(-CAR_AXLE_OFFSET - 3, CAR_WIDTH / 2 - 0.5, 6, 2.5);
    ctx.fillRect(CAR_AXLE_OFFSET - 3, -CAR_WIDTH / 2 - 2, 6, 2.5);
    ctx.fillRect(CAR_AXLE_OFFSET - 3, CAR_WIDTH / 2 - 0.5, 6, 2.5);

    // Headlight indicator
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(CAR_LENGTH / 2 + 6, 0);
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();

    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillStyle = '#7dd3fc';
    ctx.textAlign = 'center';
    ctx.fillText('Start q_0', sx, sy + 20);

    // 6. Winning Kinematic Trajectory & Footprint Boxes (Glowing Emerald #10b981)
    if (state.path.length > 1) {
      // Glow trajectory path
      ctx.save();
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(state.path[0].x, state.path[0].y);
      for (let i = 1; i < state.path.length; i++) {
        ctx.lineTo(state.path[i].x, state.path[i].y);
      }
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.restore();

      // Vehicle footprint rectangles spaced along trajectory
      const stepInterval = Math.max(1, Math.floor(state.path.length / 8));
      for (let i = stepInterval; i < state.path.length - 1; i += stepInterval) {
        const pose = state.path[i];
        ctx.save();
        ctx.translate(pose.x, pose.y);
        ctx.rotate(pose.theta);

        ctx.beginPath();
        ctx.rect(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.16)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.75)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.restore();
      }
    }

    // 7. Status Banner
    if (state.foundPath) {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#10b981';
      ctx.textAlign = 'center';
      ctx.fillText(
        `✓ Non-Holonomic Path Found (Cost: ${state.pathCost.toFixed(1)}, Explored: ${state.nodesExplored} states)`,
        0,
        -145
      );
    } else if (state.openSet.length === 0) {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.textAlign = 'center';
      ctx.fillText(
        '✕ No Feasible Path — Try enabling Reverse Gear or moving Rose Obstacles',
        0,
        -145
      );
    } else {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#38bdf8';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Expanding Kinematic Motion Primitives... (Explored ${state.nodesExplored} states, Open: ${state.openSet.length})`,
        0,
        -145
      );
    }

    // Helper Footer
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(
      'Sky Blue: Forward Arcs | Amber: Reverse Primitives | Rose: Obstacles | Emerald: Winning Kinematic Trajectory',
      0,
      155
    );

    ctx.restore();
  }
};

export default HybridAStarPlugin;
