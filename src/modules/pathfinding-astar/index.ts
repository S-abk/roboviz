// src/modules/pathfinding-astar/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

interface GridNode {
  x: number;
  y: number;
  isWall: boolean;
  g: number; // Exact cost from start
  h: number; // Heuristic cost to goal
  f: number; // Total score: g + w * h
  parent: { x: number; y: number } | null;
  status: 'unvisited' | 'open' | 'closed' | 'path';
}

export interface AStarState {
  cols: number;
  rows: number;
  grid: GridNode[][];
  start: { x: number; y: number };
  goal: { x: number; y: number };
  openList: { x: number; y: number }[];
  nodesExplored: number;
  pathLength: number;
  isFinished: boolean;
  foundPath: boolean;
  lastExpanded: { x: number; y: number } | null;
}

export interface AStarParams {
  heuristicWeight: number;
  allowDiagonals: boolean;
  showCosts: boolean;
}

function computeHeuristic(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  allowDiagonals: boolean
): number {
  const dx = Math.abs(ax - bx);
  const dy = Math.abs(ay - by);
  return allowDiagonals ? Math.hypot(dx, dy) : dx + dy;
}

export const AStarPathfindingPlugin: AlgorithmPlugin<AStarState, AStarParams> = {
  id: 'motion-planning-astar',
  title: 'A* & Weighted Dijkstra Search',
  shortDescription: 'Interactive grid pathfinding comparing Dijkstra (w=0), A* (w=1), and Greedy Best-First search.',
  category: 'motion-planning',

  controls: [
    {
      id: 'heuristicWeight',
      label: 'Heuristic Weight (w)',
      type: 'slider',
      min: 0.0,
      max: 3.0,
      step: 0.1,
      defaultValue: 1.0,
      description: 'w=0 is Dijkstra, w=1 is optimal A*, w>1 is Greedy Weighted A*'
    },
    {
      id: 'allowDiagonals',
      label: 'Allow 8-Way Diagonal Motion',
      type: 'toggle',
      defaultValue: false
    },
    {
      id: 'showCosts',
      label: 'Show Cell f(n) Scores',
      type: 'toggle',
      defaultValue: true
    }
  ],

  documentation: {
    intuition:
      'A* finds the shortest collision-free route by prioritizing grid cells with the lowest combined cost f(n) = g(n) + w·h(n). Lowering w to 0 turns A* into Dijkstra (exploring outward in all directions), while raising w > 1 makes the search greedily rush toward the goal.',
    governingEquationLatex:
      'f(n) = \\textcolor{#0ea5e9}{g(n)} + w \\cdot \\textcolor{#f59e0b}{h(n)}',
    pseudocode: [
      '1. Push Start node into Open Set with g(s) = 0, f(s) = w·h(s)',
      '2. Pop node n from Open Set with lowest f(n) score',
      '3. If n == Goal, reconstruct optimal path & terminate',
      '4. Mark node n as Closed (Explored)',
      '5. For each collision-free neighbor m of n:',
      '6.   tentative_g = g(n) + cost(n, m)',
      '7.   If tentative_g < g(m): update parent, g(m), and f(m)'
    ]
  },

  createInitialState(params: AStarParams): AStarState {
    const cols = 18;
    const rows = 12;
    const start = { x: 2, y: 6 };
    const goal = { x: 15, y: 6 };

    const grid: GridNode[][] = [];
    for (let y = 0; y < rows; y++) {
      const row: GridNode[] = [];
      for (let x = 0; x < cols; x++) {
        // Default obstacle wall barrier in the center
        const isDefaultWall = (x === 9 && y >= 2 && y <= 9) || (x === 7 && y === 2) || (x === 8 && y === 2);
        row.push({
          x,
          y,
          isWall: isDefaultWall,
          g: Infinity,
          h: computeHeuristic(x, y, goal.x, goal.y, params.allowDiagonals),
          f: Infinity,
          parent: null,
          status: 'unvisited'
        });
      }
      grid.push(row);
    }

    const startNode = grid[start.y][start.x];
    startNode.g = 0;
    startNode.f = params.heuristicWeight * startNode.h;
    startNode.status = 'open';

    return {
      cols,
      rows,
      grid,
      start,
      goal,
      openList: [{ x: start.x, y: start.y }],
      nodesExplored: 0,
      pathLength: 0,
      isFinished: false,
      foundPath: false,
      lastExpanded: null
    };
  },

  step(state: AStarState, params: AStarParams) {
    if (state.isFinished || state.openList.length === 0) {
      return {
        nextState: { ...state, isFinished: true },
        isComplete: true,
        activePseudocodeLine: 2
      };
    }

    // 1. Find node in openList with lowest f(n) (tie-break on lowest h)
    let bestIdx = 0;
    for (let i = 1; i < state.openList.length; i++) {
      const candidate = state.grid[state.openList[i].y][state.openList[i].x];
      const currentBest = state.grid[state.openList[bestIdx].y][state.openList[bestIdx].x];
      if (
        candidate.f < currentBest.f ||
        (Math.abs(candidate.f - currentBest.f) < 1e-5 && candidate.h < currentBest.h)
      ) {
        bestIdx = i;
      }
    }

    const currentCoord = state.openList.splice(bestIdx, 1)[0];
    const current = state.grid[currentCoord.y][currentCoord.x];
    state.lastExpanded = currentCoord;

    // 2. Check if Goal Reached
    if (current.x === state.goal.x && current.y === state.goal.y) {
      let curr: GridNode | null = current;
      let length = 0;
      while (curr) {
        curr.status = 'path';
        if (curr.parent) {
          length += Math.hypot(curr.x - curr.parent.x, curr.y - curr.parent.y);
          curr = state.grid[curr.parent.y][curr.parent.x];
        } else {
          curr = null;
        }
      }
      state.pathLength = Number(length.toFixed(1));
      state.isFinished = true;
      state.foundPath = true;

      return {
        nextState: state,
        isComplete: true,
        activePseudocodeLine: 2
      };
    }

    current.status = 'closed';
    state.nodesExplored++;

    // 3. Expand Neighbors
    const dirs = [
      { dx: 0, dy: -1, cost: 1 },
      { dx: 1, dy: 0, cost: 1 },
      { dx: 0, dy: 1, cost: 1 },
      { dx: -1, dy: 0, cost: 1 }
    ];
    if (params.allowDiagonals) {
      dirs.push(
        { dx: 1, dy: -1, cost: Math.SQRT2 },
        { dx: 1, dy: 1, cost: Math.SQRT2 },
        { dx: -1, dy: 1, cost: Math.SQRT2 },
        { dx: -1, dy: -1, cost: Math.SQRT2 }
      );
    }

    for (const d of dirs) {
      const nx = current.x + d.dx;
      const ny = current.y + d.dy;
      if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) continue;

      const neighbor = state.grid[ny][nx];
      if (neighbor.isWall || neighbor.status === 'closed') continue;

      const tentativeG = current.g + d.cost;
      if (tentativeG < neighbor.g) {
        neighbor.parent = { x: current.x, y: current.y };
        neighbor.g = tentativeG;
        neighbor.h = computeHeuristic(nx, ny, state.goal.x, state.goal.y, params.allowDiagonals);
        neighbor.f = neighbor.g + params.heuristicWeight * neighbor.h;

        if (neighbor.status !== 'open') {
          neighbor.status = 'open';
          state.openList.push({ x: nx, y: ny });
        }
      }
    }

    return {
      nextState: state,
      isComplete: false,
      activePseudocodeLine: 6
    };
  },

  onPointerInteraction(state: AStarState, event: CanvasPointerEvent): AStarState {
    const cellSize = 36;
    const gridW = state.cols * cellSize;
    const gridH = state.rows * cellSize;

    const localX = event.worldX + gridW / 2;
    const localY = event.worldY + gridH / 2;

    const gx = Math.floor(localX / cellSize);
    const gy = Math.floor(localY / cellSize);

    if (gx >= 0 && gx < state.cols && gy >= 0 && gy < state.rows) {
      const isStart = gx === state.start.x && gy === state.start.y;
      const isGoal = gx === state.goal.x && gy === state.goal.y;
      if (!isStart && !isGoal && state.grid[gy][gx].status === 'unvisited') {
        state.grid[gy][gx].isWall = !event.shiftKey;
      }
    }
    return state;
  },

  getMetrics(state: AStarState, params: AStarParams): MetricReadout[] {
    const modeName =
      params.heuristicWeight === 0
        ? 'Dijkstra'
        : params.heuristicWeight === 1
        ? 'Optimal A*'
        : 'Greedy A*';
    return [
      { id: 'mode', label: 'Search Mode', value: modeName, color: 'amber' },
      { id: 'explored', label: 'Nodes Explored', value: state.nodesExplored, color: 'sky' },
      {
        id: 'path',
        label: 'Path Cost',
        value: state.foundPath ? state.pathLength : 'Searching...',
        color: state.foundPath ? 'emerald' : 'default'
      }
    ];
  },

  render(ctx, state, params, viewport) {
    const cellSize = 36;
    const gridW = state.cols * cellSize;
    const gridH = state.rows * cellSize;
    const startX = (viewport.width - gridW) / 2;
    const startY = (viewport.height - gridH) / 2;

    ctx.save();
    ctx.translate(startX, startY);

    for (let y = 0; y < state.rows; y++) {
      for (let x = 0; x < state.cols; x++) {
        const cell = state.grid[y][x];
        const px = x * cellSize;
        const py = y * cellSize;

        // Fill cell based on state
        if (x === state.start.x && y === state.start.y) {
          ctx.fillStyle = '#0ea5e9';
        } else if (x === state.goal.x && y === state.goal.y) {
          ctx.fillStyle = '#10b981';
        } else if (cell.isWall) {
          ctx.fillStyle = '#334155';
        } else if (cell.status === 'path') {
          ctx.fillStyle = 'rgba(16, 185, 129, 0.75)';
        } else if (cell.status === 'open') {
          ctx.fillStyle = 'rgba(245, 158, 11, 0.30)';
        } else if (cell.status === 'closed') {
          ctx.fillStyle = 'rgba(14, 165, 233, 0.22)';
        } else {
          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        }

        ctx.fillRect(px + 1, py + 1, cellSize - 2, cellSize - 2);

        // Draw f(n) score if enabled
        if (params.showCosts && cell.f < Infinity && !cell.isWall) {
          ctx.fillStyle = '#e2e8f0';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(cell.f.toFixed(1), px + cellSize / 2, py + cellSize / 2);
        }
      }
    }

    ctx.restore();
  }
};
