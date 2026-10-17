// FILE: src/modules/perception-occupancy-grid/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface LaserRay {
  angle: number;
  hitX: number;
  hitY: number;
  distance: number;
  hitObstacle: boolean;
}

export interface OccupancyGridState {
  timeStep: number;
  cols: number;
  rows: number;
  cellSize: number;
  robotX: number;
  robotY: number;
  robotTheta: number;
  waypointIdx: number;
  trueMap: boolean[][];
  logOdds: number[][];
  activeRays: LaserRay[];
  exploredRatio: number;
  occupiedCellCount: number;
  activeHitsCount: number;
}

export interface OccupancyGridParams {
  lidarBeams: number;
  maxRange: number;
  logOddsOcc: number;
  logOddsFree: number;
  showTrueMapOverlay: boolean;
}

const COLS = 24;
const ROWS = 16;
const CELL_SIZE = 26;
const HALF_W = (COLS * CELL_SIZE) / 2; // 312
const HALF_H = (ROWS * CELL_SIZE) / 2; // 208
const ROBOT_RADIUS = 8;

function worldToCell(wx: number, wy: number): { col: number; row: number } {
  return {
    col: Math.max(0, Math.min(COLS - 1, Math.floor((wx + HALF_W) / CELL_SIZE))),
    row: Math.max(0, Math.min(ROWS - 1, Math.floor((wy + HALF_H) / CELL_SIZE)))
  };
}

function cellToWorldCenter(col: number, row: number): { x: number; y: number } {
  return {
    x: -HALF_W + col * CELL_SIZE + CELL_SIZE / 2,
    y: -HALF_H + row * CELL_SIZE + CELL_SIZE / 2
  };
}

// Exact 2D grid raycast (Amanatides & Woo / Bresenham-DDA) that returns traversed cells
// and the exact wall-boundary impact point (hitX, hitY) on the first occupied cell
function castGridRay(
  startX: number,
  startY: number,
  angle: number,
  maxRange: number,
  trueMap: boolean[][]
): {
  freeCells: { col: number; row: number }[];
  hitCell: { col: number; row: number } | null;
  hitX: number;
  hitY: number;
  distance: number;
  hitObstacle: boolean;
} {
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  let col = Math.floor((startX + HALF_W) / CELL_SIZE);
  let row = Math.floor((startY + HALF_H) / CELL_SIZE);

  const stepC = cosA >= 0 ? 1 : -1;
  const stepR = sinA >= 0 ? 1 : -1;

  const nextBoundaryX = -HALF_W + (cosA >= 0 ? col + 1 : col) * CELL_SIZE;
  const nextBoundaryY = -HALF_H + (sinA >= 0 ? row + 1 : row) * CELL_SIZE;

  let tMaxX = Math.abs(cosA) > 1e-8 ? (nextBoundaryX - startX) / cosA : Infinity;
  let tMaxY = Math.abs(sinA) > 1e-8 ? (nextBoundaryY - startY) / sinA : Infinity;

  const tDeltaX = Math.abs(cosA) > 1e-8 ? Math.abs(CELL_SIZE / cosA) : Infinity;
  const tDeltaY = Math.abs(sinA) > 1e-8 ? Math.abs(CELL_SIZE / sinA) : Infinity;

  const freeCells: { col: number; row: number }[] = [];
  let t = 0;

  while (t <= maxRange) {
    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) {
      break;
    }

    if (trueMap[row][col]) {
      const hitDist = Math.min(t, maxRange);
      return {
        freeCells,
        hitCell: { col, row },
        hitX: startX + cosA * hitDist,
        hitY: startY + sinA * hitDist,
        distance: hitDist,
        hitObstacle: true
      };
    }

    freeCells.push({ col, row });

    if (tMaxX < tMaxY) {
      t = tMaxX;
      tMaxX += tDeltaX;
      col += stepC;
    } else {
      t = tMaxY;
      tMaxY += tDeltaY;
      row += stepR;
    }
  }

  return {
    freeCells,
    hitCell: null,
    hitX: startX + cosA * maxRange,
    hitY: startY + sinA * maxRange,
    distance: maxRange,
    hitObstacle: false
  };
}

function createDefaultObstacleMap(): boolean[][] {
  const map: boolean[][] = Array.from({ length: ROWS }, () => new Array(COLS).fill(false));

  for (let c = 0; c < COLS; c++) {
    map[0][c] = true;
    map[ROWS - 1][c] = true;
  }
  for (let r = 0; r < ROWS; r++) {
    map[r][0] = true;
    map[r][COLS - 1] = true;
  }

  for (let r = 3; r <= 6; r++) {
    map[r][5] = true;
    map[r][6] = true;
  }
  for (let r = 9; r <= 12; r++) {
    map[r][5] = true;
    map[r][6] = true;
  }
  for (let r = 5; r <= 10; r++) {
    map[r][11] = true;
    map[r][12] = true;
  }
  for (let r = 3; r <= 6; r++) {
    map[r][17] = true;
    map[r][18] = true;
  }
  for (let r = 9; r <= 12; r++) {
    map[r][17] = true;
    map[r][18] = true;
  }

  return map;
}

// Waypoints defined in free corridor cells so the patrol route weaves cleanly around all pillars
const PATROL_WAYPOINTS: { col: number; row: number }[] = [
  { col: 2, row: 2 },
  { col: 8, row: 2 },
  { col: 8, row: 7 },
  { col: 3, row: 7 },
  { col: 2, row: 13 },
  { col: 9, row: 13 },
  { col: 15, row: 13 },
  { col: 15, row: 8 },
  { col: 20, row: 8 },
  { col: 21, row: 13 },
  { col: 21, row: 2 },
  { col: 15, row: 2 }
];

// Shortest collision-free grid step from startCell toward goalCell using 8-connected BFS (no corner cutting)
function findNextFreeStep(
  startCol: number,
  startRow: number,
  goalCol: number,
  goalRow: number,
  trueMap: boolean[][]
): { col: number; row: number } | null {
  if (trueMap[goalRow]?.[goalCol]) return null;
  if (startCol === goalCol && startRow === goalRow) {
    return { col: goalCol, row: goalRow };
  }

  const visited = Array.from({ length: ROWS }, () => new Array(COLS).fill(false));
  const parent: ({ col: number; row: number } | null)[][] = Array.from({ length: ROWS }, () =>
    new Array(COLS).fill(null)
  );

  const queue: { col: number; row: number }[] = [{ col: startCol, row: startRow }];
  visited[startRow][startCol] = true;

  const dirs = [
    { dc: 1, dr: 0 },
    { dc: -1, dr: 0 },
    { dc: 0, dr: 1 },
    { dc: 0, dr: -1 },
    { dc: 1, dr: 1 },
    { dc: 1, dr: -1 },
    { dc: -1, dr: 1 },
    { dc: -1, dr: -1 }
  ];

  let found = false;
  while (queue.length > 0) {
    const curr = queue.shift()!;
    if (curr.col === goalCol && curr.row === goalRow) {
      found = true;
      break;
    }

    for (const { dc, dr } of dirs) {
      const nc = curr.col + dc;
      const nr = curr.row + dr;
      if (nc < 0 || nc >= COLS || nr < 0 || nr >= ROWS) continue;
      if (visited[nr][nc] || trueMap[nr][nc]) continue;

      // Prevent diagonal squeezing past orthogonal obstacle corners
      if (dc !== 0 && dr !== 0) {
        if (trueMap[curr.row][nc] || trueMap[nr][curr.col]) continue;
      }

      visited[nr][nc] = true;
      parent[nr][nc] = curr;
      queue.push({ col: nc, row: nr });
    }
  }

  if (!found) return null;

  // Backtrack from goal to find the immediate next waypoint cell along the shortest free path
  let step = { col: goalCol, row: goalRow };
  while (parent[step.row][step.col]) {
    const p = parent[step.row][step.col]!;
    if (p.col === startCol && p.row === startRow) {
      return step;
    }
    step = p;
  }
  return step;
}

// Push robot circle out of any solid grid cell AABBs so it never clips or passes through walls
function resolveRobotGridCollision(
  rx: number,
  ry: number,
  trueMap: boolean[][]
): { x: number; y: number; collided: boolean } {
  let x = rx;
  let y = ry;
  let collided = false;
  const pad = ROBOT_RADIUS + 2;

  for (let pass = 0; pass < 3; pass++) {
    const cCell = worldToCell(x, y);
    for (let r = Math.max(0, cCell.row - 2); r <= Math.min(ROWS - 1, cCell.row + 2); r++) {
      for (let c = Math.max(0, cCell.col - 2); c <= Math.min(COLS - 1, cCell.col + 2); c++) {
        if (!trueMap[r][c]) continue;

        const minX = -HALF_W + c * CELL_SIZE;
        const maxX = minX + CELL_SIZE;
        const minY = -HALF_H + r * CELL_SIZE;
        const maxY = minY + CELL_SIZE;

        const closestX = Math.max(minX, Math.min(x, maxX));
        const closestY = Math.max(minY, Math.min(y, maxY));
        const dx = x - closestX;
        const dy = y - closestY;
        const distSq = dx * dx + dy * dy;

        if (distSq < pad * pad) {
          collided = true;
          const dist = Math.sqrt(distSq);
          if (dist > 1e-4) {
            const overlap = pad - dist;
            x += (dx / dist) * overlap;
            y += (dy / dist) * overlap;
          } else {
            // Center is inside cell AABB: push out along minimum penetration axis
            const center = cellToWorldCenter(c, r);
            const ox = x - center.x;
            const oy = y - center.y;
            if (Math.abs(ox) > Math.abs(oy)) {
              x = ox >= 0 ? maxX + pad : minX - pad;
            } else {
              y = oy >= 0 ? maxY + pad : minY - pad;
            }
          }
        }
      }
    }
  }

  return { x, y, collided };
}

export const OccupancyGridPlugin: AlgorithmPlugin<OccupancyGridState, OccupancyGridParams> = {
  id: 'perception-occupancy-grid',
  title: 'Log-Odds Occupancy Grid Mapping',
  shortDescription: 'Recursive Bayesian mapping updating a discrete grid from unknown to free or occupied via Bresenham raycasting.',
  category: 'perception-slam',

  controls: [
    {
      id: 'lidarBeams',
      label: 'LiDAR Beam Resolution',
      type: 'slider',
      min: 12,
      max: 72,
      step: 6,
      defaultValue: 36,
      description: 'Number of radial range sensors cast over 360° FOV.'
    },
    {
      id: 'maxRange',
      label: 'Max Sensor Range (d_max)',
      type: 'slider',
      min: 80,
      max: 260,
      step: 20,
      defaultValue: 180,
      unit: 'px'
    },
    {
      id: 'logOddsOcc',
      label: 'Occupied Increment (l_occ)',
      type: 'slider',
      min: 0.4,
      max: 2.0,
      step: 0.1,
      defaultValue: 0.85,
      description: 'Log-odds added to cells when a laser ray reflects off an obstacle.'
    },
    {
      id: 'logOddsFree',
      label: 'Free-Space Decrement (l_free)',
      type: 'slider',
      min: 0.1,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.4,
      description: 'Log-odds subtracted from cells traversed by an unobstructed beam.'
    },
    {
      id: 'showTrueMapOverlay',
      label: 'Show Ground Truth Obstacle Outlines',
      type: 'toggle',
      defaultValue: true
    }
  ],

  documentation: {
    intuition:
      'Occupancy grid mapping decomposes continuous physical space into a grid of independent binary random variables m_i. As a range sensor scans the arena, cells traversed by laser rays via Bresenham raycasting receive negative log-odds increments (evidence of free space), while the terminal obstacle hit cell receives a positive log-odds increment.',
    governingEquationLatex:
      '\\textcolor{#0ea5e9}{l_{t,i}} = \\textcolor{#0ea5e9}{l_{t-1,i}} + \\operatorname{inv\\_sensor}(\\textcolor{#f59e0b}{z_t}, m_i) - \\textcolor{#38bdf8}{l_0}, \\quad \\textcolor{#f43f5e}{p(m_i)} = \\frac{1}{1 + e^{-\\textcolor{#0ea5e9}{l_{t,i}}}}',
    pseudocode: [
      '1. Propagate Robot: Advance pose (x, y, theta) along collision-free corridor',
      '2. Radial LiDAR Emission: Cast N range beams uniformly across 360 degrees',
      '3. Bresenham Raycast: Trace integer grid cells (c, r) along each beam',
      '4. Free Space Update: For all empty cells traversed, l_i <- l_i - l_free',
      '5. Obstacle Hit Update: For the endpoint hit cell, l_hit <- l_hit + l_occ',
      '6. Numerical Clamping: Restrict l_i to [-5.0, +5.0] to prevent belief saturation',
      '7. Probability Evaluation: Convert log-odds to occupancy belief p(m) = 1/(1+e^-l)'
    ]
  },

  createInitialState(): OccupancyGridState {
    const logOdds: number[][] = Array.from({ length: ROWS }, () => new Array(COLS).fill(0.0));
    const startWp = cellToWorldCenter(PATROL_WAYPOINTS[0].col, PATROL_WAYPOINTS[0].row);

    return {
      timeStep: 0,
      cols: COLS,
      rows: ROWS,
      cellSize: CELL_SIZE,
      robotX: startWp.x,
      robotY: startWp.y,
      robotTheta: 0,
      waypointIdx: 1,
      trueMap: createDefaultObstacleMap(),
      logOdds,
      activeRays: [],
      exploredRatio: 0,
      occupiedCellCount: 0,
      activeHitsCount: 0
    };
  },

  step(currentState: OccupancyGridState, params: OccupancyGridParams) {
    const nextStep = currentState.timeStep + 1;
    const speed = 3.8;

    const currCell = worldToCell(currentState.robotX, currentState.robotY);
    let wpIdx = currentState.waypointIdx;

    // Find a reachable patrol waypoint and its next collision-free step
    let nextCellStep: { col: number; row: number } | null = null;
    for (let attempt = 0; attempt < PATROL_WAYPOINTS.length; attempt++) {
      const wp = PATROL_WAYPOINTS[wpIdx];
      const wpPos = cellToWorldCenter(wp.col, wp.row);
      const distToWp = Math.hypot(wpPos.x - currentState.robotX, wpPos.y - currentState.robotY);

      if (distToWp < 14 || currentState.trueMap[wp.row]?.[wp.col]) {
        wpIdx = (wpIdx + 1) % PATROL_WAYPOINTS.length;
        continue;
      }

      nextCellStep = findNextFreeStep(currCell.col, currCell.row, wp.col, wp.row, currentState.trueMap);
      if (nextCellStep) {
        break;
      }
      // Current waypoint is walled off by user-drawn obstacles; skip to next
      wpIdx = (wpIdx + 1) % PATROL_WAYPOINTS.length;
    }

    let robotX = currentState.robotX;
    let robotY = currentState.robotY;
    let robotTheta = currentState.robotTheta;

    if (nextCellStep) {
      const localTarget = cellToWorldCenter(nextCellStep.col, nextCellStep.row);
      const targetAngle = Math.atan2(localTarget.y - robotY, localTarget.x - robotX);
      let dAngle = targetAngle - robotTheta;
      while (dAngle > Math.PI) dAngle -= 2 * Math.PI;
      while (dAngle < -Math.PI) dAngle += 2 * Math.PI;
      robotTheta = robotTheta + dAngle * 0.28;

      const proposedX = robotX + Math.cos(robotTheta) * speed;
      const proposedY = robotY + Math.sin(robotTheta) * speed;
      const resolved = resolveRobotGridCollision(proposedX, proposedY, currentState.trueMap);
      robotX = resolved.x;
      robotY = resolved.y;
    } else {
      // Completely boxed in by user-drawn walls: rotate in place to scan surroundings
      robotTheta += 0.12;
    }

    const nextLogOdds = currentState.logOdds.map((row) => [...row]);

    const numBeams = params.lidarBeams;
    const maxRange = params.maxRange;
    const lOcc = params.logOddsOcc;
    const lFree = params.logOddsFree;

    const activeRays: LaserRay[] = [];
    let hitsCount = 0;

    for (let b = 0; b < numBeams; b++) {
      const angle = robotTheta + (b / numBeams) * Math.PI * 2;
      const rayResult = castGridRay(robotX, robotY, angle, maxRange, currentState.trueMap);

      for (const { col, row } of rayResult.freeCells) {
        nextLogOdds[row][col] = Math.max(-5.0, nextLogOdds[row][col] - lFree);
      }

      if (rayResult.hitObstacle && rayResult.hitCell) {
        hitsCount++;
        const { col, row } = rayResult.hitCell;
        nextLogOdds[row][col] = Math.min(5.0, nextLogOdds[row][col] + lOcc);
      }

      activeRays.push({
        angle,
        hitX: rayResult.hitX,
        hitY: rayResult.hitY,
        distance: rayResult.distance,
        hitObstacle: rayResult.hitObstacle
      });
    }

    let exploredCells = 0;
    let occupiedCells = 0;
    const totalCells = COLS * ROWS;

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const val = nextLogOdds[r][c];
        if (Math.abs(val) > 0.4) exploredCells++;
        if (val > 0.6) occupiedCells++;
      }
    }

    return {
      nextState: {
        timeStep: nextStep,
        cols: COLS,
        rows: ROWS,
        cellSize: CELL_SIZE,
        robotX,
        robotY,
        robotTheta,
        waypointIdx: wpIdx,
        trueMap: currentState.trueMap,
        logOdds: nextLogOdds,
        activeRays,
        exploredRatio: exploredCells / totalCells,
        occupiedCellCount: occupiedCells,
        activeHitsCount: hitsCount
      },
      isComplete: false,
      activePseudocodeLine: nextStep % 7
    };
  },

  onPointerInteraction(
    currentState: OccupancyGridState,
    event: CanvasPointerEvent
  ): OccupancyGridState {
    const col = Math.floor((event.worldX + HALF_W) / CELL_SIZE);
    const row = Math.floor((event.worldY + HALF_H) / CELL_SIZE);

    if (col >= 0 && col < COLS && row >= 0 && row < ROWS) {
      if (event.shiftKey) {
        if (currentState.trueMap[row][col]) {
          return currentState;
        }
        const pos = cellToWorldCenter(col, row);
        return {
          ...currentState,
          robotX: pos.x,
          robotY: pos.y
        };
      }

      const robotCell = worldToCell(currentState.robotX, currentState.robotY);
      if (col === robotCell.col && row === robotCell.row) {
        return currentState;
      }

      const nextTrueMap = currentState.trueMap.map((r) => [...r]);
      const nextLogOdds = currentState.logOdds.map((r) => [...r]);

      if (col > 0 && col < COLS - 1 && row > 0 && row < ROWS - 1) {
        const isNowSolid = event.isDragging
          ? event.button !== 'right'
          : !nextTrueMap[row][col];
        nextTrueMap[row][col] = isNowSolid;
        nextLogOdds[row][col] = isNowSolid ? 2.5 : -1.5;
      }

      const resolved = resolveRobotGridCollision(currentState.robotX, currentState.robotY, nextTrueMap);

      return {
        ...currentState,
        robotX: resolved.x,
        robotY: resolved.y,
        trueMap: nextTrueMap,
        logOdds: nextLogOdds
      };
    }

    return currentState;
  },

  getMetrics(currentState: OccupancyGridState): MetricReadout[] {
    const pctExplored = (currentState.exploredRatio * 100).toFixed(0);

    return [
      {
        id: 'explored',
        label: 'Map Explored',
        value: `${pctExplored}%`,
        color: currentState.exploredRatio > 0.7 ? 'emerald' : 'amber'
      },
      {
        id: 'occCells',
        label: 'Occupied Cells',
        value: currentState.occupiedCellCount,
        color: 'rose'
      },
      {
        id: 'hits',
        label: 'LiDAR Hits',
        value: currentState.activeHitsCount,
        color: 'amber'
      },
      {
        id: 'robotPose',
        label: 'Robot Pose',
        value: `(${currentState.robotX.toFixed(0)}, ${currentState.robotY.toFixed(0)})`,
        color: 'sky'
      }
    ];
  },

  render(ctx, currentState, params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    const startX = -HALF_W;
    const startY = -HALF_H;

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const px = startX + c * CELL_SIZE;
        const py = startY + r * CELL_SIZE;
        const l = currentState.logOdds[r][c];
        const p = 1.0 / (1.0 + Math.exp(-l));

        if (Math.abs(l) < 0.25) {
          ctx.fillStyle = '#1e293b';
        } else if (l < -0.25) {
          const freeIntensity = Math.min(1.0, Math.abs(l) / 3.0);
          ctx.fillStyle = `rgb(${Math.round(15 * (1 - freeIntensity))}, ${Math.round(23 * (1 - freeIntensity))}, ${Math.round(42 * (1 - freeIntensity))})`;
        } else {
          const occAlpha = Math.min(1.0, 0.35 + (p - 0.5) * 1.3);
          ctx.fillStyle = `rgba(244, 63, 94, ${occAlpha})`;
        }

        ctx.fillRect(px, py, CELL_SIZE, CELL_SIZE);

        ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, CELL_SIZE, CELL_SIZE);
      }
    }

    if (params.showTrueMapOverlay) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (currentState.trueMap[r][c]) {
            const px = startX + c * CELL_SIZE;
            const py = startY + r * CELL_SIZE;
            ctx.strokeRect(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4);
          }
        }
      }
    }

    ctx.lineWidth = 1;
    for (const ray of currentState.activeRays) {
      ctx.beginPath();
      ctx.strokeStyle = ray.hitObstacle
        ? 'rgba(245, 158, 11, 0.65)'
        : 'rgba(245, 158, 11, 0.2)';
      ctx.moveTo(currentState.robotX, currentState.robotY);
      ctx.lineTo(ray.hitX, ray.hitY);
      ctx.stroke();

      if (ray.hitObstacle) {
        ctx.beginPath();
        ctx.arc(ray.hitX, ray.hitY, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
      }
    }

    ctx.beginPath();
    ctx.arc(currentState.robotX, currentState.robotY, ROBOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(currentState.robotX, currentState.robotY);
    ctx.lineTo(
      currentState.robotX + Math.cos(currentState.robotTheta) * 14,
      currentState.robotY + Math.sin(currentState.robotTheta) * 14
    );
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Click/drag to paint walls (Right-click erase); Shift-click free cell to teleport robot', 0, HALF_H + 24);

    ctx.restore();
  }
};
