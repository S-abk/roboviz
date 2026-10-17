// src/modules/rl-value-iteration/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

type ActionDir = 'UP' | 'RIGHT' | 'DOWN' | 'LEFT';

interface RLCell {
  x: number;
  y: number;
  type: 'empty' | 'wall' | 'goal' | 'hazard';
  value: number;
  bestAction: ActionDir;
}

export interface RLState {
  size: number;
  grid: RLCell[][];
  iteration: number;
  maxDelta: number;
  isConverged: boolean;
}

export interface RLParams {
  gamma: number;
  stepReward: number;
}

const ACTIONS: { dir: ActionDir; dx: number; dy: number }[] = [
  { dir: 'UP', dx: 0, dy: -1 },
  { dir: 'RIGHT', dx: 1, dy: 0 },
  { dir: 'DOWN', dx: 0, dy: 1 },
  { dir: 'LEFT', dx: -1, dy: 0 }
];

export const RLValueIterationPlugin: AlgorithmPlugin<RLState, RLParams> = {
  id: 'rl-bellman-value-iteration',
  title: 'RL Grid-World Value Iteration',
  shortDescription: 'Bellman optimality backups propagating expected cumulative rewards and policy arrows.',
  category: 'reinforcement-ai',

  controls: [
    {
      id: 'gamma',
      label: 'Discount Factor (γ)',
      type: 'slider',
      min: 0.1,
      max: 0.99,
      step: 0.05,
      defaultValue: 0.9
    },
    {
      id: 'stepReward',
      label: 'Living Step Penalty R(s)',
      type: 'slider',
      min: -2.0,
      max: 0.0,
      step: 0.1,
      defaultValue: -0.2
    }
  ],

  documentation: {
    intuition:
      'Value Iteration computes the long-term utility V(s) of every state by repeatedly applying the Bellman Equation. Rewards from the +10 Goal propagate backward across the grid, teaching the AI which direction maximizes discounted future reward while avoiding the -10 Hazard.',
    governingEquationLatex:
      'V_{k+1}(s) = \\max_{a} \\left[ \\textcolor{#10b981}{R(s,a)} + \\gamma \\textcolor{#0ea5e9}{V_k(s\')} \\right]',
    pseudocode: [
      '1. Initialize V(s) = 0 for all non-terminal states',
      '2. Repeat until max |V_new(s) - V(s)| < ε:',
      '3.   For every grid state s:',
      '4.     For each action a in {UP, RIGHT, DOWN, LEFT}:',
      '5.       Compute Q(s, a) = R_step + γ · V(next_state)',
      '6.     Set V_new(s) = max_a Q(s, a)',
      '7.     Extract policy π*(s) = argmax_a Q(s, a)'
    ]
  },

  createInitialState(): RLState {
    const size = 6;
    const grid: RLCell[][] = [];
    for (let y = 0; y < size; y++) {
      const row: RLCell[] = [];
      for (let x = 0; x < size; x++) {
        let type: RLCell['type'] = 'empty';
        let value = 0;
        if (x === 5 && y === 0) {
          type = 'goal';
          value = 10;
        } else if (x === 3 && y === 1) {
          type = 'hazard';
          value = -10;
        } else if ((x === 2 && y === 2) || (x === 2 && y === 3)) {
          type = 'wall';
        }
        row.push({ x, y, type, value, bestAction: 'UP' });
      }
      grid.push(row);
    }
    return { size, grid, iteration: 0, maxDelta: Infinity, isConverged: false };
  },

  step(state: RLState, params: RLParams) {
    let maxDelta = 0;
    const nextGrid: RLCell[][] = state.grid.map((row) => row.map((cell) => ({ ...cell })));

    for (let y = 0; y < state.size; y++) {
      for (let x = 0; x < state.size; x++) {
        const cell = state.grid[y][x];
        if (cell.type !== 'empty') continue;

        let bestVal = -Infinity;
        let bestAct: ActionDir = 'UP';

        for (const act of ACTIONS) {
          const nx = x + act.dx;
          const ny = y + act.dy;
          // Bounce off walls or grid borders
          const targetCell =
            nx >= 0 && nx < state.size && ny >= 0 && ny < state.size && state.grid[ny][nx].type !== 'wall'
              ? state.grid[ny][nx]
              : cell;

          const qVal = params.stepReward + params.gamma * targetCell.value;
          if (qVal > bestVal) {
            bestVal = qVal;
            bestAct = act.dir;
          }
        }

        maxDelta = Math.max(maxDelta, Math.abs(bestVal - cell.value));
        nextGrid[y][x].value = bestVal;
        nextGrid[y][x].bestAction = bestAct;
      }
    }

    const isConverged = maxDelta < 0.001;
    return {
      nextState: {
        size: state.size,
        grid: nextGrid,
        iteration: state.iteration + 1,
        maxDelta,
        isConverged
      },
      isComplete: isConverged,
      activePseudocodeLine: isConverged ? 6 : 4
    };
  },

  onPointerInteraction(state: RLState, event: CanvasPointerEvent): RLState {
    const cellSize = 64;
    const total = state.size * cellSize;
    const gx = Math.floor((event.worldX + total / 2) / cellSize);
    const gy = Math.floor((event.worldY + total / 2) / cellSize);

    if (gx >= 0 && gx < state.size && gy >= 0 && gy < state.size && !event.isDragging) {
      const cell = state.grid[gy][gx];
      if (cell.type === 'empty') {
        cell.type = 'wall';
        cell.value = 0;
      } else if (cell.type === 'wall') {
        cell.type = 'empty';
      }
    }
    return state;
  },

  getMetrics(state: RLState): MetricReadout[] {
    return [
      { id: 'iter', label: 'Sweep Iteration', value: state.iteration, color: 'sky' },
      {
        id: 'delta',
        label: 'Max Bellman Δ',
        value: state.maxDelta === Infinity ? '—' : state.maxDelta.toFixed(3),
        color: 'amber'
      },
      {
        id: 'conv',
        label: 'Policy Status',
        value: state.isConverged ? 'Optimal π*' : 'Updating',
        color: state.isConverged ? 'emerald' : 'default'
      }
    ];
  },

  render(ctx, state, _params, viewport) {
    const cellSize = 64;
    const total = state.size * cellSize;
    const startX = (viewport.width - total) / 2;
    const startY = (viewport.height - total) / 2;

    ctx.save();
    ctx.translate(startX, startY);

    for (let y = 0; y < state.size; y++) {
      for (let x = 0; x < state.size; x++) {
        const cell = state.grid[y][x];
        const px = x * cellSize;
        const py = y * cellSize;

        if (cell.type === 'wall') {
          ctx.fillStyle = '#334155';
        } else if (cell.type === 'goal') {
          ctx.fillStyle = 'rgba(16, 185, 129, 0.55)';
        } else if (cell.type === 'hazard') {
          ctx.fillStyle = 'rgba(244, 63, 94, 0.55)';
        } else {
          const norm = Math.max(-1, Math.min(1, cell.value / 10));
          ctx.fillStyle =
            norm >= 0
              ? `rgba(16, 185, 129, ${0.08 + norm * 0.45})`
              : `rgba(244, 63, 94, ${0.08 + Math.abs(norm) * 0.45})`;
        }

        ctx.fillRect(px + 2, py + 2, cellSize - 4, cellSize - 4);

        if (cell.type !== 'wall') {
          ctx.fillStyle = '#f8fafc';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(cell.value.toFixed(1), px + cellSize / 2, py + 20);

          if (cell.type === 'empty') {
            const arrows: Record<ActionDir, string> = {
              UP: '↑',
              RIGHT: '→',
              DOWN: '↓',
              LEFT: '←'
            };
            ctx.fillStyle = '#38bdf8';
            ctx.font = 'bold 18px sans-serif';
            ctx.fillText(arrows[cell.bestAction], px + cellSize / 2, py + 45);
          } else {
            ctx.font = 'bold 10px sans-serif';
            ctx.fillText(cell.type.toUpperCase(), px + cellSize / 2, py + 45);
          }
        }
      }
    }

    ctx.restore();
  }
};
