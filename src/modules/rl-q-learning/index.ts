// FILE: src/modules/rl-q-learning/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface QLearningState {
  qTable: number[][]; // [cellIndex][action 0:Up, 1:Right, 2:Down, 3:Left]
  agentRow: number;
  agentCol: number;
  startRow: number;
  startCol: number;
  goalRow: number;
  goalCol: number;
  cliffs: boolean[];
  lastAction: number;
  lastReward: number;
  lastTdError: number;
  lastWasExploratory: boolean;
  episode: number;
  stepInEpisode: number;
  iteration: number;
}

export interface QLearningParams {
  alpha: number;
  gamma: number;
  epsilon: number;
  stepsPerTick: number;
}

const COLS = 10;
const ROWS = 6;
const CELL_SIZE = 48;
const GRID_W = COLS * CELL_SIZE;
const GRID_H = ROWS * CELL_SIZE;

// Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

function deterministicRand(seed: number): number {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

export const QLearningPlugin: AlgorithmPlugin<QLearningState, QLearningParams> = {
  id: 'rl-q-learning',
  title: 'Tabular Q-Learning & Cliff-Walking Agent',
  shortDescription:
    '10x6 Cliff-Walking Grid-World where every cell renders 4 directional Q(s,a) value triangles and a greedy policy arrow while an ε-greedy agent learns the optimal safe path.',
  category: 'reinforcement-ai',

  documentation: {
    intuition:
      'In the classic Cliff-Walking task, the bottom edge between Start and Goal is a lethal cliff (-100 reward). Every step costs -1, and reaching the Emerald Goal yields +100. Watch the directional triangles inside each grid cell brighten in Emerald as positive reward propagates backward from the Goal through Bellman Temporal-Difference (TD) updates!',
    governingEquationLatex:
      '\\textcolor{#38bdf8}{Q(s,a)} \\leftarrow \\textcolor{#38bdf8}{Q(s,a)} + \\alpha \\Big[ \\textcolor{#10b981}{r} + \\gamma \\max_{a\'} \\textcolor{#f59e0b}{Q(s\',a\')} - \\textcolor{#38bdf8}{Q(s,a)} \\Big], \\quad r_{\\text{cliff}} = \\textcolor{#f43f5e}{-100}',
    pseudocode: [
      'function QLearning_Step(state s, Q, α, γ, ε):',
      '  if rand() < ε: a = randomAction()          // ε-greedy exploration',
      '  else:          a = argmax_a Q(s, a)        // Greedy exploitation',
      '  (s_next, r, terminal) = stepEnvironment(s, a)',
      '  td_target = r + γ * max_{a\'} Q(s_next, a\')',
      '  Q(s, a) = Q(s, a) + α * (td_target - Q(s, a))',
      '  if terminal: resetAgentToStart()',
    ],
  },

  controls: [
    {
      id: 'alpha',
      label: 'Learning Rate (α)',
      type: 'slider',
      min: 0.05,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.5,
      description: 'Step size for Temporal-Difference (TD) updates.',
    },
    {
      id: 'gamma',
      label: 'Discount Factor (γ)',
      type: 'slider',
      min: 0.5,
      max: 0.99,
      step: 0.01,
      defaultValue: 0.95,
      description: 'Weight assigned to future bootstrapped rewards.',
    },
    {
      id: 'epsilon',
      label: 'Exploration Rate (ε)',
      type: 'slider',
      min: 0.0,
      max: 0.5,
      step: 0.02,
      defaultValue: 0.1,
      description: 'Probability of taking an exploratory random action.',
    },
    {
      id: 'stepsPerTick',
      label: 'Agent Steps / Frame',
      type: 'slider',
      min: 1,
      max: 20,
      step: 1,
      defaultValue: 3,
      description: 'Accelerate Q-table convergence by running multiple steps per tick.',
    },
  ],

  createInitialState(_params: QLearningParams): QLearningState {
    const qTable = Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]);
    const cliffs = new Array(ROWS * COLS).fill(false);

    for (let c = 1; c < COLS - 1; c++) {
      cliffs[(ROWS - 1) * COLS + c] = true;
    }

    return {
      qTable,
      agentRow: ROWS - 1,
      agentCol: 0,
      startRow: ROWS - 1,
      startCol: 0,
      goalRow: ROWS - 1,
      goalCol: COLS - 1,
      cliffs,
      lastAction: 0,
      lastReward: 0,
      lastTdError: 0,
      lastWasExploratory: false,
      episode: 1,
      stepInEpisode: 0,
      iteration: 0,
    };
  },

  step(currentState: QLearningState, params: QLearningParams) {
    const qTable = currentState.qTable.map((row) => [...row]);
    const cliffs = [...currentState.cliffs];
    let agentRow = currentState.agentRow;
    let agentCol = currentState.agentCol;
    let episode = currentState.episode;
    let stepInEpisode = currentState.stepInEpisode;
    let lastAction = currentState.lastAction;
    let lastReward = currentState.lastReward;
    let lastTdError = currentState.lastTdError;
    let lastWasExploratory = false;
    let iter = currentState.iteration;

    for (let k = 0; k < params.stepsPerTick; k++) {
      iter++;
      const sIdx = agentRow * COLS + agentCol;
      const rVal = deterministicRand(iter * 31 + stepInEpisode * 17);

      let action = 0;
      if (rVal < params.epsilon) {
        lastWasExploratory = true;
        action = Math.floor(deterministicRand(iter * 73 + 9) * 4);
      } else {
        lastWasExploratory = false;
        const qVals = qTable[sIdx];
        let maxQ = -Infinity;
        for (let a = 0; a < 4; a++) {
          const tieBreak = deterministicRand(iter * 13 + a) * 1e-4;
          if (qVals[a] + tieBreak > maxQ) {
            maxQ = qVals[a] + tieBreak;
            action = a;
          }
        }
      }

      const nextR = Math.max(0, Math.min(ROWS - 1, agentRow + DR[action]));
      const nextC = Math.max(0, Math.min(COLS - 1, agentCol + DC[action]));
      const nextIdx = nextR * COLS + nextC;

      let reward = -1;
      let terminal = false;

      if (cliffs[nextIdx]) {
        reward = -100;
        terminal = true;
      } else if (nextR === currentState.goalRow && nextC === currentState.goalCol) {
        reward = 100;
        terminal = true;
      }

      const maxNextQ = terminal ? 0 : Math.max(...qTable[nextIdx]);
      const tdTarget = reward + params.gamma * maxNextQ;
      const tdError = tdTarget - qTable[sIdx][action];
      qTable[sIdx][action] += params.alpha * tdError;

      lastAction = action;
      lastReward = reward;
      lastTdError = tdError;

      if (terminal || stepInEpisode > 150) {
        agentRow = currentState.startRow;
        agentCol = currentState.startCol;
        episode++;
        stepInEpisode = 0;
      } else {
        agentRow = nextR;
        agentCol = nextC;
        stepInEpisode++;
      }
    }

    const activeLine = lastWasExploratory ? 1 : Math.abs(lastTdError) > 5 ? 5 : 2;

    return {
      nextState: {
        qTable,
        agentRow,
        agentCol,
        startRow: currentState.startRow,
        startCol: currentState.startCol,
        goalRow: currentState.goalRow,
        goalCol: currentState.goalCol,
        cliffs,
        lastAction,
        lastReward,
        lastTdError,
        lastWasExploratory,
        episode,
        stepInEpisode,
        iteration: iter,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: QLearningState): MetricReadout[] {
    return [
      { id: 'ep', label: 'Episode', value: state.episode, color: 'sky' },
      { id: 'td', label: 'Last TD Error δ', value: state.lastTdError.toFixed(2), color: 'amber' },
      { id: 'rew', label: 'Last Reward r', value: state.lastReward, color: state.lastReward > 0 ? 'emerald' : state.lastReward < -10 ? 'rose' : 'default' },
      { id: 'mode', label: 'Policy Action', value: state.lastWasExploratory ? 'Explore (ε)' : 'Greedy (max Q)', color: state.lastWasExploratory ? 'amber' : 'emerald' },
    ];
  },

  onPointerInteraction(
    state: QLearningState,
    event: CanvasPointerEvent,
    _params: QLearningParams
  ): QLearningState {
    if (event.isDragging) return state;

    const localX = event.worldX + GRID_W / 2;
    const localY = event.worldY + GRID_H / 2;
    const col = Math.floor(localX / CELL_SIZE);
    const row = Math.floor(localY / CELL_SIZE);

    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return state;

    if (
      (row === state.startRow && col === state.startCol) ||
      (row === state.goalRow && col === state.goalCol)
    ) {
      return state;
    }

    const idx = row * COLS + col;

    // Shift-click or Right-click toggles cliff hazard; normal click teleports agent
    if (event.shiftKey || event.button === 'right') {
      const cliffs = [...state.cliffs];
      cliffs[idx] = !cliffs[idx];
      return { ...state, cliffs };
    }

    if (state.cliffs[idx]) {
      const cliffs = [...state.cliffs];
      cliffs[idx] = false;
      return { ...state, cliffs };
    }

    return {
      ...state,
      agentRow: row,
      agentCol: col,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: QLearningState,
    _params: QLearningParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const startX = (width - GRID_W) / 2;
    const startY = (height - GRID_H) / 2;

    ctx.save();
    ctx.translate(startX, startY);

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const idx = r * COLS + c;
        const x = c * CELL_SIZE;
        const y = r * CELL_SIZE;
        const cx = x + CELL_SIZE / 2;
        const cy = y + CELL_SIZE / 2;

        if (state.cliffs[idx]) {
          ctx.fillStyle = 'rgba(244, 63, 94, 0.28)';
          ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
          ctx.strokeStyle = '#f43f5e';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

          ctx.font = '700 10px Inter, system-ui, sans-serif';
          ctx.fillStyle = '#f43f5e';
          ctx.textAlign = 'center';
          ctx.fillText('CLIFF', cx, cy + 4);
          continue;
        }

        if (r === state.goalRow && c === state.goalCol) {
          ctx.fillStyle = 'rgba(16, 185, 129, 0.32)';
          ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

          ctx.font = '700 10px Inter, system-ui, sans-serif';
          ctx.fillStyle = '#10b981';
          ctx.textAlign = 'center';
          ctx.fillText('GOAL', cx, cy + 4);
          continue;
        }

        const qVals = state.qTable[idx];
        const corners = [
          [x, y, x + CELL_SIZE, y], // 0: Up
          [x + CELL_SIZE, y, x + CELL_SIZE, y + CELL_SIZE], // 1: Right
          [x + CELL_SIZE, y + CELL_SIZE, x, y + CELL_SIZE], // 2: Down
          [x, y + CELL_SIZE, x, y], // 3: Left
        ];

        let bestA = 0;
        let maxQ = -Infinity;
        let anyNonZero = false;

        for (let a = 0; a < 4; a++) {
          const q = qVals[a];
          if (Math.abs(q) > 0.05) anyNonZero = true;
          if (q > maxQ) {
            maxQ = q;
            bestA = a;
          }

          const [x1, y1, x2, y2] = corners[a];
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.closePath();

          if (q > 0.5) {
            const norm = Math.min(1, q / 85);
            ctx.fillStyle = `rgba(16, 185, 129, ${(0.12 + norm * 0.65).toFixed(2)})`;
          } else if (q < -0.5) {
            const norm = Math.min(1, Math.abs(q) / 90);
            ctx.fillStyle = `rgba(244, 63, 94, ${(0.1 + norm * 0.55).toFixed(2)})`;
          } else {
            ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
          }
          ctx.fill();
          ctx.strokeStyle = 'rgba(51, 65, 85, 0.55)';
          ctx.lineWidth = 0.75;
          ctx.stroke();
        }

        if (anyNonZero) {
          const arrowAngles = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(arrowAngles[bestA]);
          ctx.beginPath();
          ctx.moveTo(-5, 0);
          ctx.lineTo(6, 0);
          ctx.lineTo(2, -3.5);
          ctx.moveTo(6, 0);
          ctx.lineTo(2, 3.5);
          ctx.strokeStyle = maxQ > 0 ? '#10b981' : '#38bdf8';
          ctx.lineWidth = 1.6;
          ctx.stroke();
          ctx.restore();
        }

        if (r === state.startRow && c === state.startCol) {
          ctx.font = '700 9px Inter, system-ui, sans-serif';
          ctx.fillStyle = '#38bdf8';
          ctx.textAlign = 'center';
          ctx.fillText('START', cx, y + 11);
        }
      }
    }

    // Agent Token (Sky Blue or Amber when exploring)
    const ax = state.agentCol * CELL_SIZE + CELL_SIZE / 2;
    const ay = state.agentRow * CELL_SIZE + CELL_SIZE / 2;

    ctx.beginPath();
    ctx.arc(ax, ay, 12, 0, Math.PI * 2);
    ctx.fillStyle = state.lastWasExploratory ? '#f59e0b' : '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click any cell to teleport Agent | Shift+Click (or Right-Click) to toggle Cliff Hazards',
      16,
      24
    );
    ctx.restore();
  },
};

export default QLearningPlugin;
