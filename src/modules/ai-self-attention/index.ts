// FILE: src/modules/ai-self-attention/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface SelfAttentionState {
  tokens: string[];
  queries: number[][];
  keys: number[][];
  values: number[][];
  rawScores: number[][];
  attentionWeights: number[][];
  contextVectors: number[][];
  activeTokenIdx: number;
  phase: number;
  iteration: number;
}

export interface SelfAttentionParams {
  temperature: number;
  causalMask: boolean;
  activeTokenIdx: number;
}

const TOKENS = ['The', 'mobile', 'robot', 'avoids', 'steep', 'cliffs'];

const BASE_QUERIES: number[][] = [
  [0.9, 0.2, 0.1, 0.1],
  [0.8, 0.7, 0.1, 0.2],
  [0.7, 0.8, 0.6, 0.2],
  [0.4, 0.6, 0.9, 0.7],
  [0.1, 0.2, 0.5, 0.95],
  [0.2, 0.3, 0.8, 0.95],
];

const BASE_KEYS: number[][] = [
  [0.3, 0.1, 0.1, 0.1],
  [0.6, 0.8, 0.2, 0.1],
  [0.95, 0.85, 0.4, 0.1],
  [0.3, 0.5, 0.95, 0.5],
  [0.1, 0.2, 0.4, 0.9],
  [0.15, 0.2, 0.75, 0.98],
];

function computeAttentionMatrix(
  queries: number[][],
  keys: number[][],
  temperature: number,
  causalMask: boolean
): { rawScores: number[][]; attentionWeights: number[][]; contextVectors: number[][] } {
  const N = queries.length;
  const dk = queries[0].length;
  const scale = Math.sqrt(dk) * Math.max(0.15, temperature);

  const rawScores: number[][] = [];
  const attentionWeights: number[][] = [];

  for (let i = 0; i < N; i++) {
    const rowScores: number[] = [];
    for (let j = 0; j < N; j++) {
      if (causalMask && j > i) {
        rowScores.push(-1e9);
      } else {
        let dot = 0;
        for (let k = 0; k < dk; k++) {
          dot += queries[i][k] * keys[j][k];
        }
        rowScores.push(dot / scale);
      }
    }
    rawScores.push(rowScores);

    const maxScore = Math.max(...rowScores);
    const exps = rowScores.map((s) => (s <= -1e8 ? 0 : Math.exp(s - maxScore)));
    const sumExp = exps.reduce((acc, val) => acc + val, 0) || 1;
    attentionWeights.push(exps.map((e) => e / sumExp));
  }

  const contextVectors: number[][] = [];
  for (let i = 0; i < N; i++) {
    const ctxVec = [0, 0, 0, 0];
    for (let j = 0; j < N; j++) {
      for (let d = 0; d < 4; d++) {
        ctxVec[d] += attentionWeights[i][j] * BASE_KEYS[j][d];
      }
    }
    contextVectors.push(ctxVec);
  }

  return { rawScores, attentionWeights, contextVectors };
}

export const SelfAttentionPlugin: AlgorithmPlugin<SelfAttentionState, SelfAttentionParams> = {
  id: 'ai-self-attention',
  title: 'Transformer Scaled Dot-Product Self-Attention',
  shortDescription:
    'Interactive 6-token sequence visualizer displaying Query-Key dot-product similarity arcs, temperature-scaled Softmax distributions, and the full 6x6 Attention Matrix heatmap.',
  category: 'reinforcement-ai',

  documentation: {
    intuition:
      'Think of Self-Attention as a soft dictionary lookup: the active Query token (Sky Blue) broadcasts its Query vector qᵢ against every token\'s Key vector kⱼ. High dot-product alignment produces glowing Amber attention arcs, and the resulting Softmax distribution blends Value vectors into the contextualized Emerald output.',
    governingEquationLatex:
      '\\text{Attention}(\\textcolor{#38bdf8}{Q}, \\textcolor{#f59e0b}{K}, \\textcolor{#10b981}{V}) = \\text{softmax}\\!\\left(\\frac{\\textcolor{#38bdf8}{Q}\\textcolor{#f59e0b}{K}^T}{\\sqrt{d_k} \\cdot \\tau} + \\textcolor{#f43f5e}{M}\\right)\\textcolor{#10b981}{V}',
    pseudocode: [
      'function ScaledDotProductAttention(Q, K, V, τ, causalMask):',
      '  select active Query token q_i = Q[i]',
      '  scores[i, j] = (q_i · K[j]) / (sqrt(d_k) * τ)  // Raw alignment',
      '  if causalMask and j > i: scores[i, j] = -∞     // Mask future tokens',
      '  α[i, :] = softmax(scores[i, :])                // Attention weights',
      '  z_i = sum_j ( α[i, j] * V[j] )                 // Context vector output',
    ],
  },

  controls: [
    {
      id: 'temperature',
      label: 'Softmax Temperature (τ)',
      type: 'slider',
      min: 0.15,
      max: 2.0,
      step: 0.05,
      defaultValue: 0.45,
      description: 'Low τ sharpens attention onto the top match; high τ flattens weights uniformly.',
    },
    {
      id: 'activeTokenIdx',
      label: 'Active Query Token Index',
      type: 'slider',
      min: 0,
      max: 5,
      step: 1,
      defaultValue: 3,
      description: 'Select which token acts as the active Query q_i (or click tokens on canvas).',
    },
    {
      id: 'causalMask',
      label: 'Autoregressive Causal Mask',
      type: 'toggle',
      defaultValue: false,
      description: 'Prevent tokens from attending to future positions (j > i).',
    },
  ],

  createInitialState(params: SelfAttentionParams): SelfAttentionState {
    const res = computeAttentionMatrix(
      BASE_QUERIES,
      BASE_KEYS,
      params.temperature,
      params.causalMask
    );
    return {
      tokens: TOKENS,
      queries: BASE_QUERIES,
      keys: BASE_KEYS,
      values: BASE_KEYS,
      rawScores: res.rawScores,
      attentionWeights: res.attentionWeights,
      contextVectors: res.contextVectors,
      activeTokenIdx: params.activeTokenIdx,
      phase: 0,
      iteration: 0,
    };
  },

  step(currentState: SelfAttentionState, params: SelfAttentionParams) {
    const nextPhase = (currentState.phase + 1) % 4;
    const nextTokenIdx =
      nextPhase === 0
        ? (currentState.activeTokenIdx + 1) % TOKENS.length
        : currentState.activeTokenIdx;

    const res = computeAttentionMatrix(
      BASE_QUERIES,
      BASE_KEYS,
      params.temperature,
      params.causalMask
    );

    const phaseToLine = [1, 2, 4, 5];

    return {
      nextState: {
        ...currentState,
        rawScores: res.rawScores,
        attentionWeights: res.attentionWeights,
        contextVectors: res.contextVectors,
        activeTokenIdx: nextTokenIdx,
        phase: nextPhase,
        iteration: currentState.iteration + 1,
      },
      activePseudocodeLine: phaseToLine[nextPhase],
      isComplete: false,
    };
  },

  getMetrics(state: SelfAttentionState, params: SelfAttentionParams): MetricReadout[] {
    const weights = state.attentionWeights[state.activeTokenIdx] || [1, 0, 0, 0, 0, 0];
    const maxWeight = Math.max(...weights);
    const topTargetIdx = weights.indexOf(maxWeight);

    let entropy = 0;
    for (const w of weights) {
      if (w > 1e-6) entropy -= w * Math.log2(w);
    }

    return [
      { id: 'query', label: 'Active Query Token', value: `"${TOKENS[state.activeTokenIdx]}"`, color: 'sky' },
      { id: 'key', label: 'Top Attended Key', value: `"${TOKENS[topTargetIdx]}" (${(maxWeight * 100).toFixed(0)}%)`, color: 'amber' },
      { id: 'ent', label: 'Attention Entropy', value: entropy.toFixed(2), unit: 'bits', color: 'emerald' },
      { id: 'mask', label: 'Attention Mask', value: params.causalMask ? 'Causal (j ≤ i)' : 'Bidirectional', color: params.causalMask ? 'rose' : 'default' },
    ];
  },

  onPointerInteraction(
    state: SelfAttentionState,
    event: CanvasPointerEvent,
    params: SelfAttentionParams
  ): SelfAttentionState {
    const leftX = -255;
    const spacing = 68;

    // 1. Check click on any of the 6 token pills on the left
    for (let i = 0; i < TOKENS.length; i++) {
      const tx = leftX + i * spacing;
      if (Math.abs(event.worldX - tx) < 32 && Math.abs(event.worldY - 62) < 95) {
        const res = computeAttentionMatrix(
          BASE_QUERIES,
          BASE_KEYS,
          params.temperature,
          params.causalMask
        );
        return {
          ...state,
          ...res,
          activeTokenIdx: i,
        };
      }
    }

    // 2. Check click on the 6x6 Attention Matrix rows on the right
    const matX = 132;
    const matY = -75;
    const cell = 24;
    if (
      event.worldX >= matX &&
      event.worldX <= matX + 6 * cell &&
      event.worldY >= matY &&
      event.worldY <= matY + 6 * cell
    ) {
      const clickedRow = Math.floor((event.worldY - matY) / cell);
      if (clickedRow >= 0 && clickedRow < TOKENS.length) {
        const res = computeAttentionMatrix(
          BASE_QUERIES,
          BASE_KEYS,
          params.temperature,
          params.causalMask
        );
        return {
          ...state,
          ...res,
          activeTokenIdx: clickedRow,
        };
      }
    }

    return state;
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: SelfAttentionState,
    params: SelfAttentionParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    const res = computeAttentionMatrix(
      BASE_QUERIES,
      BASE_KEYS,
      params.temperature,
      params.causalMask
    );
    const activeIdx = state.activeTokenIdx;
    const weights = res.attentionWeights[activeIdx];

    ctx.save();
    ctx.translate(cx, cy);

    const leftX = -255;
    const spacing = 68;
    const tokenY = 62;

    // 1. Attention Arcs from Active Query Token i to all Key Tokens j
    const srcX = leftX + activeIdx * spacing;

    for (let j = 0; j < TOKENS.length; j++) {
      const dstX = leftX + j * spacing;
      const w = weights[j];
      const isMasked = params.causalMask && j > activeIdx;

      const midX = (srcX + dstX) / 2;
      const dist = Math.abs(dstX - srcX);
      const peakY = tokenY - 38 - dist * 0.42;

      ctx.beginPath();
      ctx.moveTo(srcX, tokenY - 18);
      ctx.quadraticCurveTo(midX, peakY, dstX, tokenY - 18);

      if (isMasked) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.25;
      } else {
        ctx.strokeStyle = `rgba(245, 158, 11, ${(0.12 + w * 0.88).toFixed(2)})`;
        ctx.lineWidth = 1 + w * 9;
      }
      ctx.stroke();
      ctx.setLineDash([]);

      if (!isMasked && w >= 0.08) {
        ctx.font = '600 10px Inter, system-ui, sans-serif';
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(w * 100)}%`, dstX, tokenY - 26);
      }
    }

    // 2. Token Boxes & Context Output Bars
    for (let i = 0; i < TOKENS.length; i++) {
      const tx = leftX + i * spacing;
      const isQuery = i === activeIdx;
      const w = weights[i];
      const isMasked = params.causalMask && i > activeIdx;

      ctx.beginPath();
      ctx.roundRect(tx - 29, tokenY - 16, 58, 32, 6);
      if (isQuery) {
        ctx.fillStyle = 'rgba(14, 165, 233, 0.32)';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.5;
      } else if (isMasked) {
        ctx.fillStyle = 'rgba(244, 63, 94, 0.14)';
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 1.5;
      } else {
        ctx.fillStyle = `rgba(245, 158, 11, ${(0.08 + w * 0.45).toFixed(2)})`;
        ctx.strokeStyle = w > 0.2 ? '#f59e0b' : '#475569';
        ctx.lineWidth = 1.5;
      }
      ctx.fill();
      ctx.stroke();

      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = isQuery ? '#e0f2fe' : '#f8fafc';
      ctx.textAlign = 'center';
      ctx.fillText(TOKENS[i], tx, tokenY + 4);

      ctx.font = '600 9px Inter, system-ui, sans-serif';
      ctx.fillStyle = isQuery ? '#38bdf8' : isMasked ? '#f43f5e' : '#94a3b8';
      ctx.fillText(isQuery ? 'QUERY q_i' : isMasked ? 'MASKED' : `KEY k_${i}`, tx, tokenY + 30);

      const vec = isQuery ? res.contextVectors[i] : BASE_KEYS[i];
      for (let d = 0; d < 4; d++) {
        const bx = tx - 20 + d * 11;
        const bh = vec[d] * 28;
        ctx.fillStyle = isQuery ? '#10b981' : '#38bdf8';
        ctx.fillRect(bx, tokenY + 68 - bh, 7, bh);
      }
    }

    // 3. 6x6 Full Self-Attention Heatmap Matrix (Right Panel)
    const matX = 132;
    const matY = -75;
    const cell = 24;

    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#e2e8f0';
    ctx.textAlign = 'left';
    ctx.fillText('6×6 Attention Matrix A', matX, matY - 12);

    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const x = matX + c * cell;
        const y = matY + r * cell;
        const w = res.attentionWeights[r][c];
        const masked = params.causalMask && c > r;

        ctx.fillStyle = masked
          ? 'rgba(244, 63, 94, 0.25)'
          : `rgba(245, 158, 11, ${(0.08 + w * 0.9).toFixed(2)})`;
        ctx.fillRect(x, y, cell - 2, cell - 2);

        if (r === activeIdx) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, cell - 2, cell - 2);
        }
      }
    }

    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click any Token Pill or Matrix Row to inspect its Query-Key Self-Attention distribution',
      16,
      24
    );
    ctx.restore();
  },
};

export default SelfAttentionPlugin;
