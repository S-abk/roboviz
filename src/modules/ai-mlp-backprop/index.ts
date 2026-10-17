// FILE: src/modules/ai-mlp-backprop/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface DataPoint2D {
  x: number; // Normalized in [-1, 1]
  y: number; // Normalized in [-1, 1]
  label: 0 | 1;
}

export interface MLPBackpropState {
  w1: number[][]; // [hiddenUnits][2]
  b1: number[];   // [hiddenUnits]
  w2: number[];   // [hiddenUnits]
  b2: number;
  points: DataPoint2D[];
  loss: number;
  accuracy: number;
  epoch: number;
  phase: number;
}

export interface MLPBackpropParams {
  learningRate: number;
  hiddenUnits: number;
  l2Reg: number;
  stepsPerFrame: number;
  dataset: string;
}

function sigmoid(z: number): number {
  if (z > 15) return 1;
  if (z < -15) return 0;
  return 1 / (1 + Math.exp(-z));
}

function forwardMLP(
  x: number,
  y: number,
  w1: number[][],
  b1: number[],
  w2: number[],
  b2: number
): { h: number[]; prob: number } {
  const H = b1.length;
  const h = new Array(H);
  let z2 = b2;
  for (let i = 0; i < H; i++) {
    const z1 = w1[i][0] * x + w1[i][1] * y + b1[i];
    const act = Math.tanh(z1);
    h[i] = act;
    z2 += w2[i] * act;
  }
  return { h, prob: sigmoid(z2) };
}

function initWeights(hiddenUnits: number): {
  w1: number[][];
  b1: number[];
  w2: number[];
  b2: number;
} {
  const w1: number[][] = [];
  const b1: number[] = [];
  const w2: number[] = [];

  for (let i = 0; i < hiddenUnits; i++) {
    const angle = (i / hiddenUnits) * Math.PI * 2 + 0.35;
    w1.push([Math.cos(angle) * 2.2, Math.sin(angle) * 2.2]);
    b1.push((i % 2 === 0 ? 0.5 : -0.5) * 0.8);
    w2.push(i % 2 === 0 ? 1.4 : -1.4);
  }

  return { w1, b1, w2, b2: 0.0 };
}

function buildDataset(dataset: string): DataPoint2D[] {
  if (dataset === 'circle') {
    const pts: DataPoint2D[] = [];
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      pts.push({ x: 0.25 * Math.cos(a), y: 0.25 * Math.sin(a), label: 1 });
    }
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      pts.push({ x: 0.72 * Math.cos(a), y: 0.72 * Math.sin(a), label: 0 });
    }
    return pts;
  }

  // Default XOR + Center Pocket
  return [
    { x: -0.55, y: -0.55, label: 1 },
    { x: -0.42, y: -0.35, label: 1 },
    { x: 0.55, y: 0.55, label: 1 },
    { x: 0.4, y: 0.38, label: 1 },
    { x: 0.0, y: 0.0, label: 1 },
    { x: -0.55, y: 0.55, label: 0 },
    { x: -0.38, y: 0.42, label: 0 },
    { x: 0.55, y: -0.55, label: 0 },
    { x: 0.42, y: -0.38, label: 0 },
    { x: -0.68, y: 0.0, label: 0 },
    { x: 0.68, y: 0.0, label: 0 },
    { x: 0.0, y: 0.68, label: 0 },
    { x: 0.0, y: -0.68, label: 0 },
  ];
}

export const MLPBackpropPlugin: AlgorithmPlugin<MLPBackpropState, MLPBackpropParams> = {
  id: 'ai-mlp-backprop',
  title: 'Neural Network Decision Boundary & Backpropagation',
  shortDescription:
    '2-Layer Multi-Layer Perceptron (MLP) trained live via backpropagation and gradient descent on non-linear 2D data points, warping a continuous probability field in real time.',
  category: 'reinforcement-ai',

  documentation: {
    intuition:
      'A single linear classifier can only slice the plane with a straight line. By combining H hidden tanh neurons, the network bends and stitches multiple half-planes into a curved non-linear decision boundary (Amber contour) separating Class 1 (Emerald) from Class 0 (Rose). Click anywhere to inject new training points and watch the boundary warp!',
    governingEquationLatex:
      '\\hat{y} = \\sigma\\!\\Big(\\textcolor{#10b981}{\\mathbf{W}_2}\\,\\tanh(\\textcolor{#38bdf8}{\\mathbf{W}_1}\\mathbf{x} + \\mathbf{b}_1) + b_2\\Big), \\quad \\theta \\leftarrow \\theta - \\eta \\textcolor{#f59e0b}{\\nabla_\\theta \\mathcal{L}(\\textcolor{#f43f5e}{y}, \\hat{y})}',
    pseudocode: [
      'function TrainMLP_Step(dataset X, Y, weights W1, b1, W2, b2, η):',
      '  for each (x, y) in dataset:',
      '    h = tanh(W1 · x + b1)                    // Hidden layer forward pass',
      '    y_hat = sigmoid(W2 · h + b2)             // Output probability',
      '    dz2 = y_hat - y                          // Output error gradient',
      '    dW2 += dz2 * h^T;  dh = (W2^T * dz2) ⊙ (1 - h^2)',
      '    dW1 += dh * x^T;   updateWeights(η)      // Gradient descent step',
    ],
  },

  controls: [
    {
      id: 'learningRate',
      label: 'Learning Rate (η)',
      type: 'slider',
      min: 0.02,
      max: 0.6,
      step: 0.02,
      defaultValue: 0.18,
      description: 'Gradient descent step size.',
    },
    {
      id: 'hiddenUnits',
      label: 'Hidden Neurons (H)',
      type: 'slider',
      min: 2,
      max: 10,
      step: 1,
      defaultValue: 6,
      description: 'Capacity of the hidden tanh layer (2 = simple bend, 6+ = complex pockets).',
    },
    {
      id: 'l2Reg',
      label: 'L2 Weight Decay (λ)',
      type: 'slider',
      min: 0.0,
      max: 0.02,
      step: 0.002,
      defaultValue: 0.002,
      description: 'Regularization penalty preventing sharp overfitting.',
    },
    {
      id: 'stepsPerFrame',
      label: 'Epochs / Frame',
      type: 'slider',
      min: 1,
      max: 25,
      step: 2,
      defaultValue: 9,
      description: 'Number of full-batch gradient descent updates per simulation frame.',
    },
    {
      id: 'dataset',
      label: 'Initial Dataset Pattern',
      type: 'select',
      options: [
        { label: 'Non-Linear XOR Saddle', value: 'xor' },
        { label: 'Concentric Ring Boundary', value: 'circle' },
      ],
      defaultValue: 'xor',
    },
  ],

  createInitialState(params: MLPBackpropParams): MLPBackpropState {
    const net = initWeights(params.hiddenUnits);
    const points = buildDataset(params.dataset);

    return {
      ...net,
      points,
      loss: 0.693,
      accuracy: 50,
      epoch: 0,
      phase: 0,
    };
  },

  step(currentState: MLPBackpropState, params: MLPBackpropParams) {
    let w1 = currentState.w1.map((row) => [...row]);
    let b1 = [...currentState.b1];
    let w2 = [...currentState.w2];
    let b2 = currentState.b2;

    if (b1.length !== params.hiddenUnits) {
      const reinit = initWeights(params.hiddenUnits);
      w1 = reinit.w1;
      b1 = reinit.b1;
      w2 = reinit.w2;
      b2 = reinit.b2;
    }

    const H = b1.length;
    const N = Math.max(1, currentState.points.length);
    let totalLoss = 0;
    let correct = 0;

    for (let stepIdx = 0; stepIdx < params.stepsPerFrame; stepIdx++) {
      const dw1 = Array.from({ length: H }, () => [0, 0]);
      const db1 = new Array(H).fill(0);
      const dw2 = new Array(H).fill(0);
      let db2 = 0;

      totalLoss = 0;
      correct = 0;

      for (const pt of currentState.points) {
        const { h, prob } = forwardMLP(pt.x, pt.y, w1, b1, w2, b2);
        const y = pt.label;
        const eps = 1e-6;
        totalLoss += -(y * Math.log(prob + eps) + (1 - y) * Math.log(1 - prob + eps));
        if ((prob >= 0.5 ? 1 : 0) === y) correct++;

        const dz2 = prob - y;
        db2 += dz2;

        for (let i = 0; i < H; i++) {
          dw2[i] += dz2 * h[i];
          const dz1 = dz2 * w2[i] * (1 - h[i] * h[i]);
          dw1[i][0] += dz1 * pt.x;
          dw1[i][1] += dz1 * pt.y;
          db1[i] += dz1;
        }
      }

      const lr = params.learningRate / N;
      b2 -= lr * db2;
      for (let i = 0; i < H; i++) {
        w2[i] = w2[i] * (1 - params.learningRate * params.l2Reg) - lr * dw2[i];
        w1[i][0] = w1[i][0] * (1 - params.learningRate * params.l2Reg) - lr * dw1[i][0];
        w1[i][1] = w1[i][1] * (1 - params.learningRate * params.l2Reg) - lr * dw1[i][1];
        b1[i] -= lr * db1[i];
      }
    }

    const avgLoss = totalLoss / N;
    const accuracy = (correct / N) * 100;
    const nextPhase = (currentState.phase + 1) % 4;
    const phaseLines = [2, 3, 5, 6];

    return {
      nextState: {
        w1,
        b1,
        w2,
        b2,
        points: currentState.points.map((p) => ({ ...p })),
        loss: avgLoss,
        accuracy,
        epoch: currentState.epoch + params.stepsPerFrame,
        phase: nextPhase,
      },
      activePseudocodeLine: phaseLines[nextPhase],
      isComplete: false,
    };
  },

  getMetrics(state: MLPBackpropState): MetricReadout[] {
    return [
      { id: 'loss', label: 'BCE Loss L(θ)', value: state.loss.toFixed(4), color: state.loss < 0.2 ? 'emerald' : 'amber' },
      { id: 'acc', label: 'Accuracy', value: state.accuracy.toFixed(1), unit: '%', color: state.accuracy >= 95 ? 'emerald' : 'sky' },
      { id: 'epoch', label: 'Training Epochs', value: state.epoch, color: 'sky' },
      { id: 'pts', label: 'Training Points', value: state.points.length, color: 'default' },
    ];
  },

  onPointerInteraction(
    state: MLPBackpropState,
    event: CanvasPointerEvent,
    _params: MLPBackpropParams
  ): MLPBackpropState {
    if (event.isDragging) return state;

    const scaleX = 215;
    const scaleY = 145;
    const nx = Math.max(-0.95, Math.min(0.95, event.worldX / scaleX));
    const ny = Math.max(-0.95, Math.min(0.95, event.worldY / scaleY));

    const label: 0 | 1 = event.shiftKey || event.button === 'right' ? 0 : 1;

    return {
      ...state,
      points: [...state.points, { x: nx, y: ny, label }],
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: MLPBackpropState,
    _params: MLPBackpropParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;
    const scaleX = 215;
    const scaleY = 145;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Continuous 2D Probability Heatmap Field
    const gridCols = 42;
    const gridRows = 28;
    const cellW = (scaleX * 2) / gridCols;
    const cellH = (scaleY * 2) / gridRows;

    for (let r = 0; r < gridRows; r++) {
      const ny = -1 + ((r + 0.5) / gridRows) * 2;
      for (let c = 0; c < gridCols; c++) {
        const nx = -1 + ((c + 0.5) / gridCols) * 2;
        const { prob } = forwardMLP(nx, ny, state.w1, state.b1, state.w2, state.b2);

        const px = -scaleX + c * cellW;
        const py = -scaleY + r * cellH;

        if (Math.abs(prob - 0.5) < 0.065) {
          ctx.fillStyle = 'rgba(245, 158, 11, 0.52)';
        } else if (prob >= 0.5) {
          const intensity = (prob - 0.5) * 2;
          ctx.fillStyle = `rgba(16, 185, 129, ${(0.08 + intensity * 0.34).toFixed(2)})`;
        } else {
          const intensity = (0.5 - prob) * 2;
          ctx.fillStyle = `rgba(244, 63, 94, ${(0.08 + intensity * 0.34).toFixed(2)})`;
        }
        ctx.fillRect(px, py, cellW + 0.8, cellH + 0.8);
      }
    }

    // 2. First-Layer Hidden Hyperplane Lines (Sky Blue Dashed)
    ctx.save();
    ctx.beginPath();
    ctx.rect(-scaleX, -scaleY, scaleX * 2, scaleY * 2);
    ctx.clip();

    for (let i = 0; i < state.b1.length; i++) {
      const wx = state.w1[i][0];
      const wy = state.w1[i][1];
      const b = state.b1[i];
      if (Math.abs(wy) > 0.05) {
        const yLeft = (-b - wx * -1) / wy;
        const yRight = (-b - wx * 1) / wy;
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(-scaleX, yLeft * scaleY);
        ctx.lineTo(scaleX, yRight * scaleY);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.32)';
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.restore();

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(-scaleX, -scaleY, scaleX * 2, scaleY * 2);

    // 3. Training Data Points (Emerald = Class 1, Rose = Class 0)
    for (const pt of state.points) {
      const px = pt.x * scaleX;
      const py = pt.y * scaleY;
      const { prob } = forwardMLP(pt.x, pt.y, state.w1, state.b1, state.w2, state.b2);
      const isCorrect = (prob >= 0.5 ? 1 : 0) === pt.label;

      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 1 ? '#10b981' : '#f43f5e';
      ctx.fill();
      ctx.strokeStyle = isCorrect ? '#ffffff' : '#f59e0b';
      ctx.lineWidth = isCorrect ? 2 : 3;
      ctx.stroke();
    }

    ctx.restore();

    // HUD
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Left-Click inside plot to add Class 1 (Emerald) | Right-Click or Shift+Click to add Class 0 (Rose)',
      16,
      24
    );
    ctx.restore();
  },
};

export default MLPBackpropPlugin;
