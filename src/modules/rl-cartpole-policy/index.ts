// FILE: src/modules/rl-cartpole-policy/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface CartPoleState {
  x: number;          // Cart horizontal position in px
  xDot: number;       // Cart velocity in px/s
  theta: number;      // Pole angle in rad (0 = upright vertical)
  thetaDot: number;   // Pole angular velocity in rad/s

  weights: [number, number, number, number]; // [w_x, w_xDot, w_theta, w_thetaDot]
  actionForce: number;                       // Control force u applied to cart
  reward: number;                            // Current timestep reward
  baseline: number;                          // Value baseline for advantage
  tdError: number;                           // Advantage / TD error
  falls: number;                             // Fall counter
  stepsBalanced: number;                     // Current consecutive balance streak
  bestStreak: number;                        // Best balance streak recorded

  phaseHistory: { theta: number; thetaDot: number }[]; // Live (theta, thetaDot) trajectory
  pokeEffect: { x: number; y: number; strength: number } | null;
  iteration: number;
}

export interface CartPoleParams {
  learningRate: number;
  explorationNoise: number;
  poleLength: number;
  gravity: number;
}

const CART_MASS = 1.0;
const POLE_MASS = 0.15;
const TOTAL_MASS = CART_MASS + POLE_MASS;
const CART_FRICTION = 0.18;
const POLE_FRICTION = 0.06;
const MAX_FORCE = 38;
const MAX_FALL_ANGLE = 1.1; // ~63 degrees

// Box-Muller standard normal generator
function sampleGaussian(): number {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export const CartPolePolicyPlugin: AlgorithmPlugin<CartPoleState, CartPoleParams> = {
  id: 'rl-cartpole-policy',
  title: 'Policy Gradient & Cart-Pole Balancing',
  shortDescription:
    'Continuous state-space Inverted Pendulum balanced via policy gradient updates with live (θ, dθ/dt) phase-portrait visualization.',
  category: 'reinforcement-ai',

  controls: [
    {
      id: 'learningRate',
      label: 'Learning Rate (α)',
      type: 'slider',
      min: 0.0,
      max: 0.5,
      step: 0.02,
      defaultValue: 0.15,
      description: 'Policy gradient learning rate α.',
    },
    {
      id: 'explorationNoise',
      label: 'Exploration Noise (σ)',
      type: 'slider',
      min: 0.0,
      max: 0.6,
      step: 0.05,
      defaultValue: 0.15,
      description: 'Stochastic policy exploration sigma.',
    },
    {
      id: 'poleLength',
      label: 'Pole Length',
      type: 'slider',
      min: 50,
      max: 120,
      step: 10,
      defaultValue: 85,
      unit: 'px',
      description: 'Physical length of the inverted pendulum pole.',
    },
    {
      id: 'gravity',
      label: 'Gravity (g)',
      type: 'slider',
      min: 5,
      max: 25,
      step: 2.5,
      defaultValue: 12.5,
      description: 'Gravitational acceleration pulling the pole downward.',
    },
  ],

  documentation: {
    intuition:
      'The inverted pendulum on a cart is the quintessential benchmark for dynamic stabilization and reinforcement learning. By parameterizing a continuous policy π_θ(u | s) over state s = (x, ẋ, θ, θ̇), policy gradients directly maximize expected cumulative return. The right panel displays the live phase portrait (θ, dθ/dt): an uncontrolled pendulum spirals outward into catastrophic failure, whereas the policy contracts the phase trajectory into a stable limit cycle or fixed-point sink at (0, 0).',
    governingEquationLatex:
      '\\Delta \\textcolor{#38bdf8}{\\boldsymbol{\\theta}} = \\alpha \\, (\\textcolor{#10b981}{R_t} - b) \\, \\nabla_{\\boldsymbol{\\theta}} \\log \\pi_{\\boldsymbol{\\theta}}(\\textcolor{#f59e0b}{u_t} \\mid \\textcolor{#38bdf8}{x, \\dot{x}, \\theta, \\dot{\\theta}}) - \\textcolor{#f43f5e}{c_{\\text{fall}}}',
    pseudocode: [
      'function CartPole_Step(state s=(x, ẋ, θ, θ̇), weights w, α, σ):',
      '  action_mean = dot(w, normalize(s))',
      '  force u = tanh(action_mean) + gaussianNoise(0, σ)',
      '  (s_next, reward, fell) = integratePhysics(s, u, dt)',
      '  advantage = computeAdvantage(reward, baseline)',
      '  w = w + α * advantage * gradLogPi(u, action_mean) * s',
      '  if fell: applyFallPenalty(w); resetPoleUpright()',
    ],
  },

  createInitialState(_params: CartPoleParams): CartPoleState {
    return {
      x: 0,
      xDot: 0,
      theta: 0.04,
      thetaDot: 0,
      // Prior weights initialized to stable balance basin [w_x, w_xDot, w_theta, w_thetaDot]
      weights: [0.42, 0.65, 3.10, 1.55],
      actionForce: 0,
      reward: 1.0,
      baseline: 0.9,
      tdError: 0,
      falls: 0,
      stepsBalanced: 0,
      bestStreak: 0,
      phaseHistory: [{ theta: 0.04, thetaDot: 0 }],
      pokeEffect: null,
      iteration: 0,
    };
  },

  step(currentState: CartPoleState, params: CartPoleParams) {
    let { x, xDot, theta, thetaDot, falls, stepsBalanced, bestStreak, baseline } = currentState;
    const weights: [number, number, number, number] = [...currentState.weights];

    // 1. Normalized state representation
    const sNorm: [number, number, number, number] = [
      x / 80,
      xDot / 50,
      theta / 0.28,
      thetaDot / 2.2,
    ];

    // 2. Policy forward pass: compute horizontal force u = tanh(w · s) + noise
    const linearScore =
      weights[0] * sNorm[0] +
      weights[1] * sNorm[1] +
      weights[2] * sNorm[2] +
      weights[3] * sNorm[3];

    const meanForce = MAX_FORCE * Math.tanh(linearScore);
    const noise = sampleGaussian() * params.explorationNoise * MAX_FORCE;
    const actionForce = Math.max(-50, Math.min(50, meanForce + noise));

    // 3. Multi-substep Euler physics integration for numerical fidelity
    const subSteps = 2;
    const dt = 0.025;
    const halfPoleLen = (params.poleLength / 85) * 0.55;
    const g = params.gravity;

    for (let step = 0; step < subSteps; step++) {
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);

      // Track soft bumper rail restoring force at boundary limits
      let railSpring = 0;
      if (Math.abs(x) > 135) {
        railSpring = -2.8 * (x - 135 * Math.sign(x)) - 0.4 * xDot;
      }

      const netForce = actionForce + railSpring;
      const temp = (netForce + POLE_MASS * halfPoleLen * thetaDot * thetaDot * sinT - CART_FRICTION * xDot) / TOTAL_MASS;

      const num = g * sinT - cosT * temp - (POLE_FRICTION * thetaDot) / (POLE_MASS * halfPoleLen);
      const den = halfPoleLen * (4 / 3 - (POLE_MASS * cosT * cosT) / TOTAL_MASS);

      const thetaDDot = num / den;
      const xDDot = temp - (POLE_MASS * halfPoleLen * thetaDDot * cosT) / TOTAL_MASS;

      thetaDot += thetaDDot * dt;
      theta = normalizeAngle(theta + thetaDot * dt);
      xDot += xDDot * dt;
      x += xDot * dt;
    }

    // 4. Reward & advantage computation
    const fell = Math.abs(theta) > MAX_FALL_ANGLE;
    let reward = Math.cos(theta) - 0.25 * Math.pow(x / 100, 2) - 0.05 * Math.pow(thetaDot, 2);

    if (fell) {
      reward = -12.0;
    }

    baseline = 0.95 * baseline + 0.05 * reward;
    const advantage = reward - baseline;

    // 5. Policy gradient weight update
    const alpha = params.learningRate;
    if (alpha > 0) {
      const sigmaSq = params.explorationNoise * params.explorationNoise + 0.05;
      const gradTanh = 1 - Math.pow(Math.tanh(linearScore), 2);
      const deltaU = (actionForce - meanForce) / (MAX_FORCE * sigmaSq);

      for (let i = 0; i < 4; i++) {
        const grad = deltaU * gradTanh * sNorm[i];
        weights[i] += alpha * advantage * grad;
        // Keep weights in reasonable stable bounds
        weights[i] = Math.max(-2, Math.min(8, weights[i]));
      }
    }

    // 6. Handle fall penalty & upright perturbation reset
    let activeLine = 3;
    if (fell) {
      falls += 1;
      activeLine = 6;
      // Proactive corrective adjustment on angle and velocity gain
      weights[2] = Math.max(2.0, weights[2] + 0.12);
      weights[3] = Math.max(1.0, weights[3] + 0.06);

      // Reset pole near upright with small perturbation
      theta = (Math.random() - 0.5) * 0.10;
      thetaDot = (Math.random() - 0.5) * 0.15;
      x *= 0.35;
      xDot = 0;
      stepsBalanced = 0;
    } else {
      stepsBalanced += 1;
      if (stepsBalanced > bestStreak) {
        bestStreak = stepsBalanced;
      }
      activeLine = stepsBalanced % 3 === 0 ? 4 : 5;
    }

    // Update phase portrait history (FIFO 140 points)
    const phaseHistory = [...currentState.phaseHistory, { theta, thetaDot }].slice(-140);
    const pokeEffect = currentState.pokeEffect ? {
      ...currentState.pokeEffect,
      strength: currentState.pokeEffect.strength * 0.85,
    } : null;

    return {
      nextState: {
        x,
        xDot,
        theta,
        thetaDot,
        weights,
        actionForce,
        reward,
        baseline,
        tdError: advantage,
        falls,
        stepsBalanced,
        bestStreak,
        phaseHistory,
        pokeEffect: pokeEffect && pokeEffect.strength > 0.05 ? pokeEffect : null,
        iteration: currentState.iteration + 1,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: CartPoleState): MetricReadout[] {
    const angleDeg = ((state.theta * 180) / Math.PI).toFixed(1);
    const absDeg = Math.abs((state.theta * 180) / Math.PI);
    return [
      {
        id: 'angle',
        label: 'Pole Angle θ',
        value: `${angleDeg}°`,
        color: absDeg < 8 ? 'emerald' : absDeg < 25 ? 'amber' : 'rose',
      },
      {
        id: 'thetaDot',
        label: 'Angular Rate dθ/dt',
        value: state.thetaDot.toFixed(2),
        unit: 'rad/s',
        color: 'sky',
      },
      {
        id: 'force',
        label: 'Policy Force u',
        value: state.actionForce.toFixed(1),
        unit: 'N',
        color: 'amber',
      },
      {
        id: 'streak',
        label: 'Balance Streak',
        value: `${state.stepsBalanced} (${state.bestStreak} max)`,
        color: 'emerald',
      },
      {
        id: 'falls',
        label: 'Falls / Resets',
        value: state.falls,
        color: state.falls > 0 ? 'rose' : 'emerald',
      },
    ];
  },

  onPointerInteraction(
    state: CartPoleState,
    event: CanvasPointerEvent,
    _params: CartPoleParams
  ): CartPoleState {
    // Inject an angular velocity impulse on click/drag to poke the pendulum
    const dir = event.worldX > 0 ? 1 : -1;
    const impulse = event.isDragging ? dir * 0.5 : dir * 3.2;
    const nextThetaDot = state.thetaDot + impulse;

    return {
      ...state,
      thetaDot: nextThetaDot,
      pokeEffect: {
        x: event.worldX,
        y: event.worldY,
        strength: 1.0,
      },
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: CartPoleState,
    params: CartPoleParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();

    // 0. Panel Divider
    ctx.beginPath();
    ctx.moveTo(cx, 16);
    ctx.lineTo(cx, height - 16);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // ==========================================
    // LEFT PANEL: PHYSICAL CART-POLE ENVIRONMENT
    // ==========================================
    const leftCx = cx / 2;
    const leftCy = cy + 24;

    // Header label
    ctx.save();
    ctx.font = '700 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('CART-POLE PHYSICAL SYSTEM', 24, 28);
    ctx.restore();

    // Horizontal Rail
    const railY = leftCy + 36;
    const railHalfW = 145;
    ctx.beginPath();
    ctx.moveTo(leftCx - railHalfW, railY);
    ctx.lineTo(leftCx + railHalfW, railY);
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 4;
    ctx.stroke();

    // Rail tick marks & end bumpers
    for (let dx = -120; dx <= 120; dx += 30) {
      ctx.beginPath();
      ctx.moveTo(leftCx + dx, railY - 3);
      ctx.lineTo(leftCx + dx, railY + 3);
      ctx.strokeStyle = dx === 0 ? '#10b981' : '#64748b';
      ctx.lineWidth = dx === 0 ? 2 : 1;
      ctx.stroke();
    }

    // End stops (Rose)
    for (const endX of [leftCx - railHalfW, leftCx + railHalfW]) {
      ctx.beginPath();
      ctx.arc(endX, railY, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#f43f5e';
      ctx.fill();
    }

    // Upright Target Guide (Emerald Dashed)
    ctx.beginPath();
    ctx.moveTo(leftCx, railY - 10);
    ctx.lineTo(leftCx, railY - params.poleLength - 30);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.28)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.setLineDash([]);

    // Cart Position
    const cartX = leftCx + state.x;
    const cartY = railY - 14;
    const cartW = 56;
    const cartH = 26;

    // Force Vector Arrow (Amber #f59e0b)
    if (Math.abs(state.actionForce) > 1.0) {
      const arrowLen = (state.actionForce / MAX_FORCE) * 36;
      const startX = state.actionForce > 0 ? cartX + cartW / 2 : cartX - cartW / 2;
      const endX = startX + arrowLen;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(startX, cartY);
      ctx.lineTo(endX, cartY);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Arrow head
      const arrowHeadSize = 6;
      const headDir = state.actionForce > 0 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(endX, cartY);
      ctx.lineTo(endX - headDir * arrowHeadSize, cartY - 4);
      ctx.lineTo(endX - headDir * arrowHeadSize, cartY + 4);
      ctx.closePath();
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.restore();
    }

    // Cart Chassis (Sky Blue #0ea5e9)
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(cartX - cartW / 2, cartY - cartH / 2, cartW, cartH, 6);
    ctx.fillStyle = 'rgba(14, 165, 233, 0.32)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Wheels
    for (const wx of [-18, 18]) {
      ctx.beginPath();
      ctx.arc(cartX + wx, railY - 3, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#0f172a';
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    ctx.restore();

    // Inverted Pendulum Pole
    const pivotX = cartX;
    const pivotY = cartY - 8;
    const poleLen = params.poleLength;
    const tipX = pivotX + poleLen * Math.sin(state.theta);
    const tipY = pivotY - poleLen * Math.cos(state.theta);

    // Dynamic pole color based on deviation |theta|
    const absTheta = Math.abs(state.theta);
    let poleColor = '#10b981'; // Emerald
    if (absTheta > 0.45) {
      poleColor = '#f43f5e'; // Rose
    } else if (absTheta > 0.18) {
      poleColor = '#f59e0b'; // Amber
    }

    // Pole Beam
    ctx.save();
    ctx.shadowColor = poleColor;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(tipX, tipY);
    ctx.strokeStyle = poleColor;
    ctx.lineWidth = 4.5;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();

    // Tip Bob
    ctx.beginPath();
    ctx.arc(tipX, tipY, 8, 0, Math.PI * 2);
    ctx.fillStyle = poleColor;
    ctx.fill();
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Pivot Joint
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#f8fafc';
    ctx.fill();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Poke visual indicator ripple
    if (state.pokeEffect) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(state.pokeEffect.x + cx, state.pokeEffect.y + cy, 18 * (1 - state.pokeEffect.strength + 0.2), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(245, 158, 11, ${state.pokeEffect.strength.toFixed(2)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }

    // Left panel HUD hint
    ctx.save();
    ctx.font = '500 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Click/Drag anywhere to poke the pendulum and test dynamic recovery', 24, height - 22);
    ctx.restore();

    // ==========================================
    // RIGHT PANEL: LIVE (θ, dθ/dt) PHASE PORTRAIT & WEIGHTS
    // ==========================================
    const rightCx = cx + cx / 2;
    const rightCy = cy - 35;
    const phaseBoxW = 210;
    const phaseBoxH = 150;

    // Header label
    ctx.save();
    ctx.font = '700 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.fillText('PHASE PORTRAIT (θ, dθ/dt) & LEARNED POLICY', cx + 24, 28);
    ctx.restore();

    // Phase Portrait Box Background & Grid
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.65)';
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(rightCx - phaseBoxW / 2, rightCy - phaseBoxH / 2, phaseBoxW, phaseBoxH);
    ctx.fill();
    ctx.stroke();

    // Coordinate Axes (θ on horizontal, dθ/dt on vertical)
    ctx.beginPath();
    ctx.moveTo(rightCx - phaseBoxW / 2, rightCy);
    ctx.lineTo(rightCx + phaseBoxW / 2, rightCy);
    ctx.moveTo(rightCx, rightCy - phaseBoxH / 2);
    ctx.lineTo(rightCx, rightCy + phaseBoxH / 2);
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Origin Attractor Sink (0, 0)
    ctx.beginPath();
    ctx.arc(rightCx, rightCy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();

    // Axis Labels
    ctx.save();
    ctx.font = '500 10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.fillText('θ (rad)', rightCx + phaseBoxW / 2 - 40, rightCy + 13);
    ctx.fillText('dθ/dt', rightCx + 6, rightCy - phaseBoxH / 2 + 13);
    ctx.restore();

    // Phase Portrait Trajectory Trail
    const maxTheta = 1.2;
    const maxThetaDot = 5.0;
    const mapThetaToX = (t: number) => rightCx + (t / maxTheta) * (phaseBoxW / 2 - 10);
    const mapThetaDotToY = (td: number) => rightCy - (td / maxThetaDot) * (phaseBoxH / 2 - 10);

    if (state.phaseHistory.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(rightCx - phaseBoxW / 2, rightCy - phaseBoxH / 2, phaseBoxW, phaseBoxH);
      ctx.clip();

      for (let i = 1; i < state.phaseHistory.length; i++) {
        const p0 = state.phaseHistory[i - 1];
        const p1 = state.phaseHistory[i];
        const alpha = (i / state.phaseHistory.length) * 0.85;

        ctx.beginPath();
        ctx.moveTo(mapThetaToX(p0.theta), mapThetaDotToY(p0.thetaDot));
        ctx.lineTo(mapThetaToX(p1.theta), mapThetaDotToY(p1.thetaDot));
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha.toFixed(2)})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Current point in phase space (Bright glowing dot)
      const curX = mapThetaToX(state.theta);
      const curY = mapThetaDotToY(state.thetaDot);
      ctx.shadowColor = poleColor;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = poleColor;
      ctx.fill();
      ctx.restore();
    }

    // Learned Policy Weights Gauges
    const weightLabels = ['w_x (pos)', 'w_ẋ (vel)', 'w_θ (ang)', 'w_θ̇ (rate)'];
    const barStartY = rightCy + phaseBoxH / 2 + 22;
    const barW = 110;

    ctx.save();
    ctx.font = '600 11px monospace';

    for (let i = 0; i < 4; i++) {
      const wy = barStartY + i * 20;
      const wVal = state.weights[i];

      // Label
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'left';
      ctx.fillText(weightLabels[i], rightCx - phaseBoxW / 2, wy + 9);

      // Gauge bar background
      const barX = rightCx - 10;
      ctx.fillStyle = 'rgba(30, 41, 59, 0.8)';
      ctx.fillRect(barX, wy, barW, 10);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, wy, barW, 10);

      // Value fill (scale from 0 to 6)
      const fillW = Math.max(2, Math.min(barW, (wVal / 5.5) * barW));
      ctx.fillStyle = i >= 2 ? '#10b981' : '#38bdf8';
      ctx.fillRect(barX, wy, fillW, 10);

      // Numeric value
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'right';
      ctx.fillText(wVal.toFixed(2), barX + barW + 36, wy + 9);
    }
    ctx.restore();

    ctx.restore();
  },
};

export default CartPolePolicyPlugin;
