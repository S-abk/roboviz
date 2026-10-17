// FILE: src/modules/control-pid/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface PIDHistorySample {
  t: number;
  y: number;
  setpoint: number;
  pTerm: number;
  iTerm: number;
  dTerm: number;
}

export interface PIDState {
  y: number;
  vy: number;
  setpoint: number;
  integralError: number;
  prevError: number;
  pTerm: number;
  iTerm: number;
  dTerm: number;
  controlSignal: number;
  history: PIDHistorySample[];
  time: number;
  iteration: number;
}

export interface PIDParams {
  kp: number;
  ki: number;
  kd: number;
  windDisturbance: number;
  droneMass: number;
}

const DT = 0.04;
const GRAVITY = 35;

export const PIDControlPlugin: AlgorithmPlugin<PIDState, PIDParams> = {
  id: 'control-pid',
  title: 'PID Controller & Disturbance Rejection',
  shortDescription:
    'Split-screen 1D quadrotor altitude controller with live scrolling oscilloscope chart, P/I/D term breakdown, and wind disturbance rejection.',
  category: 'kinematics-control',

  documentation: {
    intuition:
      'Think of a quadrotor trying to hover at a target altitude under gravity and wind: Kp acts like a virtual spring pulling toward the target; Kd acts like a viscous shock absorber braking vertical velocity to prevent overshoot; and Ki accumulates steady-state offset to cancel persistent gravity droop and downdrafts.',
    governingEquationLatex:
      'u(t) = \\textcolor{#38bdf8}{K_p \\, e(t)} + \\textcolor{#f59e0b}{K_i \\int_0^t e(\\tau)\\,d\\tau} + \\textcolor{#10b981}{K_d \\frac{de(t)}{dt}}, \\quad \\textcolor{#f43f5e}{e(t)} = \\textcolor{#10b981}{r(t)} - \\textcolor{#0ea5e9}{y(t)}',
    pseudocode: [
      'function PID_Step(setpoint r, altitude y, dt):',
      '  error = r - y',
      '  P_term = Kp * error                           // Proportional spring',
      '  integral = clamp(integral + error * dt)       // Anti-windup accumulator',
      '  I_term = Ki * integral                        // Steady-state bias cancellation',
      '  D_term = Kd * (error - prev_error) / dt       // Velocity damping',
      '  u = P_term + I_term + D_term; applyDronePhysics(u, gravity, wind)',
    ],
  },

  controls: [
    {
      id: 'kp',
      label: 'Proportional Gain (Kp)',
      type: 'slider',
      min: 0.0,
      max: 6.0,
      step: 0.2,
      defaultValue: 2.4,
      description: 'Spring stiffness toward target altitude.',
    },
    {
      id: 'ki',
      label: 'Integral Gain (Ki)',
      type: 'slider',
      min: 0.0,
      max: 4.0,
      step: 0.1,
      defaultValue: 1.1,
      description: 'Accumulates error over time to eliminate gravity droop and wind offset.',
    },
    {
      id: 'kd',
      label: 'Derivative Gain (Kd)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.2,
      defaultValue: 1.8,
      description: 'Damps vertical velocity to suppress overshoot and ringing.',
    },
    {
      id: 'windDisturbance',
      label: 'Wind Downdraft Force',
      type: 'slider',
      min: -45,
      max: 45,
      step: 5,
      defaultValue: 0,
      unit: 'N',
      description: 'Constant external vertical wind force acting on the drone.',
    },
    {
      id: 'droneMass',
      label: 'Quadrotor Mass (m)',
      type: 'slider',
      min: 0.6,
      max: 2.5,
      step: 0.1,
      defaultValue: 1.0,
      unit: 'kg',
      description: 'Inertial mass of the vehicle (higher mass = slower response & larger gravity load).',
    },
  ],

  createInitialState(_params: PIDParams): PIDState {
    const initialY = 85;
    const setpoint = -45;
    const initialErr = setpoint - initialY;
    return {
      y: initialY,
      vy: 0,
      setpoint,
      integralError: 0,
      prevError: initialErr,
      pTerm: 0,
      iTerm: 0,
      dTerm: 0,
      controlSignal: 0,
      history: [{ t: 0, y: initialY, setpoint, pTerm: 0, iTerm: 0, dTerm: 0 }],
      time: 0,
      iteration: 0,
    };
  },

  step(currentState: PIDState, params: PIDParams) {
    const error = currentState.setpoint - currentState.y;

    // Anti-windup clamped integral
    const integralError = Math.max(-180, Math.min(180, currentState.integralError + error * DT));

    // Derivative on measurement (-vy) avoids derivative kick on step setpoint changes
    const dError = -currentState.vy;

    const pTerm = params.kp * error;
    const iTerm = params.ki * integralError;
    const dTerm = params.kd * dError;

    const controlSignal = pTerm + iTerm + dTerm;

    // Without integral action (Ki = 0), gravity causes a realistic steady-state droop below the setpoint!
    const netForce =
      controlSignal + params.droneMass * GRAVITY - params.windDisturbance - 0.35 * currentState.vy;
    const ay = netForce / Math.max(0.2, params.droneMass);

    let vy = currentState.vy + ay * DT;
    let y = currentState.y + vy * DT;

    if (y > 155) {
      y = 155;
      vy = 0;
    } else if (y < -155) {
      y = -155;
      vy = 0;
    }

    const time = currentState.time + DT;
    const history = [
      ...currentState.history,
      { t: time, y, setpoint: currentState.setpoint, pTerm, iTerm, dTerm },
    ].slice(-200);

    const nextIter = currentState.iteration + 1;
    const activeLine = Math.abs(error) > 25 ? 2 : Math.abs(vy) > 15 ? 5 : 6;

    return {
      nextState: {
        y,
        vy,
        setpoint: currentState.setpoint,
        integralError,
        prevError: error,
        pTerm,
        iTerm,
        dTerm,
        controlSignal,
        history,
        time,
        iteration: nextIter,
      },
      activePseudocodeLine: activeLine,
      isComplete: false,
    };
  },

  getMetrics(state: PIDState): MetricReadout[] {
    const error = state.setpoint - state.y;
    return [
      { id: 'err', label: 'Tracking Error e(t)', value: (-error).toFixed(1), unit: 'px', color: 'rose' },
      { id: 'p', label: 'P-Term Effort', value: (-state.pTerm).toFixed(0), color: 'sky' },
      { id: 'i', label: 'I-Term Effort', value: (-state.iTerm).toFixed(0), color: 'amber' },
      { id: 'd', label: 'D-Term Effort', value: (-state.dTerm).toFixed(0), color: 'emerald' },
    ];
  },

  onPointerInteraction(
    state: PIDState,
    event: CanvasPointerEvent,
    _params: PIDParams
  ): PIDState {
    const clampedY = Math.max(-145, Math.min(145, event.worldY));

    if (event.shiftKey || event.button === 'right' || (event.worldX > -210 && event.worldX < -95)) {
      return {
        ...state,
        y: clampedY,
        vy: 0,
      };
    }

    return {
      ...state,
      setpoint: clampedY,
    };
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: PIDState,
    params: PIDParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const { width, height } = viewport;
    const cx = width / 2;
    const cy = height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    const splitX = -55;
    ctx.beginPath();
    ctx.moveTo(splitX, -height / 2 + 15);
    ctx.lineTo(splitX, height / 2 - 15);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // =========================================================================
    // LEFT PANEL: PHYSICAL QUADROTOR & ALTITUDE COLUMN
    // =========================================================================
    const droneX = -175;

    for (let ry = -140; ry <= 140; ry += 35) {
      ctx.beginPath();
      ctx.moveTo(droneX - 75, ry);
      ctx.lineTo(droneX - 65, ry);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (Math.abs(params.windDisturbance) > 0) {
      const windDir = params.windDisturbance > 0 ? -1 : 1;
      ctx.font = '600 10px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#f59e0b';
      ctx.fillText(
        `Wind: ${params.windDisturbance > 0 ? 'Updraft ↑' : 'Downdraft ↓'} (${Math.abs(params.windDisturbance)}N)`,
        droneX - 62,
        -152
      );
      for (let wx = -235; wx <= -115; wx += 40) {
        ctx.beginPath();
        ctx.moveTo(wx, -135);
        ctx.lineTo(wx, -135 + windDir * 18);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Setpoint Horizontal Line (Emerald)
    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.moveTo(-width / 2 + 20, state.setpoint);
    ctx.lineTo(splitX - 10, state.setpoint);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.fillText('Setpoint r(t)', -width / 2 + 24, state.setpoint - 7);

    // Error Vector e(t) between Drone and Setpoint (Rose)
    if (Math.abs(state.setpoint - state.y) > 4) {
      ctx.beginPath();
      ctx.moveTo(droneX + 52, state.y);
      ctx.lineTo(droneX + 52, state.setpoint);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.font = '600 10px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.fillText('e(t)', droneX + 58, (state.y + state.setpoint) / 2 + 4);
    }

    // Thruster Plume (Amber)
    const upwardThrust = Math.max(
      4,
      Math.min(48, (-state.controlSignal + params.droneMass * GRAVITY) * 0.35)
    );
    ctx.beginPath();
    ctx.moveTo(droneX - 14, state.y + 7);
    ctx.lineTo(droneX, state.y + 7 + upwardThrust);
    ctx.lineTo(droneX + 14, state.y + 7);
    ctx.closePath();
    ctx.fillStyle = 'rgba(245, 158, 11, 0.75)';
    ctx.fill();

    // Quadrotor Body (Sky Blue)
    ctx.beginPath();
    ctx.roundRect(droneX - 28, state.y - 7, 56, 14, 5);
    ctx.fillStyle = '#0ea5e9';
    ctx.fill();
    ctx.strokeStyle = '#e0f2fe';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Rotor Guard Arms
    ctx.beginPath();
    ctx.moveTo(droneX - 38, state.y - 10);
    ctx.lineTo(droneX - 14, state.y - 10);
    ctx.moveTo(droneX + 14, state.y - 10);
    ctx.lineTo(droneX + 38, state.y - 10);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.stroke();

    // =========================================================================
    // RIGHT PANEL: SCROLLING OSCILLOSCOPE CHART
    // =========================================================================
    const chartLeft = splitX + 20;
    const chartRight = width / 2 - 20;
    const chartWidth = chartRight - chartLeft;

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.lineWidth = 1;
    for (let gy = -120; gy <= 120; gy += 60) {
      ctx.beginPath();
      ctx.moveTo(chartLeft, gy);
      ctx.lineTo(chartRight, gy);
      ctx.stroke();
    }

    if (state.history.length > 1) {
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      for (let i = 0; i < state.history.length; i++) {
        const px = chartLeft + (i / 199) * chartWidth;
        const py = state.history[i].setpoint;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      for (let i = 0; i < state.history.length; i++) {
        const px = chartLeft + (i / 199) * chartWidth;
        const py = state.history[i].y;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Live P / I / D Effort Bar Gauges
    const barBaseY = 148;
    const terms = [
      { label: 'P', val: -state.pTerm, color: '#38bdf8', x: chartLeft + 35 },
      { label: 'I', val: -state.iTerm, color: '#f59e0b', x: chartLeft + 85 },
      { label: 'D', val: -state.dTerm, color: '#10b981', x: chartLeft + 135 },
    ];

    for (const term of terms) {
      const barH = Math.max(-45, Math.min(45, term.val * 0.22));
      ctx.fillStyle = term.color;
      ctx.fillRect(term.x - 10, barBaseY - Math.max(0, barH), 20, Math.abs(barH) + 2);
      ctx.font = '600 11px Inter, system-ui, sans-serif';
      ctx.fillText(term.label, term.x - 4, barBaseY + 15);
    }

    ctx.restore();

    // HUD Caption
    ctx.save();
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(
      'Click/Drag Chart to step Setpoint r(t) | Drag Drone (or Right-Click) to inject physical impulse',
      16,
      24
    );
    ctx.restore();
  },
};

export default PIDControlPlugin;
