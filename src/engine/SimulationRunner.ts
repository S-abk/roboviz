// src/engine/SimulationRunner.ts
import { AlgorithmPlugin } from '../types/plugin';

/**
 * Safe deep clone that preserves Infinity, NaN, nested arrays, and plain objects
 * across both localhost and non-HTTPS LAN connections.
 */
export function safeClone<T>(val: T): T {
  if (val === null || typeof val !== 'object') {
    return val;
  }
  if (Array.isArray(val)) {
    return val.map(safeClone) as unknown as T;
  }
  const copy: Record<string, any> = {};
  for (const key of Object.keys(val)) {
    copy[key] = safeClone((val as Record<string, any>)[key]);
  }
  return copy as T;
}

export class SimulationRunner<TState, TParams extends Record<string, any>> {
  private plugin: AlgorithmPlugin<TState, TParams>;
  private params: TParams;
  private history: TState[] = [];
  private activeLineHistory: number[] = [];
  private cursor: number = 0;
  private isPlaying: boolean = false;
  private stepsPerSecond: number = 20;
  private speedMultiplier: number = 1;
  private stepAccumulator: number = 0;
  private lastStepTime: number = 0;
  private rafId: number | null = null;
  private onUpdateCallback: () => void;

  constructor(
    plugin: AlgorithmPlugin<TState, TParams>,
    initialParams: TParams,
    onUpdate: () => void
  ) {
    this.plugin = plugin;
    this.params = safeClone(initialParams);
    this.onUpdateCallback = onUpdate;

    // Initialize state WITHOUT firing onUpdateCallback prematurely
    const initialState = this.plugin.createInitialState(this.params);
    this.history = [safeClone(initialState)];
    this.activeLineHistory = [0];
    this.cursor = 0;
  }

  public setSpeed(multiplier: number): void {
    this.speedMultiplier = multiplier;
    this.stepAccumulator = 0;
  }

  public getSpeed(): number {
    return this.speedMultiplier;
  }

  public reset(): void {
    this.pause();
    const initialState = this.plugin.createInitialState(this.params);
    this.history = [safeClone(initialState)];
    this.activeLineHistory = [0];
    this.cursor = 0;
    this.stepAccumulator = 0;
    this.onUpdateCallback();
  }

  public getCurrentState(): TState {
    return this.history[this.cursor];
  }

  public replaceCurrentState(newState: TState): void {
    this.history[this.cursor] = safeClone(newState);
    this.onUpdateCallback();
  }

  public getStepInfo() {
    return {
      currentStep: this.cursor,
      totalCachedSteps: this.history.length - 1,
      isPlaying: this.isPlaying,
      activePseudocodeLine: this.activeLineHistory[this.cursor] ?? 0
    };
  }

  public stepForward(): { activeLine?: number; canContinue: boolean } {
    if (this.cursor < this.history.length - 1) {
      this.cursor++;
      this.onUpdateCallback();
      return {
        activeLine: this.activeLineHistory[this.cursor],
        canContinue: true
      };
    }

    const current = this.getCurrentState();
    const { nextState, isComplete, activePseudocodeLine } = this.plugin.step(
      safeClone(current),
      this.params
    );

    const lineIdx = activePseudocodeLine ?? 0;

    if (this.history.length < 2000) {
      this.history.push(safeClone(nextState));
      this.activeLineHistory.push(lineIdx);
      this.cursor++;
    } else {
      this.history[this.cursor] = safeClone(nextState);
      this.activeLineHistory[this.cursor] = lineIdx;
    }

    if (isComplete) this.pause();
    this.onUpdateCallback();
    return { activeLine: lineIdx, canContinue: !isComplete };
  }

  public stepBackward(): void {
    this.pause();
    if (this.cursor > 0) {
      this.cursor--;
      this.onUpdateCallback();
    }
  }

  public scrubTo(stepIndex: number): void {
    this.pause();
    this.cursor = Math.max(0, Math.min(stepIndex, this.history.length - 1));
    this.onUpdateCallback();
  }

  public updateParams(newParams: Partial<TParams>, resetSimulation = false): void {
    this.params = { ...this.params, ...newParams };
    if (resetSimulation) {
      this.reset();
    } else {
      this.history = this.history.slice(0, this.cursor + 1);
      this.activeLineHistory = this.activeLineHistory.slice(0, this.cursor + 1);
      this.onUpdateCallback();
    }
  }

  public play(): void {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.lastStepTime = performance.now();
    this.stepAccumulator = 0;

    const tick = (now: number) => {
      if (!this.isPlaying) return;
      const interval = 1000 / this.stepsPerSecond;
      if (now - this.lastStepTime >= interval) {
        this.lastStepTime = now;
        this.stepAccumulator += this.speedMultiplier;
        const stepsToRun = Math.floor(this.stepAccumulator);
        if (stepsToRun > 0) {
          this.stepAccumulator -= stepsToRun;
          for (let i = 0; i < stepsToRun; i++) {
            const { canContinue } = this.stepForward();
            if (!canContinue) return;
          }
        }
      }
      this.rafId = requestAnimationFrame(tick);
    };

    this.rafId = requestAnimationFrame(tick);
    this.onUpdateCallback();
  }

  public pause(): void {
    this.isPlaying = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.stepAccumulator = 0;
    this.onUpdateCallback();
  }

  public togglePlay(): void {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }
}
