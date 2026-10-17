export type AlgorithmCategory = 
  | 'perception-slam'
  | 'motion-planning'
  | 'kinematics-control'
  | 'reinforcement-ai';

export type ControlSchema =
  | {
      id: string;
      label: string;
      type: 'slider';
      min: number;
      max: number;
      step: number;
      defaultValue: number;
      unit?: string;
      description?: string;
    }
  | {
      id: string;
      label: string;
      type: 'toggle';
      defaultValue: boolean;
      description?: string;
    }
  | {
      id: string;
      label: string;
      type: 'select';
      options: { label: string; value: string }[];
      defaultValue: string;
    };

export interface MetricReadout {
  id: string;
  label: string;
  value: string | number;
  unit?: string;
  color?: 'default' | 'emerald' | 'amber' | 'rose' | 'sky';
}

export interface CanvasPointerEvent {
  worldX: number;
  worldY: number;
  isDragging: boolean;
  button: 'left' | 'right';
  shiftKey: boolean;
}

export interface AlgorithmPlugin<TState = unknown, TParams = Record<string, any>> {
  id: string;
  title: string;
  shortDescription: string;
  category: AlgorithmCategory;
  controls: ControlSchema[];
  documentation: {
    intuition: string;
    governingEquationLatex: string;
    pseudocode: string[];
  };
  createInitialState(params: TParams): TState;
  step(currentState: TState, params: TParams): {
    nextState: TState;
    isComplete: boolean;
    activePseudocodeLine?: number;
  };
  onPointerInteraction?(
    currentState: TState,
    event: CanvasPointerEvent,
    params: TParams
  ): TState;
  getMetrics(currentState: TState, params: TParams): MetricReadout[];
  render(
    ctx: CanvasRenderingContext2D,
    currentState: TState,
    params: TParams,
    viewport: { width: number; height: number; dpr: number }
  ): void;
}
