// src/main.ts
import { PLUGIN_REGISTRY, CATEGORY_LABELS } from './registry';
import { AlgorithmPlugin, AlgorithmCategory } from './types/plugin';
import { SimulationRunner, safeClone } from './engine/SimulationRunner';

declare const katex: any;

// Parameters that structurally re-initialize the simulation when changed
const RESET_ON_CHANGE_CONTROLS: Record<string, string[]> = {
  'motion-planning-astar': ['heuristicWeight', 'allowDiagonals'],
  'motion-planning-prm': ['connectionRadius', 'maxNeighbors'],
  'motion-planning-hybrid-astar': ['minTurnRadius', 'reversePenalty', 'heuristicWeight', 'allowReverse'],
  'perception-particle-filter': ['particleCount'],
  'perception-icp': ['shape', 'sensorNoise'],
  'perception-pose-graph-slam': ['odomNoise', 'loopClosureRadius'],
  'motion-planning-potential-fields': ['scenario'],
  'control-pure-pursuit': ['pathType'],
  'ai-self-attention': ['activeTokenIdx', 'temperature', 'projectionDimension', 'causalMask'],
  'ai-mlp-backprop': ['dataset', 'hiddenUnits']
};

// Plugins where parameter tweaks should step immediately when paused for live kinematic feedback
const STEP_ON_CHANGE_PLUGINS = new Set([
  'kinematics-2d-ik',
  'kinematics-jacobian-ik'
]);

// Pedagogical quick-start presets showcasing signature behaviors and failure modes
const MODULE_PRESETS: Record<
  string,
  { label: string; params: Record<string, any>; reset?: boolean }[]
> = {
  'perception-particle-filter': [
    {
      label: 'High Noise Cloud',
      params: { particleCount: 500, motionNoise: 3.5, sensorNoise: 26, resampleRatio: 0.4, showSensorBeams: true }
    },
    {
      label: 'Sharp Convergence',
      params: { particleCount: 250, motionNoise: 0.8, sensorNoise: 6, resampleRatio: 0.75, showSensorBeams: true }
    }
  ],
  'perception-icp': [
    {
      label: 'Clean L-Wall Scan',
      params: { shape: 'corner', dampingFactor: 0.75, maxCorrespondenceDist: 160, sensorNoise: 0 }
    },
    {
      label: 'Noisy Stellar Scan',
      params: { shape: 'star', dampingFactor: 0.6, maxCorrespondenceDist: 150, sensorNoise: 4 }
    },
    {
      label: 'Tight Threshold Trap',
      params: { shape: 'robot', dampingFactor: 0.75, maxCorrespondenceDist: 50, sensorNoise: 1 }
    }
  ],
  'perception-occupancy-grid': [
    {
      label: 'Dense 360° LiDAR',
      params: { lidarBeams: 72, maxRange: 220, logOddsOcc: 1.1, logOddsFree: 0.45, showTrueMapOverlay: true },
      reset: false
    },
    {
      label: 'Sparse Short-Range',
      params: { lidarBeams: 18, maxRange: 100, logOddsOcc: 0.7, logOddsFree: 0.3, showTrueMapOverlay: true },
      reset: false
    }
  ],
  'perception-ekf-slam': [
    {
      label: 'High Odometry Drift + Loop Snap',
      params: { sensorRange: 145, motionNoise: 0.11, measurementNoise: 0.06, showCorrelations: true }
    },
    {
      label: 'Long-Range Multi-Landmark',
      params: { sensorRange: 220, motionNoise: 0.04, measurementNoise: 0.06, showCorrelations: true }
    },
    {
      label: 'Myopic Short-Range (R=90)',
      params: { sensorRange: 90, motionNoise: 0.06, measurementNoise: 0.10, showCorrelations: true }
    }
  ],
  'perception-pose-graph-slam': [
    {
      label: 'Severe Odometry Drift',
      params: { odomNoise: 0.10, loopClosureRadius: 75, relaxationRate: 0.25, informationWeight: 6.0 }
    },
    {
      label: 'Missed Loop Closure (r=25)',
      params: { odomNoise: 0.06, loopClosureRadius: 25, relaxationRate: 0.25, informationWeight: 5.0 }
    },
    {
      label: 'Stiff Loop Snap (Ω=10)',
      params: { odomNoise: 0.05, loopClosureRadius: 70, relaxationRate: 0.35, informationWeight: 10.0 }
    }
  ],
  'motion-planning-astar': [
    {
      label: 'Dijkstra (w=0)',
      params: { heuristicWeight: 0.0, allowDiagonals: false, showCosts: true }
    },
    {
      label: 'Optimal A* (w=1)',
      params: { heuristicWeight: 1.0, allowDiagonals: true, showCosts: true }
    },
    {
      label: 'Greedy Best-First (w=3)',
      params: { heuristicWeight: 3.0, allowDiagonals: true, showCosts: true }
    }
  ],
  'motion-planning-rrt-star': [
    {
      label: 'Asymptotic RRT* Rewiring',
      params: { stepSize: 22, goalBias: 0.12, rewireRadius: 55, enableRewiring: true }
    },
    {
      label: 'Unoptimized Standard RRT',
      params: { stepSize: 22, goalBias: 0.08, rewireRadius: 45, enableRewiring: false }
    }
  ],
  'motion-planning-prm': [
    {
      label: 'Dense Multi-Query Graph',
      params: { targetNodes: 150, connectionRadius: 115, maxNeighbors: 8, obstacleInflation: 6 }
    },
    {
      label: 'Sparse Roadmap (N=60)',
      params: { targetNodes: 60, connectionRadius: 85, maxNeighbors: 4, obstacleInflation: 6 }
    },
    {
      label: 'High Clearance Buffer',
      params: { targetNodes: 110, connectionRadius: 100, maxNeighbors: 6, obstacleInflation: 14 }
    }
  ],
  'motion-planning-potential-fields': [
    {
      label: 'U-Trap Local Minimum',
      params: { kAtt: 0.8, kRep: 1600, repDistance: 80, scenario: 'u-trap', showForceQuivers: true }
    },
    {
      label: 'Narrow Gate Pass',
      params: { kAtt: 1.2, kRep: 900, repDistance: 45, scenario: 'gate', showForceQuivers: true }
    },
    {
      label: 'Gate Repulsion Blockade',
      params: { kAtt: 0.6, kRep: 2600, repDistance: 115, scenario: 'gate', showForceQuivers: true }
    }
  ],
  'motion-planning-dwa': [
    {
      label: 'Cautious Obstacle Buffer',
      params: { predictTime: 2.2, maxSpeed: 60, maxYawRate: 1.8, headingWeight: 1.2, clearanceWeight: 2.6, velocityWeight: 0.5 },
      reset: false
    },
    {
      label: 'Aggressive High-Speed',
      params: { predictTime: 1.2, maxSpeed: 105, maxYawRate: 2.2, headingWeight: 2.4, clearanceWeight: 0.6, velocityWeight: 1.8 },
      reset: false
    }
  ],
  'motion-planning-hybrid-astar': [
    {
      label: 'Agile Car + Reverse',
      params: { minTurnRadius: 40, reversePenalty: 1.4, heuristicWeight: 1.3, allowReverse: true }
    },
    {
      label: 'Wide Turning Radius (R=80)',
      params: { minTurnRadius: 80, reversePenalty: 1.4, heuristicWeight: 1.3, allowReverse: true }
    },
    {
      label: 'Forward-Only Cruiser',
      params: { minTurnRadius: 48, reversePenalty: 2.0, heuristicWeight: 1.3, allowReverse: false }
    }
  ],
  'control-pid': [
    {
      label: 'Pure P (Gravity Droop)',
      params: { kp: 1.8, ki: 0.0, kd: 0.0, windDisturbance: 0, droneMass: 1.0 }
    },
    {
      label: 'Underdamped Ringing',
      params: { kp: 5.0, ki: 0.2, kd: 0.2, windDisturbance: 0, droneMass: 1.0 }
    },
    {
      label: 'Tuned PID + Wind',
      params: { kp: 3.2, ki: 2.0, kd: 2.4, windDisturbance: 25, droneMass: 1.0 }
    }
  ],
  'kinematics-jacobian-ik': [
    {
      label: '8-DOF Hyper-Tentacle',
      params: { numLinks: 8, linkLength: 26, dampingAlpha: 0.3, ccdSweepsPerStep: 3 },
      reset: false
    },
    {
      label: '3-DOF Rigid Manipulator',
      params: { numLinks: 3, linkLength: 46, dampingAlpha: 0.2, ccdSweepsPerStep: 1 },
      reset: false
    }
  ],
  'control-pure-pursuit': [
    {
      label: 'Fishtailing (Small Ld)',
      params: { lookaheadDistance: 18, vehicleSpeed: 85, wheelbase: 32, pathType: 'oval' }
    },
    {
      label: 'Corner Cutting (Large Ld)',
      params: { lookaheadDistance: 120, vehicleSpeed: 75, wheelbase: 32, pathType: 'scurve' }
    },
    {
      label: 'Tuned Chicane',
      params: { lookaheadDistance: 48, vehicleSpeed: 75, wheelbase: 32, pathType: 'scurve' }
    }
  ],
  'control-lqr': [
    {
      label: 'Aggressive Fast Docking (Low R)',
      params: { qPosition: 3.8, qVelocity: 1.2, rEffort: 0.3, crossCoupling: 0.2 },
      reset: false
    },
    {
      label: 'Fuel-Minimal Glide (High R)',
      params: { qPosition: 0.8, qVelocity: 0.6, rEffort: 4.2, crossCoupling: 0.2 },
      reset: false
    },
    {
      label: 'Strong Coriolis Swirl (ω=0.7)',
      params: { qPosition: 1.8, qVelocity: 0.9, rEffort: 1.2, crossCoupling: 0.7 },
      reset: false
    }
  ],
  'control-mpc': [
    {
      label: 'Long Horizon Swerve (N=22)',
      params: { horizonSteps: 22, trackingWeight: 2.0, controlWeight: 0.7, obstacleWeight: 3.5 },
      reset: false
    },
    {
      label: 'Myopic Short Horizon (N=6)',
      params: { horizonSteps: 6, trackingWeight: 2.0, controlWeight: 0.5, obstacleWeight: 2.0 },
      reset: false
    },
    {
      label: 'High Comfort Smooth (R=2.8)',
      params: { horizonSteps: 16, trackingWeight: 1.5, controlWeight: 2.8, obstacleWeight: 3.0 },
      reset: false
    }
  ],
  'rl-q-learning': [
    {
      label: 'Fast Training (20x/tick)',
      params: { alpha: 0.6, gamma: 0.95, epsilon: 0.15, stepsPerTick: 20 },
      reset: false
    },
    {
      label: 'Greedy Policy (ε=0)',
      params: { alpha: 0.5, gamma: 0.95, epsilon: 0.0, stepsPerTick: 2 },
      reset: false
    },
    {
      label: 'High Exploration (ε=0.4)',
      params: { alpha: 0.5, gamma: 0.95, epsilon: 0.4, stepsPerTick: 4 },
      reset: false
    }
  ],
  'rl-cartpole-policy': [
    {
      label: 'Stable Policy (Low Noise)',
      params: { learningRate: 0.12, explorationNoise: 0.05, poleLength: 85, gravity: 12.5 },
      reset: false
    },
    {
      label: 'High Gravity Perturbation',
      params: { learningRate: 0.22, explorationNoise: 0.15, poleLength: 70, gravity: 22.5 },
      reset: false
    },
    {
      label: 'Stochastic Exploration',
      params: { learningRate: 0.25, explorationNoise: 0.45, poleLength: 90, gravity: 12.5 },
      reset: false
    }
  ],
  'ai-kmeans-gmm': [
    {
      label: 'Soft EM Anisotropic Ellipses',
      params: { numClusters: 4, pointSpread: 32, softnessTemp: 1.0, stepRate: 0.45 },
      reset: false
    },
    {
      label: 'Hard K-Means Partition (T=0.2)',
      params: { numClusters: 4, pointSpread: 32, softnessTemp: 0.2, stepRate: 0.5 },
      reset: false
    },
    {
      label: 'Over-Clustered 6 Components',
      params: { numClusters: 6, pointSpread: 35, softnessTemp: 0.9, stepRate: 0.45 },
      reset: false
    }
  ],
  'ai-self-attention': [
    {
      label: 'Sharp Focus (τ=0.2)',
      params: { temperature: 0.2, activeTokenIdx: 3, causalMask: false }
    },
    {
      label: 'Causal Masked',
      params: { temperature: 0.45, activeTokenIdx: 4, causalMask: true }
    }
  ],
  'ai-mlp-backprop': [
    {
      label: 'Linear Underfitting (H=2)',
      params: { learningRate: 0.18, hiddenUnits: 2, l2Reg: 0.002, stepsPerFrame: 9, dataset: 'xor' }
    },
    {
      label: 'Deep Non-Linear (H=8)',
      params: { learningRate: 0.24, hiddenUnits: 8, l2Reg: 0.002, stepsPerFrame: 15, dataset: 'xor' }
    },
    {
      label: 'Concentric Ring Pocket',
      params: { learningRate: 0.22, hiddenUnits: 8, l2Reg: 0.002, stepsPerFrame: 15, dataset: 'circle' }
    }
  ]
};

class RoboVizApp {
  private activePlugin!: AlgorithmPlugin<any, any>;
  private activeParams: Record<string, any> = {};
  private runner!: SimulationRunner<any, any>;
  private currentSpeed: number = 1;
  private activePresetLabel: string | null = null;
  private searchQuery: string = '';

  private canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
  private ctx = this.canvas.getContext('2d')!;
  private isDragging = false;

  constructor() {
    this.setupCanvasInteractions();
    this.setupPlaybackControls();
    this.setupKeyboardShortcuts();
    this.setupHashRouting();
    this.setupCurriculumNavigation();

    const initialHash = window.location.hash.slice(1);
    const initialPlugin =
      PLUGIN_REGISTRY.find((p) => p.id === initialHash) || PLUGIN_REGISTRY[0];
    this.selectPlugin(initialPlugin);

    window.addEventListener('resize', () => this.renderFrame());
  }

  private setupCurriculumNavigation() {
    const searchInput = document.getElementById('catalog-search') as HTMLInputElement | null;
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        this.searchQuery = searchInput.value.trim().toLowerCase();
        this.renderSidebarCatalog();
      });
    }

    const prevBtn = document.getElementById('btn-prev-module');
    const nextBtn = document.getElementById('btn-next-module');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => this.navigateModule(-1));
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => this.navigateModule(1));
    }

    const resetParamsBtn = document.getElementById('btn-reset-params');
    if (resetParamsBtn) {
      resetParamsBtn.addEventListener('click', () => this.resetParamsToDefaults());
    }
  }

  private navigateModule(delta: number) {
    const idx = PLUGIN_REGISTRY.findIndex((p) => p.id === this.activePlugin?.id);
    if (idx === -1) return;
    const nextIdx = (idx + delta + PLUGIN_REGISTRY.length) % PLUGIN_REGISTRY.length;
    this.selectPlugin(PLUGIN_REGISTRY[nextIdx]);
  }

  private resetParamsToDefaults() {
    if (!this.activePlugin) return;
    this.activePresetLabel = null;
    this.updatePresetButtons();
    const defaults: Record<string, any> = {};
    for (const ctrl of this.activePlugin.controls) {
      defaults[ctrl.id] = ctrl.defaultValue;
    }
    this.applyPreset(defaults, true);
  }

  private setupHashRouting() {
    window.addEventListener('hashchange', () => {
      const hashId = window.location.hash.slice(1);
      if (hashId && hashId !== this.activePlugin?.id) {
        const targetPlugin = PLUGIN_REGISTRY.find((p) => p.id === hashId);
        if (targetPlugin) {
          this.selectPlugin(targetPlugin);
        }
      }
    });
  }

  private selectPlugin(plugin: AlgorithmPlugin<any, any>) {
    if (this.runner) {
      this.runner.pause();
    }

    this.activePlugin = plugin;
    this.activePresetLabel = null;
    this.activeParams = {};

    if (window.location.hash.slice(1) !== plugin.id) {
      window.history.replaceState(null, '', '#' + plugin.id);
    }

    for (const ctrl of plugin.controls) {
      this.activeParams[ctrl.id] = ctrl.defaultValue;
    }

    // 1. Instantiate runner first so `this.runner` is defined before any UI sync
    this.runner = new SimulationRunner(plugin, this.activeParams, () => {
      this.syncUI();
      this.renderFrame();
    });
    this.runner.setSpeed(this.currentSpeed);

    const speedSelect = document.getElementById('speed-select') as HTMLSelectElement | null;
    if (speedSelect) {
      speedSelect.value = String(this.currentSpeed);
    }

    // 2. Populate header & documentation
    document.getElementById('module-title')!.textContent = plugin.title;
    document.getElementById('module-subtitle')!.textContent = plugin.shortDescription;
    document.getElementById('intuition-text')!.textContent = plugin.documentation.intuition;

    this.renderEquation(plugin.documentation.governingEquationLatex);
    this.renderControlsSchema(plugin);
    this.renderSidebarCatalog();

    // 3. Initial UI & Canvas draw
    this.syncUI();
    this.renderFrame();
  }

  private renderSidebarCatalog() {
    const nav = document.getElementById('catalog-nav')!;
    nav.innerHTML = '';

    const q = this.searchQuery;
    let totalMatched = 0;

    const categories = Object.keys(CATEGORY_LABELS) as AlgorithmCategory[];
    for (const cat of categories) {
      const catLabel = CATEGORY_LABELS[cat].toLowerCase();
      const pluginsInCat = PLUGIN_REGISTRY.filter((p) => {
        if (p.category !== cat) return false;
        if (!q) return true;
        return (
          p.title.toLowerCase().includes(q) ||
          p.shortDescription.toLowerCase().includes(q) ||
          p.id.toLowerCase().includes(q) ||
          catLabel.includes(q)
        );
      });
      if (pluginsInCat.length === 0) continue;
      totalMatched += pluginsInCat.length;

      const groupDiv = document.createElement('div');
      groupDiv.className = 'space-y-1';
      groupDiv.innerHTML = `<div class="px-2 py-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">${CATEGORY_LABELS[cat]}</div>`;

      for (const plugin of pluginsInCat) {
        const isActive = this.activePlugin?.id === plugin.id;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `w-full text-left px-3 py-2 rounded-md text-xs font-medium transition flex items-center justify-between ${
          isActive
            ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
            : 'text-slate-300 hover:bg-slate-800/60'
        }`;
        btn.textContent = plugin.title;
        btn.addEventListener('click', () => this.selectPlugin(plugin));
        groupDiv.appendChild(btn);
      }
      nav.appendChild(groupDiv);
    }

    if (totalMatched === 0) {
      nav.innerHTML = `<div class="px-3 py-6 text-center text-xs text-slate-500">No algorithms match "${this.searchQuery}"</div>`;
    }

    const countEl = document.getElementById('catalog-count');
    if (countEl) {
      countEl.textContent = q
        ? `${totalMatched} / ${PLUGIN_REGISTRY.length} Modules`
        : `${PLUGIN_REGISTRY.length} Modules`;
    }
  }

  private handleParamUpdate(ctrlId: string, val: any) {
    this.activePresetLabel = null;
    this.updatePresetButtons();
    this.activeParams[ctrlId] = val;
    const resetList = RESET_ON_CHANGE_CONTROLS[this.activePlugin.id] || [];
    const shouldReset = resetList.includes(ctrlId);

    this.runner.updateParams({ [ctrlId]: val }, shouldReset);

    if (!shouldReset && STEP_ON_CHANGE_PLUGINS.has(this.activePlugin.id)) {
      this.runner.stepForward();
    }
  }

  private getPresetButtonClass(isActive: boolean): string {
    return `px-2.5 py-1 text-xs font-medium rounded-full transition cursor-pointer border ${
      isActive
        ? 'bg-sky-500/20 text-sky-300 border-sky-400 font-semibold'
        : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border-slate-700 hover:border-slate-600'
    }`;
  }

  private updatePresetButtons() {
    const buttons = document.querySelectorAll<HTMLButtonElement>('[data-preset-label]');
    buttons.forEach((btn) => {
      const isActive = btn.dataset.presetLabel === this.activePresetLabel;
      btn.className = this.getPresetButtonClass(isActive);
    });
  }

  private applyPreset(presetParams: Record<string, any>, shouldReset = true) {
    this.activeParams = { ...this.activeParams, ...presetParams };
    this.syncControlInputs();
    this.runner.updateParams(presetParams, shouldReset);
    if (STEP_ON_CHANGE_PLUGINS.has(this.activePlugin.id)) {
      this.runner.stepForward();
    }
  }

  private syncControlInputs() {
    for (const ctrl of this.activePlugin.controls) {
      const val = this.activeParams[ctrl.id];
      if (val === undefined) continue;

      if (ctrl.type === 'slider') {
        const input = document.querySelector<HTMLInputElement>(`input[data-ctrl-id="${ctrl.id}"]`);
        const valSpan = document.getElementById(`val-${ctrl.id}`);
        if (input) input.value = String(val);
        if (valSpan) valSpan.textContent = `${val}${ctrl.unit || ''}`;
      } else if (ctrl.type === 'toggle') {
        const input = document.querySelector<HTMLInputElement>(`input[data-ctrl-id="${ctrl.id}"]`);
        if (input) input.checked = Boolean(val);
      } else if (ctrl.type === 'select') {
        const select = document.querySelector<HTMLSelectElement>(`select[data-ctrl-id="${ctrl.id}"]`);
        if (select) select.value = String(val);
      }
    }
  }

  private renderControlsSchema(plugin: AlgorithmPlugin<any, any>) {
    const container = document.getElementById('controls-container')!;
    container.innerHTML = '';

    const presets = MODULE_PRESETS[plugin.id];
    if (presets && presets.length > 0) {
      const presetSection = document.createElement('div');
      presetSection.className = 'pb-3 border-b border-slate-800 space-y-2';
      presetSection.innerHTML = `
        <div class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-sky-400"></span>
          Try Scenario Presets
        </div>
        <div class="flex flex-wrap gap-1.5" id="presets-button-group"></div>
      `;
      const btnGroup = presetSection.querySelector('#presets-button-group')!;
      for (const preset of presets) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.presetLabel = preset.label;
        btn.className = this.getPresetButtonClass(this.activePresetLabel === preset.label);
        btn.textContent = preset.label;
        btn.addEventListener('click', () => {
          this.activePresetLabel = preset.label;
          this.updatePresetButtons();
          this.applyPreset(preset.params, preset.reset ?? true);
        });
        btnGroup.appendChild(btn);
      }
      container.appendChild(presetSection);
    }

    for (const ctrl of plugin.controls) {
      const wrapper = document.createElement('div');
      wrapper.className = 'space-y-1';

      if (ctrl.type === 'slider') {
        wrapper.innerHTML = `
          <div class="flex justify-between text-xs">
            <label class="text-slate-300 font-medium">${ctrl.label}</label>
            <span id="val-${ctrl.id}" class="font-mono text-sky-400">${this.activeParams[ctrl.id]}${ctrl.unit || ''}</span>
          </div>
          <input type="range" data-ctrl-id="${ctrl.id}" min="${ctrl.min}" max="${ctrl.max}" step="${ctrl.step}" value="${this.activeParams[ctrl.id]}"
            class="w-full accent-sky-400 cursor-pointer" />
          ${ctrl.description ? `<p class="text-[11px] text-slate-500 leading-tight">${ctrl.description}</p>` : ''}
        `;
        const input = wrapper.querySelector('input')!;
        const valSpan = wrapper.querySelector(`#val-${ctrl.id}`)!;
        input.addEventListener('input', () => {
          const val = parseFloat(input.value);
          valSpan.textContent = `${val}${ctrl.unit || ''}`;
          this.handleParamUpdate(ctrl.id, val);
        });
      } else if (ctrl.type === 'toggle') {
        wrapper.innerHTML = `
          <label class="flex items-center justify-between cursor-pointer py-1">
            <span class="text-xs text-slate-300 font-medium">${ctrl.label}</span>
            <input type="checkbox" data-ctrl-id="${ctrl.id}" ${this.activeParams[ctrl.id] ? 'checked' : ''} class="accent-sky-400 w-4 h-4 rounded" />
          </label>
          ${ctrl.description ? `<p class="text-[11px] text-slate-500 leading-tight">${ctrl.description}</p>` : ''}
        `;
        const checkbox = wrapper.querySelector('input')!;
        checkbox.addEventListener('change', () => {
          this.handleParamUpdate(ctrl.id, checkbox.checked);
        });
      } else if (ctrl.type === 'select') {
        const optionsHtml = ctrl.options
          .map(
            (opt) =>
              `<option value="${opt.value}" ${opt.value === this.activeParams[ctrl.id] ? 'selected' : ''}>${opt.label}</option>`
          )
          .join('');

        wrapper.innerHTML = `
          <label class="block text-xs text-slate-300 font-medium mb-1">${ctrl.label}</label>
          <select data-ctrl-id="${ctrl.id}" class="w-full bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-md px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500 cursor-pointer">
            ${optionsHtml}
          </select>
        `;
        const selectEl = wrapper.querySelector('select')!;
        selectEl.addEventListener('change', () => {
          this.handleParamUpdate(ctrl.id, selectEl.value);
        });
      }

      container.appendChild(wrapper);
    }
  }

  private renderEquation(latex: string) {
    const mathEl = document.getElementById('math-container')!;
    const tryRender = () => {
      if (typeof katex !== 'undefined') {
        katex.render(latex, mathEl, { throwOnError: false, displayMode: true });
      } else {
        mathEl.textContent = latex;
        setTimeout(tryRender, 200);
      }
    };
    tryRender();
  }

  private syncUI() {
    if (!this.runner) return;
    const state = this.runner.getCurrentState();
    const stepInfo = this.runner.getStepInfo();

    const metrics = this.activePlugin.getMetrics(state, this.activeParams);
    const strip = document.getElementById('telemetry-strip')!;
    strip.innerHTML = metrics
      .map((m) => {
        const colorMap: Record<string, string> = {
          sky: 'text-sky-400 border-sky-500/30 bg-sky-500/10',
          emerald: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
          amber: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
          rose: 'text-rose-400 border-rose-500/30 bg-rose-500/10',
          default: 'text-slate-200 border-slate-700 bg-slate-800'
        };
        const badgeStyle = colorMap[m.color || 'default'];
        return `
          <div class="px-2.5 py-1 rounded border text-xs flex items-center gap-2 ${badgeStyle}">
            <span class="text-slate-400">${m.label}:</span>
            <strong class="font-mono">${m.value}${m.unit || ''}</strong>
          </div>
        `;
      })
      .join('');

    document.getElementById('step-counter')!.textContent = String(stepInfo.currentStep);
    const scrubber = document.getElementById('timeline-scrubber') as HTMLInputElement;
    scrubber.max = String(stepInfo.totalCachedSteps);
    scrubber.value = String(stepInfo.currentStep);

    const playBtn = document.getElementById('btn-play-pause')!;
    playBtn.textContent = stepInfo.isPlaying ? 'Pause' : 'Play';

    const activeLine = stepInfo.activePseudocodeLine;
    const codeContainer = document.getElementById('pseudocode-container')!;
    codeContainer.innerHTML = this.activePlugin.documentation.pseudocode
      .map((line, idx) => {
        const isCurrent = idx === activeLine;
        return `<div class="px-2 py-0.5 rounded transition-colors ${
          isCurrent
            ? 'bg-sky-500/20 text-sky-300 border-l-2 border-sky-400 font-semibold'
            : 'text-slate-400'
        }">${line}</div>`;
      })
      .join('');
  }

  private renderFrame() {
    if (!this.runner) return;
    const container = document.getElementById('canvas-container')!;
    const rect = container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;

    this.ctx.save();
    this.ctx.scale(dpr, dpr);

    this.ctx.fillStyle = '#020617';
    this.ctx.fillRect(0, 0, rect.width, rect.height);
    this.drawBackgroundGrid(rect.width, rect.height);

    this.activePlugin.render(this.ctx, this.runner.getCurrentState(), this.activeParams, {
      width: rect.width,
      height: rect.height,
      dpr
    });

    this.ctx.restore();
  }

  private drawBackgroundGrid(width: number, height: number) {
    const step = 32;
    this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    for (let x = (width / 2) % step; x < width; x += step) {
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
    }
    for (let y = (height / 2) % step; y < height; y += step) {
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
    }
    this.ctx.stroke();
  }

  private setupCanvasInteractions() {
    const dispatchPointer = (e: PointerEvent, isDragging: boolean) => {
      if (!this.activePlugin?.onPointerInteraction || !this.runner) return;
      const rect = this.canvas.getBoundingClientRect();
      const worldX = e.clientX - rect.left - rect.width / 2;
      const worldY = e.clientY - rect.top - rect.height / 2;

      const updatedState = this.activePlugin.onPointerInteraction(
        safeClone(this.runner.getCurrentState()),
        {
          worldX,
          worldY,
          isDragging,
          button: e.button === 2 || e.buttons === 2 ? 'right' : 'left',
          shiftKey: e.shiftKey
        },
        this.activeParams
      );

      this.runner.replaceCurrentState(updatedState);
    };

    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.canvas.addEventListener('pointerdown', (e) => {
      this.isDragging = true;
      dispatchPointer(e, false);
    });

    window.addEventListener('pointermove', (e) => {
      if (!this.isDragging) return;
      dispatchPointer(e, true);
    });

    window.addEventListener('pointerup', () => {
      this.isDragging = false;
    });
  }

  private setupPlaybackControls() {
    document.getElementById('btn-reset')!.addEventListener('click', () => this.runner.reset());
    document.getElementById('btn-step-back')!.addEventListener('click', () => this.runner.stepBackward());
    document.getElementById('btn-step-fwd')!.addEventListener('click', () => {
      this.runner.stepForward();
    });
    document.getElementById('btn-play-pause')!.addEventListener('click', () => {
      this.runner.togglePlay();
    });

    const speedSelect = document.getElementById('speed-select') as HTMLSelectElement | null;
    if (speedSelect) {
      speedSelect.addEventListener('change', () => {
        this.currentSpeed = parseFloat(speedSelect.value);
        this.runner.setSpeed(this.currentSpeed);
      });
    }

    const scrubber = document.getElementById('timeline-scrubber') as HTMLInputElement;
    scrubber.addEventListener('input', () => {
      this.runner.scrubTo(parseInt(scrubber.value, 10));
    });
  }

  private setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLSelectElement ||
        target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        this.runner.togglePlay();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        this.runner.pause();
        this.runner.stepBackward();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        this.runner.pause();
        this.runner.stepForward();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        this.runner.reset();
      } else if (e.key === '[') {
        e.preventDefault();
        this.navigateModule(-1);
      } else if (e.key === ']') {
        e.preventDefault();
        this.navigateModule(1);
      }
    });
  }
}

// Safe bootstrap whether DOMContentLoaded has already fired or not
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new RoboVizApp());
} else {
  new RoboVizApp();
}
