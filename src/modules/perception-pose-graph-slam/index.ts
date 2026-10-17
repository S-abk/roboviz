// FILE: src/modules/perception-pose-graph-slam/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface Pose2D {
  x: number;
  y: number;
  theta: number;
}

export interface GraphEdge {
  from: number;
  to: number;
  dx: number;
  dy: number;
  dtheta: number;
  weight: number;
  isLoopClosure: boolean;
}

export type PoseGraphStatus = 'accumulating' | 'relaxing' | 'converged';

export interface PoseGraphState {
  truePoses: Pose2D[];
  estimatedPoses: Pose2D[];
  odometryEdges: GraphEdge[];
  loopEdges: GraphEdge[];
  numActivePoses: number;
  status: PoseGraphStatus;
  iteration: number;
  relaxationIterations: number;
  ateRms: number;
  draggedNodeIndex: number | null;
  activeLine: number;
}

export interface PoseGraphParams {
  odomNoise: number;
  loopClosureRadius: number;
  relaxationRate: number;
  informationWeight: number;
}

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

/**
 * Builds a 24-node closed stadium loop of ground-truth robot poses.
 * Composed of top/bottom straight lines and left/right semicircles.
 */
function generateStadiumPoses(): Pose2D[] {
  const straightLen = 170;
  const radius = 75;
  const halfStraight = straightLen / 2; // 85
  const halfPi = Math.PI / 2;
  const curveArcLen = Math.PI * radius; // ~235.6
  const totalPerimeter = 2 * straightLen + 2 * curveArcLen; // ~811.2
  const numPoses = 24;
  const ds = totalPerimeter / numPoses;

  const poses: Pose2D[] = [];

  for (let k = 0; k < numPoses; k++) {
    const s = k * ds;

    if (s < straightLen) {
      // 1. Top straight: heading East (0)
      const x = -halfStraight + s;
      const y = -radius;
      poses.push({ x, y, theta: 0 });
    } else if (s < straightLen + curveArcLen) {
      // 2. Right semicircle: centered at (halfStraight, 0)
      const sCurve = s - straightLen;
      const phi = -halfPi + sCurve / radius;
      const x = halfStraight + radius * Math.cos(phi);
      const y = radius * Math.sin(phi);
      poses.push({ x, y, theta: normalizeAngle(phi + halfPi) });
    } else if (s < 2 * straightLen + curveArcLen) {
      // 3. Bottom straight: heading West (PI)
      const sStraight = s - (straightLen + curveArcLen);
      const x = halfStraight - sStraight;
      const y = radius;
      poses.push({ x, y, theta: Math.PI });
    } else {
      // 4. Left semicircle: centered at (-halfStraight, 0)
      const sCurve = s - (2 * straightLen + curveArcLen);
      const phi = halfPi + sCurve / radius;
      const x = -halfStraight + radius * Math.cos(phi);
      const y = radius * Math.sin(phi);
      poses.push({ x, y, theta: normalizeAngle(phi + halfPi) });
    }
  }

  return poses;
}

function computeAteRms(truePoses: Pose2D[], estPoses: Pose2D[], count: number): number {
  if (count <= 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < count; i++) {
    const dx = estPoses[i].x - truePoses[i].x;
    const dy = estPoses[i].y - truePoses[i].y;
    sumSq += dx * dx + dy * dy;
  }
  return Math.sqrt(sumSq / count);
}

export const PoseGraphSLAMPlugin: AlgorithmPlugin<PoseGraphState, PoseGraphParams> = {
  id: 'perception-pose-graph-slam',
  title: 'Pose-Graph SLAM & Loop Closure',
  shortDescription:
    'Odometry drift accumulation along a loop trajectory corrected globally via loop-closure constraints and Gauss-Newton graph relaxation.',
  category: 'perception-slam',

  controls: [
    {
      id: 'odomNoise',
      label: 'Odometry Noise Drift',
      type: 'slider',
      min: 0.01,
      max: 0.12,
      step: 0.01,
      defaultValue: 0.05,
      description: 'Rotational and translational drift per odometry step.'
    },
    {
      id: 'loopClosureRadius',
      label: 'Loop Closure Search Radius',
      type: 'slider',
      min: 25,
      max: 110,
      step: 5,
      defaultValue: 65,
      unit: 'px',
      description: 'Proximity threshold for detecting a loop closure between non-adjacent poses.'
    },
    {
      id: 'relaxationRate',
      label: 'Relaxation Step Rate (alpha)',
      type: 'slider',
      min: 0.05,
      max: 0.5,
      step: 0.05,
      defaultValue: 0.25,
      description: 'Gauss-Newton / spring relaxation step size per iteration.'
    },
    {
      id: 'informationWeight',
      label: 'Loop Information Weight (Omega)',
      type: 'slider',
      min: 1.0,
      max: 10.0,
      step: 1.0,
      defaultValue: 5.0,
      description: 'Relative confidence weight Ω_loop of loop closure vs odometry edges.'
    }
  ],

  documentation: {
    intuition:
      'Dead-reckoning odometry inexorably accumulates incremental translation and orientation noise, causing open-loop robot trajectories to bend, drift, and fail to close upon returning to previously mapped terrain. Loop closure detection identifies when the current sensor observation matches an earlier visited pose (separated by topological distance |i - j| > 10). By adding high-confidence relative transform constraints into a pose graph, Gauss-Newton / spring relaxation distributes residual errors across all intermediate odometry edges, straightening corrupted paths and snapping drifted poses back into global metric consistency.',
    governingEquationLatex:
      '\\textcolor{#38bdf8}{\\mathbf{x}^*} = \\arg\\min_{\\mathbf{x}} \\sum_{i} \\|\\textcolor{#f43f5e}{\\mathbf{e}_{i,i+1}^{\\text{odom}}}\\|_{\\Omega}^2 + \\sum_{(i,j) \\in \\mathcal{L}} \\|\\textcolor{#f59e0b}{\\mathbf{e}_{ij}^{\\text{loop}}}(\\textcolor{#38bdf8}{\\mathbf{x}_i}, \\textcolor{#10b981}{\\mathbf{x}_j})\\|_{\\Omega_{ij}}^2',
    pseudocode: [
      '1. Accumulate Odometry: x_{k+1} = x_k ⊕ (u_k + w_k) with drifting sensor noise',
      '2. Query Candidates: Find pose pairs (i, j) with |i - j| > 10 within loopClosureRadius',
      '3. Loop Closure: Register constraint e_{ij}^{loop} = z_{ij} ⊖ (x_j ⊖ x_i) with weight Ω',
      '4. Linearize Graph: Compute Jacobian matrices J_i, J_j and Information matrix H = J^T Ω J',
      '5. Solve Normal Equations: H Δx = -b where b = J^T Ω e (Gauss-Newton relaxation)',
      '6. Update Node Poses: x_k <- x_k + α · Δx_k for all non-anchored graph nodes',
      '7. Convergence Check: Terminate when RMS trajectory error ||Δx|| < ε or iterations complete'
    ]
  },

  createInitialState(params: PoseGraphParams): PoseGraphState {
    const truePoses = generateStadiumPoses();
    const estPoses: Pose2D[] = [{ ...truePoses[0] }];

    return {
      truePoses,
      estimatedPoses: estPoses,
      odometryEdges: [],
      loopEdges: [],
      numActivePoses: 1,
      status: 'accumulating',
      iteration: 0,
      relaxationIterations: 0,
      ateRms: 0,
      draggedNodeIndex: null,
      activeLine: 0
    };
  },

  step(currentState: PoseGraphState, params: PoseGraphParams) {
    const truePoses = currentState.truePoses;
    const estPoses = currentState.estimatedPoses.map((p) => ({ ...p }));
    const odomEdges = currentState.odometryEdges.map((e) => ({ ...e }));
    const loopEdges = currentState.loopEdges.map((e) => ({ ...e }));
    let numActive = currentState.numActivePoses;
    let status = currentState.status;
    let relaxCount = currentState.relaxationIterations;
    let activeLine = 0;
    let isComplete = false;

    // Phase 1: Sequential odometry accumulation
    if (numActive < 24) {
      const k = numActive;
      const prevTrue = truePoses[k - 1];
      const currTrue = truePoses[k];
      const prevEst = estPoses[k - 1];

      const dxTrue = currTrue.x - prevTrue.x;
      const dyTrue = currTrue.y - prevTrue.y;
      const dthTrue = normalizeAngle(currTrue.theta - prevTrue.theta);

      // Deterministic drifting odometry noise
      const noiseScale = params.odomNoise / 0.05;
      const noiseX = (Math.sin(k * 1.55) * 3.8 + 1.2) * noiseScale;
      const noiseY = (Math.cos(k * 1.35) * 3.4 - 1.8) * noiseScale;
      const noiseTh = (Math.sin(k * 0.95) * 0.035 + 0.028) * noiseScale;

      const noisyDx = dxTrue + noiseX;
      const noisyDy = dyTrue + noiseY;
      const noisyDth = normalizeAngle(dthTrue + noiseTh);

      const nextEst: Pose2D = {
        x: prevEst.x + noisyDx,
        y: prevEst.y + noisyDy,
        theta: normalizeAngle(prevEst.theta + noisyDth)
      };
      estPoses.push(nextEst);

      odomEdges.push({
        from: k - 1,
        to: k,
        dx: noisyDx,
        dy: noisyDy,
        dtheta: noisyDth,
        weight: 1.0,
        isLoopClosure: false
      });

      numActive++;
      activeLine = 0;

      // When numActivePoses >= 18, check for loop closures with early poses i < j - 10
      if (numActive >= 18) {
        const j = numActive - 1;
        for (let i = 0; i <= j - 11; i++) {
          const dGroundTruth = Math.hypot(
            truePoses[j].x - truePoses[i].x,
            truePoses[j].y - truePoses[i].y
          );
          if (dGroundTruth <= params.loopClosureRadius) {
            const alreadyExists = loopEdges.some((e) => e.from === i && e.to === j);
            if (!alreadyExists) {
              const dxLoop = truePoses[j].x - truePoses[i].x;
              const dyLoop = truePoses[j].y - truePoses[i].y;
              const dthLoop = normalizeAngle(truePoses[j].theta - truePoses[i].theta);

              loopEdges.push({
                from: i,
                to: j,
                dx: dxLoop,
                dy: dyLoop,
                dtheta: dthLoop,
                weight: params.informationWeight,
                isLoopClosure: true
              });
              activeLine = 2; // Loop closure registration
            }
          }
        }
      }

      if (numActive === 24) {
        status = 'relaxing';
      }
    } else {
      // Phase 2: Gauss-Newton / spring graph relaxation
      status = 'relaxing';
      relaxCount++;

      // Multiple micro-relaxation sweeps per step for responsive convergence
      const microSweeps = 2;
      let maxStepShift = 0;

      for (let sweep = 0; sweep < microSweeps; sweep++) {
        const nextPositions = estPoses.map((p) => ({ ...p }));

        // Pose 0 is anchored to ground truth truePoses[0]
        nextPositions[0] = { ...truePoses[0] };

        for (let p = 1; p < 24; p++) {
          let sumFx = 0;
          let sumFy = 0;
          let sumFth = 0;
          let sumWeights = 0;

          // Odometry edge constraints incident to node p
          for (const edge of odomEdges) {
            if (edge.to === p) {
              const desiredX = estPoses[edge.from].x + edge.dx;
              const desiredY = estPoses[edge.from].y + edge.dy;
              const desiredTh = estPoses[edge.from].theta + edge.dtheta;
              sumFx += 1.0 * (desiredX - estPoses[p].x);
              sumFy += 1.0 * (desiredY - estPoses[p].y);
              sumFth += 1.0 * normalizeAngle(desiredTh - estPoses[p].theta);
              sumWeights += 1.0;
            } else if (edge.from === p) {
              const desiredX = estPoses[edge.to].x - edge.dx;
              const desiredY = estPoses[edge.to].y - edge.dy;
              const desiredTh = estPoses[edge.to].theta - edge.dtheta;
              sumFx += 1.0 * (desiredX - estPoses[p].x);
              sumFy += 1.0 * (desiredY - estPoses[p].y);
              sumFth += 1.0 * normalizeAngle(desiredTh - estPoses[p].theta);
              sumWeights += 1.0;
            }
          }

          // Loop closure edge constraints incident to node p
          const loopWeight = params.informationWeight;
          for (const edge of loopEdges) {
            if (edge.to === p) {
              const desiredX = estPoses[edge.from].x + edge.dx;
              const desiredY = estPoses[edge.from].y + edge.dy;
              const desiredTh = estPoses[edge.from].theta + edge.dtheta;
              sumFx += loopWeight * (desiredX - estPoses[p].x);
              sumFy += loopWeight * (desiredY - estPoses[p].y);
              sumFth += loopWeight * normalizeAngle(desiredTh - estPoses[p].theta);
              sumWeights += loopWeight;
            } else if (edge.from === p) {
              const desiredX = estPoses[edge.to].x - edge.dx;
              const desiredY = estPoses[edge.to].y - edge.dy;
              const desiredTh = estPoses[edge.to].theta - edge.dtheta;
              sumFx += loopWeight * (desiredX - estPoses[p].x);
              sumFy += loopWeight * (desiredY - estPoses[p].y);
              sumFth += loopWeight * normalizeAngle(desiredTh - estPoses[p].theta);
              sumWeights += loopWeight;
            }
          }

          if (sumWeights > 0) {
            const shiftX = (sumFx / sumWeights) * params.relaxationRate;
            const shiftY = (sumFy / sumWeights) * params.relaxationRate;
            const shiftTh = (sumFth / sumWeights) * params.relaxationRate;

            nextPositions[p].x += shiftX;
            nextPositions[p].y += shiftY;
            nextPositions[p].theta = normalizeAngle(nextPositions[p].theta + shiftTh);

            const shiftMag = Math.hypot(shiftX, shiftY);
            if (shiftMag > maxStepShift) {
              maxStepShift = shiftMag;
            }
          }
        }

        for (let i = 0; i < 24; i++) {
          estPoses[i] = nextPositions[i];
        }
      }

      if (maxStepShift < 0.12 && relaxCount >= 20) {
        status = 'converged';
        isComplete = true;
        activeLine = 6;
      } else {
        activeLine = 4;
      }
    }

    const ateRms = computeAteRms(truePoses, estPoses, numActive);

    return {
      nextState: {
        truePoses,
        estimatedPoses: estPoses,
        odometryEdges: odomEdges,
        loopEdges,
        numActivePoses: numActive,
        status,
        iteration: currentState.iteration + 1,
        relaxationIterations: relaxCount,
        ateRms,
        draggedNodeIndex: currentState.draggedNodeIndex,
        activeLine
      },
      isComplete,
      activePseudocodeLine: activeLine
    };
  },

  onPointerInteraction(
    currentState: PoseGraphState,
    event: CanvasPointerEvent,
    _params: PoseGraphParams
  ): PoseGraphState {
    const estPoses = currentState.estimatedPoses.map((p) => ({ ...p }));
    let draggedIndex = currentState.draggedNodeIndex;

    if (!event.isDragging) {
      let nearestIdx = -1;
      let minDist = 42;
      for (let i = 1; i < estPoses.length; i++) {
        const d = Math.hypot(event.worldX - estPoses[i].x, event.worldY - estPoses[i].y);
        if (d < minDist) {
          minDist = d;
          nearestIdx = i;
        }
      }
      draggedIndex = nearestIdx >= 0 ? nearestIdx : estPoses.length - 1;
    }

    if (draggedIndex !== null && draggedIndex > 0 && draggedIndex < estPoses.length) {
      estPoses[draggedIndex] = {
        ...estPoses[draggedIndex],
        x: Math.max(-270, Math.min(270, event.worldX)),
        y: Math.max(-170, Math.min(170, event.worldY))
      };
    }

    const ateRms = computeAteRms(currentState.truePoses, estPoses, currentState.numActivePoses);

    return {
      ...currentState,
      estimatedPoses: estPoses,
      draggedNodeIndex: draggedIndex,
      relaxationIterations: 0,
      status: currentState.numActivePoses >= 24 ? 'relaxing' : currentState.status,
      ateRms
    };
  },

  getMetrics(currentState: PoseGraphState): MetricReadout[] {
    const ateColor =
      currentState.ateRms < 6.0
        ? 'emerald'
        : currentState.ateRms < 18.0
        ? 'amber'
        : 'rose';

    const statusLabel =
      currentState.status === 'converged'
        ? 'Converged Aligned'
        : currentState.status === 'relaxing'
        ? 'Gauss-Newton Relaxing'
        : `Accumulating (${currentState.numActivePoses}/24)`;

    const statusColor =
      currentState.status === 'converged'
        ? 'emerald'
        : currentState.status === 'relaxing'
        ? 'amber'
        : 'sky';

    return [
      {
        id: 'ate',
        label: 'RMS Trajectory Error (ATE)',
        value: currentState.ateRms.toFixed(2),
        unit: ' px',
        color: ateColor
      },
      {
        id: 'nodes',
        label: 'Active Graph Poses',
        value: `${currentState.numActivePoses}/24`,
        color: 'sky'
      },
      {
        id: 'loops',
        label: 'Loop Closures',
        value: `${currentState.loopEdges.length} detected`,
        color: currentState.loopEdges.length > 0 ? 'amber' : 'default'
      },
      {
        id: 'state',
        label: 'Solver Status',
        value: statusLabel,
        color: statusColor
      }
    ];
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: PoseGraphState,
    _params: PoseGraphParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const cx = viewport.width / 2;
    const cy = viewport.height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Draw Loop Closure Spring Constraints (Glowing Amber #f59e0b)
    for (const loop of state.loopEdges) {
      if (loop.from < state.estimatedPoses.length && loop.to < state.estimatedPoses.length) {
        const p1 = state.estimatedPoses[loop.from];
        const p2 = state.estimatedPoses[loop.to];

        ctx.save();
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        // Draw spring indicator rings at endpoints
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, 4, 0, Math.PI * 2);
        ctx.arc(p2.x, p2.y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Midpoint badge
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        ctx.font = '600 10px Inter, monospace';
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'center';
        ctx.fillText(`Ω_loop=${loop.weight.toFixed(0)}`, midX, midY - 6);
        ctx.restore();
      }
    }

    // 2. Draw Error Residual Lines to Ground Truth (Rose #f43f5e)
    for (let i = 0; i < state.numActivePoses; i++) {
      const gt = state.truePoses[i];
      const est = state.estimatedPoses[i];
      const errDist = Math.hypot(est.x - gt.x, est.y - gt.y);

      if (errDist > 1.5) {
        ctx.beginPath();
        ctx.moveTo(gt.x, gt.y);
        ctx.lineTo(est.x, est.y);
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.55)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 3. Draw Ground Truth Track & Nodes (Emerald #10b981)
    if (state.truePoses.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.truePoses[0].x, state.truePoses[0].y);
      for (let i = 1; i < state.truePoses.length; i++) {
        ctx.lineTo(state.truePoses[i].x, state.truePoses[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    for (let i = 0; i < state.truePoses.length; i++) {
      const p = state.truePoses[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981';
      ctx.fill();

      // Heading tick
      const hx = p.x + 8 * Math.cos(p.theta);
      const hy = p.y + 8 * Math.sin(p.theta);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(hx, hy);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Highlight Anchor Pose 0
    const anchor = state.truePoses[0];
    ctx.beginPath();
    ctx.arc(anchor.x, anchor.y, 7.5, 0, Math.PI * 2);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillStyle = '#a7f3d0';
    ctx.textAlign = 'center';
    ctx.fillText('Anchor (Pose 0)', anchor.x, anchor.y - 12);

    // 4. Draw Estimated Drifting/Relaxed Path (Sky Blue #38bdf8)
    if (state.estimatedPoses.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.estimatedPoses[0].x, state.estimatedPoses[0].y);
      for (let i = 1; i < state.estimatedPoses.length; i++) {
        ctx.lineTo(state.estimatedPoses[i].x, state.estimatedPoses[i].y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    // Draw Estimated Nodes
    for (let i = 0; i < state.estimatedPoses.length; i++) {
      const p = state.estimatedPoses[i];
      const isDragged = i === state.draggedNodeIndex;

      ctx.beginPath();
      ctx.arc(p.x, p.y, isDragged ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = isDragged ? '#f43f5e' : '#0ea5e9';
      ctx.fill();
      ctx.strokeStyle = isDragged ? '#fda4af' : '#e0f2fe';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Heading arrow
      const arrowLen = isDragged ? 14 : 10;
      const hx = p.x + arrowLen * Math.cos(p.theta);
      const hy = p.y + arrowLen * Math.sin(p.theta);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(hx, hy);
      ctx.strokeStyle = isDragged ? '#fda4af' : '#7dd3fc';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (i === state.numActivePoses - 1 && state.status === 'accumulating') {
        ctx.font = '600 10px monospace';
        ctx.fillStyle = '#38bdf8';
        ctx.textAlign = 'center';
        ctx.fillText(`Current x_${i}`, p.x, p.y + 15);
      }
    }

    // 5. Status Banner
    if (state.status === 'converged') {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#10b981';
      ctx.textAlign = 'center';
      ctx.fillText(
        `✓ Pose Graph Aligned (ATE RMS ${state.ateRms.toFixed(2)} px) — Drag any pose node to test dynamic relaxation`,
        0,
        -145
      );
    } else if (state.status === 'relaxing') {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#f59e0b';
      ctx.textAlign = 'center';
      ctx.fillText(
        `⚡ Relaxing Pose Graph: Loop closures pulling drifted nodes into alignment (ATE: ${state.ateRms.toFixed(1)} px)`,
        0,
        -145
      );
    } else {
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      ctx.fillStyle = '#38bdf8';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Accumulating Odometry Drift (${state.numActivePoses}/24 poses) — Loop closures will register at end of loop`,
        0,
        -145
      );
    }

    // Helper Footer
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(
      'Emerald: Ground Truth Stadium | Sky Blue: Estimated Poses | Rose Dashes: Drift Error | Amber: Loop Closures (Ω)',
      0,
      155
    );

    ctx.restore();
  }
};

export default PoseGraphSLAMPlugin;
