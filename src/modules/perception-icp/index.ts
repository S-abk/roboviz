// FILE: src/modules/perception-icp/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface Point2D {
  x: number;
  y: number;
}

export interface Correspondence {
  sourceIdx: number;
  targetIdx: number;
  sourcePt: Point2D;
  targetPt: Point2D;
  dist: number;
  isOutlier: boolean;
}

export type ICPStatus = 'running' | 'aligned' | 'local-minimum' | 'no-inliers';

export interface ICPState {
  iteration: number;
  referenceCloud: Point2D[]; // Emerald (P)
  sourceCloud: Point2D[];    // Sky Blue (Q)
  correspondences: Correspondence[];
  centroidP: Point2D;
  centroidQ: Point2D;
  lastDeltaTheta: number;
  lastDeltaT: Point2D;
  rmsError: number;
  allPointsRmsError: number;
  inlierRatio: number;
  isConverged: boolean;
  status: ICPStatus;
  shapeType: string;
}

export interface ICPParams {
  maxCorrespondenceDist: number;
  dampingFactor: number;
  sensorNoise: number;
  shape: string;
}

function generateShapePoints(type: string): Point2D[] {
  const points: Point2D[] = [];
  if (type === 'corner') {
    for (let x = -140; x <= 0; x += 14) points.push({ x, y: 90 });
    for (let y = 90; y >= -90; y -= 14) points.push({ x: 0, y });
    for (let x = 0; x <= 140; x += 14) points.push({ x, y: -90 });
  } else if (type === 'star') {
    const numPts = 35;
    for (let i = 0; i < numPts; i++) {
      const angle = (i / numPts) * Math.PI * 2;
      const r = i % 2 === 0 ? 110 : 55;
      points.push({ x: r * Math.cos(angle), y: r * Math.sin(angle) });
    }
  } else {
    for (let x = -120; x <= 120; x += 15) points.push({ x, y: 70 });
    for (let x = -70; x <= 50; x += 15) points.push({ x: -120, y: -40 });
    points.length = 0;
    for (let x = -120; x <= 120; x += 15) points.push({ x, y: 70 });
    for (let x = -70; x <= 50; x += 15) points.push({ x, y: -40 });
    for (let y = -40; y <= 70; y += 15) points.push({ x: -120, y });
    for (let y = -40; y <= 70; y += 15) points.push({ x: 120, y });
    for (let a = 0; a <= Math.PI; a += Math.PI / 4) {
      points.push({ x: -65 + 24 * Math.cos(a), y: 70 + 24 * Math.sin(a) });
      points.push({ x: 65 + 24 * Math.cos(a), y: 70 + 24 * Math.sin(a) });
    }
  }
  return points;
}

function transformCloud(pts: Point2D[], angle: number, tx: number, ty: number, noise = 0): Point2D[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return pts.map((p, idx) => {
    const rx = cos * p.x - sin * p.y + tx;
    const ry = sin * p.x + cos * p.y + ty;
    // Deterministic pseudo-random noise so resets are reproducible
    const n1 = Math.sin(idx * 12.9898 + 4.1414) * 43758.5453;
    const n2 = Math.cos(idx * 78.233 + 9.151) * 23421.631;
    const nx = noise > 0 ? ((n1 - Math.floor(n1)) - 0.5) * noise * 2 : 0;
    const ny = noise > 0 ? ((n2 - Math.floor(n2)) - 0.5) * noise * 2 : 0;
    return { x: rx + nx, y: ry + ny };
  });
}

function evaluateAssociations(
  P: Point2D[],
  Q: Point2D[],
  maxDist: number
): {
  correspondences: Correspondence[];
  inlierQ: Point2D[];
  inlierP: Point2D[];
  muQ: Point2D;
  muP: Point2D;
  rmsError: number;
  allPointsRmsError: number;
  inlierRatio: number;
} {
  const correspondences: Correspondence[] = [];
  const inlierQ: Point2D[] = [];
  const inlierP: Point2D[] = [];
  let inlierSumSq = 0;
  let allSumSq = 0;

  for (let i = 0; i < Q.length; i++) {
    const q = Q[i];
    let bestDist = Infinity;
    let bestIdx = 0;

    for (let j = 0; j < P.length; j++) {
      const d = Math.hypot(q.x - P[j].x, q.y - P[j].y);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = j;
      }
    }

    allSumSq += bestDist * bestDist;
    const isOutlier = bestDist > maxDist;
    correspondences.push({
      sourceIdx: i,
      targetIdx: bestIdx,
      sourcePt: { ...q },
      targetPt: { ...P[bestIdx] },
      dist: bestDist,
      isOutlier
    });

    if (!isOutlier) {
      inlierQ.push(q);
      inlierP.push(P[bestIdx]);
      inlierSumSq += bestDist * bestDist;
    }
  }

  const inlierCount = inlierQ.length;
  const inlierRatio = Q.length > 0 ? inlierCount / Q.length : 0;

  // Compute centroids of inliers (or fallback to full cloud centroids if no inliers)
  let sumQx = 0;
  let sumQy = 0;
  let sumPx = 0;
  let sumPy = 0;

  if (inlierCount > 0) {
    for (let i = 0; i < inlierCount; i++) {
      sumQx += inlierQ[i].x;
      sumQy += inlierQ[i].y;
      sumPx += inlierP[i].x;
      sumPy += inlierP[i].y;
    }
  } else {
    for (let i = 0; i < Q.length; i++) {
      sumQx += Q[i].x;
      sumQy += Q[i].y;
    }
    for (let j = 0; j < P.length; j++) {
      sumPx += P[j].x;
      sumPy += P[j].y;
    }
  }

  const denomQ = inlierCount > 0 ? inlierCount : Math.max(1, Q.length);
  const denomP = inlierCount > 0 ? inlierCount : Math.max(1, P.length);
  const muQ = { x: sumQx / denomQ, y: sumQy / denomQ };
  const muP = { x: sumPx / denomP, y: sumPy / denomP };

  const allPointsRmsError = Math.sqrt(allSumSq / Math.max(1, Q.length));
  const rmsError = inlierCount > 0 ? Math.sqrt(inlierSumSq / inlierCount) : allPointsRmsError;

  return {
    correspondences,
    inlierQ,
    inlierP,
    muQ,
    muP,
    rmsError,
    allPointsRmsError,
    inlierRatio
  };
}

export const ICPPlugin: AlgorithmPlugin<ICPState, ICPParams> = {
  id: 'perception-icp',
  title: 'Iterative Closest Point (ICP) Scan Matching',
  shortDescription: 'Rigid 2D point cloud alignment minimizing Euclidean correspondence distances via SVD / centroid alignment.',
  category: 'perception-slam',

  controls: [
    {
      id: 'maxCorrespondenceDist',
      label: 'Max Matching Distance (Threshold)',
      type: 'slider',
      min: 40,
      max: 260,
      step: 10,
      defaultValue: 150,
      unit: 'px',
      description: 'Pairs further than this are rejected as outliers.'
    },
    {
      id: 'dampingFactor',
      label: 'Step Damping Factor (alpha)',
      type: 'slider',
      min: 0.2,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.75,
      description: 'Scales rotation & translation update per iteration to visualize convergence.'
    },
    {
      id: 'sensorNoise',
      label: 'Sensor Measurement Noise',
      type: 'slider',
      min: 0,
      max: 8,
      step: 1,
      defaultValue: 1,
      unit: 'px'
    },
    {
      id: 'shape',
      label: 'Geometry / Scan Profile',
      type: 'select',
      options: [
        { label: 'Robot Chassis', value: 'robot' },
        { label: 'L-Wall Corridor', value: 'corner' },
        { label: 'Stellar Contour', value: 'star' }
      ],
      defaultValue: 'robot'
    }
  ],

  documentation: {
    intuition:
      'Iterative Closest Point (ICP) estimates the rigid transformation (rotation R and translation t) that registers a newly acquired sensor scan Q with a reference scan P. At each iteration, closest-point associations are formed, centroids are aligned, and the optimal rotation is solved in closed form from the cross-covariance matrix H. Because ICP relies on local nearest neighbors, large initial misalignments or tight outlier thresholds can trap it in a local minimum.',
    governingEquationLatex:
      '\\arg\\min_{\\textcolor{#0ea5e9}{R}, \\textcolor{#0ea5e9}{t}} \\sum_{i=1}^{M} \\|\\textcolor{#10b981}{p_{\\pi(i)}} - (\\textcolor{#0ea5e9}{R} \\cdot \\textcolor{#38bdf8}{q_i} + \\textcolor{#0ea5e9}{t})\\|^2, \\quad \\textcolor{#f59e0b}{H} = \\sum_{i=1}^{M} (\\textcolor{#38bdf8}{q_i} - \\bar{q})(\\textcolor{#10b981}{p_{\\pi(i)}} - \\bar{p})^T',
    pseudocode: [
      '1. Associate: For each point q_i in Q, find nearest neighbor p_j in P',
      '2. Trim Outliers: Discard pairings with Euclidean distance > maxCorrespondenceDist',
      '3. Centroids: Compute centers of mass mu_q and mu_p of inlier correspondences',
      '4. Covariance: Compute cross-covariance matrix H = sum (q_i - mu_q)(p_j - mu_p)^T',
      '5. Optimal Rotation: Solve delta_theta = atan2(H_xy - H_yx, H_xx + H_yy)',
      '6. Optimal Translation: Compute delta_t = mu_p - R(delta_theta) * mu_q',
      '7. Update: Transform Q <- alpha * (R * Q + delta_t) and verify global RMS alignment'
    ]
  },

  createInitialState(params: ICPParams): ICPState {
    const shapeType = params.shape || 'robot';
    const referenceCloud = generateShapePoints(shapeType);

    const initialAngle = 0.52;
    const initialTx = 65;
    const initialTy = -48;
    const sourceCloud = transformCloud(
      referenceCloud,
      initialAngle,
      initialTx,
      initialTy,
      params.sensorNoise || 0
    );

    const assoc = evaluateAssociations(
      referenceCloud,
      sourceCloud,
      params.maxCorrespondenceDist || 150
    );

    return {
      iteration: 0,
      referenceCloud,
      sourceCloud,
      correspondences: assoc.correspondences,
      centroidP: assoc.muP,
      centroidQ: assoc.muQ,
      lastDeltaTheta: initialAngle,
      lastDeltaT: { x: initialTx, y: initialTy },
      rmsError: assoc.rmsError,
      allPointsRmsError: assoc.allPointsRmsError,
      inlierRatio: assoc.inlierRatio,
      isConverged: false,
      status: 'running',
      shapeType
    };
  },

  step(currentState: ICPState, params: ICPParams) {
    if (currentState.status !== 'running') {
      return {
        nextState: currentState,
        isComplete: true,
        activePseudocodeLine: 6
      };
    }

    const P = currentState.referenceCloud;
    const Q = currentState.sourceCloud;
    const maxDist = params.maxCorrespondenceDist;
    const damping = params.dampingFactor;

    const { inlierQ, inlierP, muQ, muP, correspondences } = evaluateAssociations(
      P,
      Q,
      maxDist
    );

    const inlierCount = inlierQ.length;
    if (inlierCount < 3) {
      return {
        nextState: {
          ...currentState,
          iteration: currentState.iteration + 1,
          correspondences,
          isConverged: false,
          status: 'no-inliers'
        },
        isComplete: true,
        activePseudocodeLine: 1
      };
    }

    // Compute Cross-Covariance Matrix H
    let Hxx = 0;
    let Hxy = 0;
    let Hyx = 0;
    let Hyy = 0;

    for (let i = 0; i < inlierCount; i++) {
      const qx = inlierQ[i].x - muQ.x;
      const qy = inlierQ[i].y - muQ.y;
      const px = inlierP[i].x - muP.x;
      const py = inlierP[i].y - muP.y;

      Hxx += qx * px;
      Hxy += qx * py;
      Hyx += qy * px;
      Hyy += qy * py;
    }

    // Closed-form 2D SVD rotation angle
    const rawDeltaTheta = Math.atan2(Hxy - Hyx, Hxx + Hyy);
    const deltaTheta = rawDeltaTheta * damping;

    // Centroid-centered damped rotation + translation:
    // q_new = R(deltaTheta) * (q - muQ) + muQ + damping * (muP - muQ)
    const cosT = Math.cos(deltaTheta);
    const sinT = Math.sin(deltaTheta);
    const shiftX = (muP.x - muQ.x) * damping;
    const shiftY = (muP.y - muQ.y) * damping;

    const newQ: Point2D[] = Q.map((q) => {
      const dx = q.x - muQ.x;
      const dy = q.y - muQ.y;
      return {
        x: cosT * dx - sinT * dy + muQ.x + shiftX,
        y: sinT * dx + cosT * dy + muQ.y + shiftY
      };
    });

    const postAssoc = evaluateAssociations(P, newQ, maxDist);
    const nextIter = currentState.iteration + 1;

    const stepMotionSettled =
      Math.abs(rawDeltaTheta) < 0.0025 &&
      Math.hypot(muP.x - muQ.x, muP.y - muQ.y) < 0.45;

    const rmsImprovement = Math.abs(postAssoc.allPointsRmsError - currentState.allPointsRmsError);
    const rmsTolerance = Math.max(4.0, (params.sensorNoise || 0) * 1.85 + 2.0);

    // True global alignment requires high inlier coverage AND low full-cloud RMS error
    const isTrulyAligned =
      postAssoc.inlierRatio >= 0.85 &&
      postAssoc.allPointsRmsError <= rmsTolerance &&
      stepMotionSettled;

    // Stalled in a local minimum if the step motion vanishes or max iterations reached without true alignment
    const isStalledInLocalMin =
      !isTrulyAligned &&
      ((nextIter > 5 && stepMotionSettled && rmsImprovement < 0.01) || nextIter >= 80);

    const status: ICPStatus = isTrulyAligned
      ? 'aligned'
      : isStalledInLocalMin
      ? 'local-minimum'
      : 'running';

    return {
      nextState: {
        iteration: nextIter,
        referenceCloud: P,
        sourceCloud: newQ,
        correspondences: postAssoc.correspondences,
        centroidP: postAssoc.muP,
        centroidQ: postAssoc.muQ,
        lastDeltaTheta: deltaTheta,
        lastDeltaT: { x: shiftX, y: shiftY },
        rmsError: postAssoc.rmsError,
        allPointsRmsError: postAssoc.allPointsRmsError,
        inlierRatio: postAssoc.inlierRatio,
        isConverged: isTrulyAligned,
        status,
        shapeType: currentState.shapeType
      },
      isComplete: status !== 'running',
      activePseudocodeLine: status !== 'running' ? 6 : nextIter % 6
    };
  },

  onPointerInteraction(
    currentState: ICPState,
    event: CanvasPointerEvent,
    params: ICPParams
  ): ICPState {
    const dx = event.worldX;
    const dy = event.worldY;

    // Compute geometric centroid of full source cloud Q
    let cx = 0;
    let cy = 0;
    for (const q of currentState.sourceCloud) {
      cx += q.x;
      cy += q.y;
    }
    cx /= Math.max(1, currentState.sourceCloud.length);
    cy /= Math.max(1, currentState.sourceCloud.length);

    let perturbedQ: Point2D[];
    if (event.shiftKey) {
      // Rotate smoothly around source cloud centroid Q
      const dTheta = 0.14;
      const cos = Math.cos(dTheta);
      const sin = Math.sin(dTheta);
      perturbedQ = currentState.sourceCloud.map((p) => {
        const rx = p.x - cx;
        const ry = p.y - cy;
        return {
          x: cx + cos * rx - sin * ry,
          y: cy + sin * rx + cos * ry
        };
      });
    } else {
      const shiftX = (dx - cx) * 0.55;
      const shiftY = (dy - cy) * 0.55;
      perturbedQ = currentState.sourceCloud.map((p) => ({
        x: p.x + shiftX,
        y: p.y + shiftY
      }));
    }

    const assoc = evaluateAssociations(
      currentState.referenceCloud,
      perturbedQ,
      params.maxCorrespondenceDist
    );

    return {
      ...currentState,
      sourceCloud: perturbedQ,
      correspondences: assoc.correspondences,
      centroidP: assoc.muP,
      centroidQ: assoc.muQ,
      rmsError: assoc.rmsError,
      allPointsRmsError: assoc.allPointsRmsError,
      inlierRatio: assoc.inlierRatio,
      isConverged: false,
      status: 'running',
      iteration: 0
    };
  },

  getMetrics(currentState: ICPState): MetricReadout[] {
    const errorColor =
      currentState.allPointsRmsError < 4.5
        ? 'emerald'
        : currentState.allPointsRmsError < 18.0
        ? 'amber'
        : 'rose';

    const rotDeg = (currentState.lastDeltaTheta * (180 / Math.PI)).toFixed(2);
    const transMag = Math.hypot(currentState.lastDeltaT.x, currentState.lastDeltaT.y).toFixed(1);

    return [
      {
        id: 'rms',
        label: 'Full-Cloud RMS Error',
        value: currentState.allPointsRmsError.toFixed(2),
        unit: ' px',
        color: errorColor
      },
      {
        id: 'inliers',
        label: 'Inlier Ratio',
        value: `${(currentState.inlierRatio * 100).toFixed(0)}%`,
        color: currentState.inlierRatio >= 0.85 ? 'emerald' : currentState.inlierRatio >= 0.4 ? 'amber' : 'rose'
      },
      {
        id: 'deltaTheta',
        label: 'Step Rotation (dTheta)',
        value: `${rotDeg}°`,
        color: 'sky'
      },
      {
        id: 'deltaT',
        label: 'Step Translation (|dt|)',
        value: `${transMag} px`,
        color: 'sky'
      }
    ];
  },

  render(ctx, currentState, _params, viewport) {
    ctx.save();
    ctx.translate(viewport.width / 2, viewport.height / 2);

    // 1. Draw Correspondence Links
    for (const corr of currentState.correspondences) {
      ctx.beginPath();
      ctx.moveTo(corr.sourcePt.x, corr.sourcePt.y);
      ctx.lineTo(corr.targetPt.x, corr.targetPt.y);

      if (corr.isOutlier) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
        ctx.beginPath();
        ctx.moveTo(corr.sourcePt.x - 3, corr.sourcePt.y - 3);
        ctx.lineTo(corr.sourcePt.x + 3, corr.sourcePt.y + 3);
        ctx.moveTo(corr.sourcePt.x + 3, corr.sourcePt.y - 3);
        ctx.lineTo(corr.sourcePt.x - 3, corr.sourcePt.y - 3);
        ctx.stroke();
      } else {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // 2. Draw Reference Cloud P (Emerald #10b981)
    if (currentState.referenceCloud.length > 1) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(currentState.referenceCloud[0].x, currentState.referenceCloud[0].y);
      for (let i = 1; i < currentState.referenceCloud.length; i++) {
        ctx.lineTo(currentState.referenceCloud[i].x, currentState.referenceCloud[i].y);
      }
      ctx.stroke();
    }

    for (const p of currentState.referenceCloud) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981';
      ctx.fill();
      ctx.strokeStyle = '#a7f3d0';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // 3. Draw Source Cloud Q (Sky Blue #0ea5e9 / #38bdf8)
    if (currentState.sourceCloud.length > 1) {
      ctx.strokeStyle = 'rgba(14, 165, 233, 0.45)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(currentState.sourceCloud[0].x, currentState.sourceCloud[0].y);
      for (let i = 1; i < currentState.sourceCloud.length; i++) {
        ctx.lineTo(currentState.sourceCloud[i].x, currentState.sourceCloud[i].y);
      }
      ctx.stroke();
    }

    for (const q of currentState.sourceCloud) {
      ctx.beginPath();
      ctx.arc(q.x, q.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#0ea5e9';
      ctx.fill();
      ctx.strokeStyle = '#e0f2fe';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // 4. Draw Centroids
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(currentState.centroidP.x, currentState.centroidP.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(currentState.centroidQ.x, currentState.centroidQ.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(currentState.centroidQ.x, currentState.centroidQ.y);
    ctx.lineTo(currentState.centroidP.x, currentState.centroidP.y);
    ctx.stroke();

    // 5. Status Banner
    if (currentState.status === 'aligned') {
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#10b981';
      ctx.textAlign = 'center';
      ctx.fillText(
        `✓ Scan Aligned (Iter ${currentState.iteration}, RMS ${currentState.allPointsRmsError.toFixed(1)} px)`,
        0,
        -140
      );
    } else if (currentState.status === 'local-minimum') {
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#f59e0b';
      ctx.textAlign = 'center';
      ctx.fillText(
        `⚠ Trapped in Local Minimum (RMS ${currentState.allPointsRmsError.toFixed(1)} px) — Drag Q closer or raise Threshold`,
        0,
        -140
      );
    } else if (currentState.status === 'no-inliers') {
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.textAlign = 'center';
      ctx.fillText(
        `✕ Too Few Inliers (< 3) — Drag Scan Q closer to P or increase Max Matching Distance`,
        0,
        -140
      );
    }

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Drag on canvas to translate scan Q; Shift+Click/Drag to rotate around centroid', 0, 165);

    ctx.restore();
  }
};
