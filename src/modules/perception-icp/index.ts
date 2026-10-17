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
  inlierRatio: number;
  isConverged: boolean;
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
  return pts.map((p) => {
    const rx = cos * p.x - sin * p.y + tx;
    const ry = sin * p.x + cos * p.y + ty;
    const nx = noise > 0 ? (Math.random() - 0.5) * noise * 2 : 0;
    const ny = noise > 0 ? (Math.random() - 0.5) * noise * 2 : 0;
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
  inlierRatio: number;
} {
  const correspondences: Correspondence[] = [];
  const inlierQ: Point2D[] = [];
  const inlierP: Point2D[] = [];
  let sumSq = 0;

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
      sumSq += bestDist * bestDist;
    }
  }

  const inlierCount = inlierQ.length;
  const inlierRatio = Q.length > 0 ? inlierCount / Q.length : 0;

  let sumQx = 0;
  let sumQy = 0;
  let sumPx = 0;
  let sumPy = 0;

  for (let i = 0; i < inlierCount; i++) {
    sumQx += inlierQ[i].x;
    sumQy += inlierQ[i].y;
    sumPx += inlierP[i].x;
    sumPy += inlierP[i].y;
  }

  const denom = Math.max(1, inlierCount);
  const muQ = { x: sumQx / denom, y: sumQy / denom };
  const muP = { x: sumPx / denom, y: sumPy / denom };
  const rmsError = Math.sqrt(sumSq / denom);

  return { correspondences, inlierQ, inlierP, muQ, muP, rmsError, inlierRatio };
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
      'Iterative Closest Point (ICP) estimates the rigid transformation (rotation R and translation t) that registers a newly acquired sensor scan Q with a reference scan P. At each iteration, closest-point associations are formed, centroids are aligned, and the optimal rotation is solved in closed form from the cross-covariance matrix H.',
    governingEquationLatex:
      '\\arg\\min_{\\textcolor{#0ea5e9}{R}, \\textcolor{#0ea5e9}{t}} \\sum_{i=1}^{M} \\|\\textcolor{#10b981}{p_{\\pi(i)}} - (\\textcolor{#0ea5e9}{R} \\cdot \\textcolor{#38bdf8}{q_i} + \\textcolor{#0ea5e9}{t})\\|^2, \\quad \\textcolor{#f59e0b}{H} = \\sum_{i=1}^{M} (\\textcolor{#38bdf8}{q_i} - \\bar{q})(\\textcolor{#10b981}{p_{\\pi(i)}} - \\bar{p})^T',
    pseudocode: [
      '1. Associate: For each point q_i in Q, find nearest neighbor p_j in P',
      '2. Trim Outliers: Discard pairings with Euclidean distance > maxCorrespondenceDist',
      '3. Centroids: Compute centers of mass mu_q and mu_p of inlier correspondences',
      '4. Covariance: Compute cross-covariance matrix H = sum (q_i - mu_q)(p_j - mu_p)^T',
      '5. Optimal Rotation: Solve delta_theta = atan2(H_xy - H_yx, H_xx + H_yy)',
      '6. Optimal Translation: Compute delta_t = mu_p - R(delta_theta) * mu_q',
      '7. Update: Transform Q <- alpha * (R * Q + delta_t) and compute residual RMS error'
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
      inlierRatio: assoc.inlierRatio,
      isConverged: false,
      shapeType
    };
  },

  step(currentState: ICPState, params: ICPParams) {
    if (currentState.isConverged) {
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

    const { inlierQ, inlierP, muQ, muP, inlierRatio, correspondences } = evaluateAssociations(
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
          isConverged: true
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
    const transMag = Math.hypot(shiftX, shiftY);
    const nextIter = currentState.iteration + 1;

    const isConverged =
      (Math.abs(rawDeltaTheta) < 0.002 && Math.hypot(muP.x - muQ.x, muP.y - muQ.y) < 0.4) ||
      nextIter >= 50 ||
      (nextIter > 3 && Math.abs(postAssoc.rmsError - currentState.rmsError) < 0.005);

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
        inlierRatio,
        isConverged,
        shapeType: currentState.shapeType
      },
      isComplete: isConverged,
      activePseudocodeLine: isConverged ? 6 : nextIter % 6
    };
  },

  onPointerInteraction(
    currentState: ICPState,
    event: CanvasPointerEvent,
    params: ICPParams
  ): ICPState {
    const dx = event.worldX;
    const dy = event.worldY;

    let perturbedQ: Point2D[];
    if (event.shiftKey) {
      const angle = Math.atan2(dy, dx) * 0.25;
      perturbedQ = transformCloud(currentState.sourceCloud, angle, 0, 0);
    } else {
      const shiftX = (dx - currentState.centroidQ.x) * 0.45;
      const shiftY = (dy - currentState.centroidQ.y) * 0.45;
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
      inlierRatio: assoc.inlierRatio,
      isConverged: false,
      iteration: 0
    };
  },

  getMetrics(currentState: ICPState): MetricReadout[] {
    const errorColor =
      currentState.rmsError < 3.0
        ? 'emerald'
        : currentState.rmsError < 15.0
        ? 'amber'
        : 'rose';

    const rotDeg = (currentState.lastDeltaTheta * (180 / Math.PI)).toFixed(2);
    const transMag = Math.hypot(currentState.lastDeltaT.x, currentState.lastDeltaT.y).toFixed(1);

    return [
      {
        id: 'rms',
        label: 'RMS Error',
        value: currentState.rmsError.toFixed(2),
        unit: ' px',
        color: errorColor
      },
      {
        id: 'inliers',
        label: 'Inlier Ratio',
        value: `${(currentState.inlierRatio * 100).toFixed(0)}%`,
        color: currentState.inlierRatio > 0.8 ? 'emerald' : 'amber'
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
        ctx.lineTo(corr.sourcePt.x - 3, corr.sourcePt.y + 3);
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

    // 5. Convergence Badge
    if (currentState.isConverged) {
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#10b981';
      ctx.textAlign = 'center';
      ctx.fillText(`Scan Aligned (Iter ${currentState.iteration})`, 0, -140);
    }

    ctx.font = '10px monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Drag on canvas to misalign scan Q; hold Shift to rotate', 0, 165);

    ctx.restore();
  }
};
