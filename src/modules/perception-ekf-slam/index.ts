// FILE: src/modules/perception-ekf-slam/index.ts
import { AlgorithmPlugin, CanvasPointerEvent, MetricReadout } from '../../types/plugin';

export interface LandmarkState {
  id: number;
  trueX: number;
  trueY: number;
  estX: number;
  estY: number;
  observed: boolean;
  cov: [[number, number], [number, number]];
  corrWithRobot: number;
}

export interface EKFSLAMState {
  stepCount: number;
  lapCount: number;
  // Ground truth robot pose
  trueRobot: { x: number; y: number; theta: number };
  // Estimated robot pose (extracted from mu[0..2])
  estRobot: { x: number; y: number; theta: number };
  // Full joint state vector mu (3 robot states + 2 * 8 landmark states = 19 elements)
  mu: number[];
  // Full joint covariance matrix Sigma (19 x 19)
  sigma: number[][];
  // 8 Static Landmarks
  landmarks: LandmarkState[];
  // Trajectory breadcrumbs
  trueTrajectory: { x: number; y: number }[];
  estTrajectory: { x: number; y: number }[];
  // Active sensor measurement rays
  activeRays: {
    fromX: number;
    fromY: number;
    toX: number;
    toY: number;
    range: number;
    bearing: number;
    landmarkId: number;
  }[];
  // Trajectory RMS error (ATE)
  ateRms: number;
  // Robot positional 2-sigma radius (px)
  robotSigmaRadius: number;
  // Loop closure indicator counter (for glowing animations)
  loopClosureFlash: number;
  statusText: string;
  activeLine: number;
  draggedLandmarkIndex: number | null;
}

export interface EKFSLAMParams {
  sensorRange: number;
  motionNoise: number;
  measurementNoise: number;
  showCorrelations: boolean;
}

const NUM_LANDMARKS = 8;
const STATE_DIM = 3 + 2 * NUM_LANDMARKS; // 19

const DEFAULT_LANDMARKS: { id: number; x: number; y: number }[] = [
  { id: 0, x: -115, y: -130 },
  { id: 1, x: 0, y: -135 },
  { id: 2, x: 115, y: -130 },
  { id: 3, x: 215, y: 0 },
  { id: 4, x: 115, y: 130 },
  { id: 5, x: 0, y: 135 },
  { id: 6, x: -115, y: 130 },
  { id: 7, x: -215, y: 0 }
];

function normalizeAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Computes ground truth position along an elongated stadium circuit.
 * Straight length: 200 px, Curve radius: 75 px.
 */
function getStadiumPose(distanceAlongTrack: number): { x: number; y: number; theta: number } {
  const straight = 200;
  const radius = 75;
  const halfStraight = straight / 2; // 100
  const curveLen = Math.PI * radius; // ~235.6
  const totalPerimeter = 2 * straight + 2 * curveLen; // ~871.24

  let s = distanceAlongTrack % totalPerimeter;
  if (s < 0) s += totalPerimeter;

  if (s < straight) {
    // 1. Top straight moving East
    return {
      x: -halfStraight + s,
      y: -radius,
      theta: 0
    };
  } else if (s < straight + curveLen) {
    // 2. Right curve centered at (halfStraight, 0)
    const arc = s - straight;
    const phi = -Math.PI / 2 + arc / radius;
    return {
      x: halfStraight + radius * Math.cos(phi),
      y: radius * Math.sin(phi),
      theta: normalizeAngle(phi + Math.PI / 2)
    };
  } else if (s < 2 * straight + curveLen) {
    // 3. Bottom straight moving West
    const sStraight = s - (straight + curveLen);
    return {
      x: halfStraight - sStraight,
      y: radius,
      theta: Math.PI
    };
  } else {
    // 4. Left curve centered at (-halfStraight, 0)
    const arc = s - (2 * straight + curveLen);
    const phi = Math.PI / 2 + arc / radius;
    return {
      x: -halfStraight + radius * Math.cos(phi),
      y: radius * Math.sin(phi),
      theta: normalizeAngle(phi + Math.PI / 2)
    };
  }
}

/**
 * Computes 2-sigma ellipse parameters (radii and rotation angle) from 2x2 covariance.
 */
function getEllipseParameters(cov: [[number, number], [number, number]]): {
  r1: number;
  r2: number;
  angle: number;
} {
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const trace = a + c;
  const det = a * c - b * b;
  const disc = Math.max(0, (trace * trace) / 4 - det);
  const sqrtDisc = Math.sqrt(disc);
  const lambda1 = Math.max(0.1, trace / 2 + sqrtDisc);
  const lambda2 = Math.max(0.1, trace / 2 - sqrtDisc);
  const angle = 0.5 * Math.atan2(2 * b, a - c);

  return {
    r1: 2 * Math.sqrt(lambda1), // 2-sigma semi-major
    r2: 2 * Math.sqrt(lambda2), // 2-sigma semi-minor
    angle
  };
}

export const EKFSLAMPlugin: AlgorithmPlugin<EKFSLAMState, EKFSLAMParams> = {
  id: 'perception-ekf-slam',
  title: 'Extended Kalman Filter (EKF) SLAM',
  shortDescription:
    'Joint robot-landmark state and covariance matrix estimation where cross-correlations shrink all landmark uncertainty ellipses upon loop closure.',
  category: 'perception-slam',

  controls: [
    {
      id: 'sensorRange',
      label: 'Sensor Range',
      type: 'slider',
      min: 80,
      max: 240,
      step: 10,
      defaultValue: 145,
      unit: 'px',
      description: 'Maximum range of the range-bearing landmark sensor.'
    },
    {
      id: 'motionNoise',
      label: 'Motion Process Noise (Q)',
      type: 'slider',
      min: 0.01,
      max: 0.15,
      step: 0.01,
      defaultValue: 0.05,
      description: 'Odometry process noise Q injected during motion prediction.'
    },
    {
      id: 'measurementNoise',
      label: 'Measurement Sensor Noise (R)',
      type: 'slider',
      min: 0.02,
      max: 0.3,
      step: 0.02,
      defaultValue: 0.08,
      description: 'Range-bearing observation noise R during Kalman updates.'
    },
    {
      id: 'showCorrelations',
      label: 'Show Covariance Cross-Correlations',
      type: 'toggle',
      defaultValue: true,
      description: 'Render off-diagonal covariance correlation links between robot and landmarks.'
    }
  ],

  documentation: {
    intuition:
      'Simultaneous Localization and Mapping (SLAM) poses a classic chicken-and-egg dilemma: a robot requires an accurate map to localize itself, yet needs precise self-localization to map unknown landmarks. EKF SLAM solves this by stacking the robot pose [x, y, θ] and all landmark coordinates into a unified state vector μ, tracked by a monolithic joint covariance matrix Σ. As the robot traverses unmapped territory, process noise Q causes robot pose uncertainty to inflate and drift. When observing landmarks, off-diagonal cross-covariance blocks develop between the robot and every landmark, and among the landmarks themselves. Crucially, when completing a circuit and re-observing an early anchor landmark with low uncertainty, the Kalman gain propagates through these cross-correlation channels, simultaneously snapping the robot and previously drifted landmarks back toward ground truth while collapsing their uncertainty ellipses!',
    governingEquationLatex:
      '\\textcolor{#38bdf8}{\\mathbf{K}_t} = \\textcolor{#38bdf8}{\\mathbf{\\Sigma}_{t|t-1}}\\mathbf{H}_t^T(\\mathbf{H}_t\\textcolor{#38bdf8}{\\mathbf{\\Sigma}_{t|t-1}}\\mathbf{H}_t^T + \\textcolor{#f43f5e}{\\mathbf{R}_t})^{-1}, \\quad \\textcolor{#38bdf8}{\\boldsymbol{\\mu}_t} = \\textcolor{#38bdf8}{\\boldsymbol{\\mu}_{t|t-1}} + \\mathbf{K}_t(\\textcolor{#f59e0b}{\\mathbf{z}_t} - \\hat{\\mathbf{z}}_t)',
    pseudocode: [
      '1. Odometry Prediction: μ_R <- f(μ_R, u_t) with nonlinear kinematics',
      '2. Covariance Growth: Σ <- F_x Σ F_x^T + F_x^T Q_t F_x (robot uncertainty expands)',
      '3. Landmark Detection: Scan sensorRange for range r_t and bearing φ_t observations',
      '4. Landmark Initialization: Project new landmarks into state μ with Jacobian J_R, J_z',
      '5. Innovation & Jacobians: Calculate residual ν = z - h(μ) and measurement Jacobian H_t',
      '6. Kalman Gain: K_t = Σ H_t^T (H_t Σ H_t^T + R_t)^{-1} across robot & all landmarks',
      '7. Global Cross-Update: μ <- μ + K_t ν, Σ <- (I - K_t H_t) Σ (loop closure shrinks ellipses)'
    ]
  },

  createInitialState(_params: EKFSLAMParams): EKFSLAMState {
    const trueRobot = getStadiumPose(0);
    const estRobot = { ...trueRobot };

    // Initialize 19-dimensional state vector
    const mu = new Array(STATE_DIM).fill(0);
    mu[0] = estRobot.x;
    mu[1] = estRobot.y;
    mu[2] = estRobot.theta;

    // Initialize 19x19 covariance matrix
    const sigma: number[][] = Array.from({ length: STATE_DIM }, () =>
      new Array(STATE_DIM).fill(0)
    );

    // Initial small uncertainty for anchored robot pose
    sigma[0][0] = 0.5;
    sigma[1][1] = 0.5;
    sigma[2][2] = 0.001;

    // Unobserved landmarks start with large uncertainty
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const idxX = 3 + 2 * i;
      const idxY = 4 + 2 * i;
      sigma[idxX][idxX] = 1200;
      sigma[idxY][idxY] = 1200;
    }

    const landmarks: LandmarkState[] = DEFAULT_LANDMARKS.map((lm) => ({
      id: lm.id,
      trueX: lm.x,
      trueY: lm.y,
      estX: lm.x,
      estY: lm.y,
      observed: false,
      cov: [
        [1200, 0],
        [0, 1200]
      ],
      corrWithRobot: 0
    }));

    return {
      stepCount: 0,
      lapCount: 0,
      trueRobot,
      estRobot,
      mu,
      sigma,
      landmarks,
      trueTrajectory: [{ x: trueRobot.x, y: trueRobot.y }],
      estTrajectory: [{ x: estRobot.x, y: estRobot.y }],
      activeRays: [],
      ateRms: 0,
      robotSigmaRadius: 2 * Math.sqrt(0.5),
      loopClosureFlash: 0,
      statusText: 'Lap 1: Traversing loop & mapping landmarks',
      activeLine: 0,
      draggedLandmarkIndex: null
    };
  },

  step(currentState: EKFSLAMState, params: EKFSLAMParams) {
    const step = currentState.stepCount + 1;
    const stepDist = 11.0;
    const currentDist = step * stepDist;

    // 1. Ground Truth Motion
    const prevTrue = currentState.trueRobot;
    const nextTrue = getStadiumPose(currentDist);
    const dTrueDist = Math.hypot(nextTrue.x - prevTrue.x, nextTrue.y - prevTrue.y);
    const dTrueTheta = normalizeAngle(nextTrue.theta - prevTrue.theta);

    // 2. Odometry Process Noise Prediction
    const motionNoiseFactor = params.motionNoise / 0.05;
    const noisyDist =
      dTrueDist + gaussianRandom() * 0.8 * motionNoiseFactor + 0.15 * motionNoiseFactor;
    const noisyTheta =
      dTrueTheta + gaussianRandom() * 0.015 * motionNoiseFactor + 0.004 * motionNoiseFactor;

    // Clone state vector and covariance
    const mu = [...currentState.mu];
    const sigma = currentState.sigma.map((row) => [...row]);

    // Update robot pose estimate in mu
    const curTheta = mu[2];
    const halfDTheta = noisyTheta / 2;
    mu[0] += noisyDist * Math.cos(curTheta + halfDTheta);
    mu[1] += noisyDist * Math.sin(curTheta + halfDTheta);
    mu[2] = normalizeAngle(curTheta + noisyTheta);

    // Robot Motion Jacobian F_R (3x3)
    const fR = [
      [1, 0, -noisyDist * Math.sin(curTheta + halfDTheta)],
      [0, 1, noisyDist * Math.cos(curTheta + halfDTheta)],
      [0, 0, 1]
    ];

    // Odometry Process Noise Q (3x3)
    const posVar = Math.pow(motionNoiseFactor * 1.5, 2);
    const rotVar = Math.pow(motionNoiseFactor * 0.025, 2);
    const Q = [
      [posVar, 0, 0],
      [0, posVar, 0],
      [0, 0, rotVar]
    ];

    // Propagate Robot-Robot Covariance Sigma_RR <- F_R * Sigma_RR * F_R^T + Q
    const sigmaRR = [
      [sigma[0][0], sigma[0][1], sigma[0][2]],
      [sigma[1][0], sigma[1][1], sigma[1][2]],
      [sigma[2][0], sigma[2][1], sigma[2][2]]
    ];

    // F_R * Sigma_RR
    const fSig: number[][] = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        fSig[r][c] =
          fR[r][0] * sigmaRR[0][c] + fR[r][1] * sigmaRR[1][c] + fR[r][2] * sigmaRR[2][c];
      }
    }

    // (F_R * Sigma_RR) * F_R^T + Q
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        sigma[r][c] =
          fSig[r][0] * fR[c][0] +
          fSig[r][1] * fR[c][1] +
          fSig[r][2] * fR[c][2] +
          Q[r][c];
      }
    }

    // Propagate Robot-Landmark cross-covariances Sigma_R,m_i <- F_R * Sigma_R,m_i
    for (let col = 3; col < STATE_DIM; col++) {
      const colVec = [sigma[0][col], sigma[1][col], sigma[2][col]];
      const updated0 = fR[0][0] * colVec[0] + fR[0][1] * colVec[1] + fR[0][2] * colVec[2];
      const updated1 = fR[1][0] * colVec[0] + fR[1][1] * colVec[1] + fR[1][2] * colVec[2];
      const updated2 = fR[2][0] * colVec[0] + fR[2][1] * colVec[1] + fR[2][2] * colVec[2];

      sigma[0][col] = updated0;
      sigma[1][col] = updated1;
      sigma[2][col] = updated2;
      // Maintain symmetry
      sigma[col][0] = updated0;
      sigma[col][1] = updated1;
      sigma[col][2] = updated2;
    }

    // 3. Sensor Sweep & Measurement Updates
    const activeRays: EKFSLAMState['activeRays'] = [];
    const landmarks = currentState.landmarks.map((lm) => ({ ...lm }));
    let activeLine = 1;
    let loopClosureHappened = false;

    // Measurement noise parameters
    const measNoiseFactor = params.measurementNoise / 0.08;
    const sigmaR = 3.5 * measNoiseFactor;
    const sigmaPhi = 0.04 * measNoiseFactor;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lm = landmarks[i];
      const dxTrue = lm.trueX - nextTrue.x;
      const dyTrue = lm.trueY - nextTrue.y;
      const trueDist = Math.hypot(dxTrue, dyTrue);

      if (trueDist <= params.sensorRange) {
        // Generate noisy observation z = [r, phi]^T
        const trueBearing = normalizeAngle(Math.atan2(dyTrue, dxTrue) - nextTrue.theta);
        const zR = Math.max(1, trueDist + gaussianRandom() * sigmaR);
        const zPhi = normalizeAngle(trueBearing + gaussianRandom() * sigmaPhi);

        activeRays.push({
          fromX: mu[0],
          fromY: mu[1],
          toX: lm.trueX,
          toY: lm.trueY,
          range: zR,
          bearing: zPhi,
          landmarkId: lm.id
        });

        const idxX = 3 + 2 * i;
        const idxY = 4 + 2 * i;

        if (!lm.observed) {
          // 4. Landmark Initialization: project into state vector
          activeLine = 3;
          lm.observed = true;
          const globalAngle = normalizeAngle(mu[2] + zPhi);
          mu[idxX] = mu[0] + zR * Math.cos(globalAngle);
          mu[idxY] = mu[1] + zR * Math.sin(globalAngle);

          // Landmark Jacobian w.r.t robot pose J_R (2x3)
          const jR = [
            [1, 0, -zR * Math.sin(globalAngle)],
            [0, 1, zR * Math.cos(globalAngle)]
          ];
          // Landmark Jacobian w.r.t sensor z J_z (2x2)
          const jZ = [
            [Math.cos(globalAngle), -zR * Math.sin(globalAngle)],
            [Math.sin(globalAngle), zR * Math.cos(globalAngle)]
          ];
          const rMat = [
            [sigmaR * sigmaR, 0],
            [0, sigmaPhi * sigmaPhi]
          ];

          // Sigma_mm = J_R * Sigma_RR * J_R^T + J_Z * R * J_Z^T
          const jRSigRR: number[][] = [
            [0, 0, 0],
            [0, 0, 0]
          ];
          for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 3; c++) {
              jRSigRR[r][c] =
                jR[r][0] * sigma[0][c] + jR[r][1] * sigma[1][c] + jR[r][2] * sigma[2][c];
            }
          }

          for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 2; c++) {
              let term1 = 0;
              let term2 = 0;
              for (let k = 0; k < 3; k++) term1 += jRSigRR[r][k] * jR[c][k];
              for (let k = 0; k < 2; k++) term2 += jZ[r][k] * rMat[k][k] * jZ[c][k];
              const val = term1 + term2;
              sigma[idxX + r][idxX + c] = val;
            }
          }

          // Cross-covariances Sigma_Rm = Sigma_RR * J_R^T
          for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 2; c++) {
              let val = 0;
              for (let k = 0; k < 3; k++) val += sigma[r][k] * jR[c][k];
              sigma[r][idxX + c] = val;
              sigma[idxX + c][r] = val;
            }
          }

          // Cross-covariances with other observed landmarks: Sigma_m_j, m_i = Sigma_m_j, R * J_R^T
          for (let j = 0; j < NUM_LANDMARKS; j++) {
            if (j !== i && landmarks[j].observed) {
              const otherX = 3 + 2 * j;
              const otherY = 4 + 2 * j;
              for (let oj = 0; oj < 2; oj++) {
                const oIdx = oj === 0 ? otherX : otherY;
                for (let c = 0; c < 2; c++) {
                  let val = 0;
                  for (let k = 0; k < 3; k++) val += sigma[oIdx][k] * jR[c][k];
                  sigma[oIdx][idxX + c] = val;
                  sigma[idxX + c][oIdx] = val;
                }
              }
            }
          }
        } else {
          // 5 & 6. EKF Measurement Update on re-observed landmark
          activeLine = 5;

          // Check if re-observing Landmark 0 after completing a lap (loop closure event!)
          if (i === 0 && step > 45 && sigma[0][0] > 18) {
            loopClosureHappened = true;
            activeLine = 6;
          }

          const dxHat = mu[idxX] - mu[0];
          const dyHat = mu[idxY] - mu[1];
          const q = Math.max(1, dxHat * dxHat + dyHat * dyHat);
          const sqrtQ = Math.sqrt(q);
          const zHatR = sqrtQ;
          const zHatPhi = normalizeAngle(Math.atan2(dyHat, dxHat) - mu[2]);

          // Innovation vector nu = z - z_hat
          const nuR = zR - zHatR;
          const nuPhi = normalizeAngle(zPhi - zHatPhi);

          // Measurement Jacobian H (2 x 19)
          const hRow0 = new Array(STATE_DIM).fill(0);
          const hRow1 = new Array(STATE_DIM).fill(0);

          hRow0[0] = -dxHat / sqrtQ;
          hRow0[1] = -dyHat / sqrtQ;
          hRow0[2] = 0;
          hRow0[idxX] = dxHat / sqrtQ;
          hRow0[idxY] = dyHat / sqrtQ;

          hRow1[0] = dyHat / q;
          hRow1[1] = -dxHat / q;
          hRow1[2] = -1;
          hRow1[idxX] = -dyHat / q;
          hRow1[idxY] = dxHat / q;

          // P_H = H * Sigma (2 x 19)
          const pH0 = new Array(STATE_DIM).fill(0);
          const pH1 = new Array(STATE_DIM).fill(0);
          for (let c = 0; c < STATE_DIM; c++) {
            pH0[c] =
              hRow0[0] * sigma[0][c] +
              hRow0[1] * sigma[1][c] +
              hRow0[idxX] * sigma[idxX][c] +
              hRow0[idxY] * sigma[idxY][c];
            pH1[c] =
              hRow1[0] * sigma[0][c] +
              hRow1[1] * sigma[1][c] +
              hRow1[2] * sigma[2][c] +
              hRow1[idxX] * sigma[idxX][c] +
              hRow1[idxY] * sigma[idxY][c];
          }

          // Innovation Covariance S = H * Sigma * H^T + R (2 x 2)
          let s00 =
            pH0[0] * hRow0[0] +
            pH0[1] * hRow0[1] +
            pH0[idxX] * hRow0[idxX] +
            pH0[idxY] * hRow0[idxY] +
            sigmaR * sigmaR;
          let s01 =
            pH0[0] * hRow1[0] +
            pH0[1] * hRow1[1] +
            pH0[2] * hRow1[2] +
            pH0[idxX] * hRow1[idxX] +
            pH0[idxY] * hRow1[idxY];
          let s10 = s01;
          let s11 =
            pH1[0] * hRow1[0] +
            pH1[1] * hRow1[1] +
            pH1[2] * hRow1[2] +
            pH1[idxX] * hRow1[idxX] +
            pH1[idxY] * hRow1[idxY] +
            sigmaPhi * sigmaPhi;

          const detS = s00 * s11 - s01 * s10;
          if (detS > 1e-9) {
            const invDet = 1.0 / detS;
            const invS00 = s11 * invDet;
            const invS01 = -s01 * invDet;
            const invS10 = -s10 * invDet;
            const invS11 = s00 * invDet;

            // Kalman Gain K = Sigma * H^T * S^-1 = (H * Sigma)^T * S^-1 (19 x 2)
            const kCol0 = new Array(STATE_DIM);
            const kCol1 = new Array(STATE_DIM);
            for (let r = 0; r < STATE_DIM; r++) {
              kCol0[r] = pH0[r] * invS00 + pH1[r] * invS10;
              kCol1[r] = pH0[r] * invS01 + pH1[r] * invS11;
            }

            // State Vector Update: mu <- mu + K * nu
            for (let r = 0; r < STATE_DIM; r++) {
              mu[r] += kCol0[r] * nuR + kCol1[r] * nuPhi;
            }
            mu[2] = normalizeAngle(mu[2]);

            // Covariance Update: Sigma <- Sigma - K * (H * Sigma)
            for (let r = 0; r < STATE_DIM; r++) {
              for (let c = 0; c < STATE_DIM; c++) {
                sigma[r][c] -= kCol0[r] * pH0[c] + kCol1[r] * pH1[c];
              }
            }

            // Symmetrize and ensure positive semi-definiteness on diagonal
            for (let r = 0; r < STATE_DIM; r++) {
              for (let c = r; c < STATE_DIM; c++) {
                const avg = (sigma[r][c] + sigma[c][r]) / 2;
                sigma[r][c] = avg;
                sigma[c][r] = avg;
              }
              if (sigma[r][r] < 0.05) sigma[r][r] = 0.05;
            }
          }
        }
      }
    }

    // Extract updated estimated positions & covariances for landmarks
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const idxX = 3 + 2 * i;
      const idxY = 4 + 2 * i;
      landmarks[i].estX = mu[idxX];
      landmarks[i].estY = mu[idxY];
      landmarks[i].cov = [
        [sigma[idxX][idxX], sigma[idxX][idxY]],
        [sigma[idxY][idxX], sigma[idxY][idxY]]
      ];

      // Normalized correlation with robot position
      const denom = Math.sqrt(
        Math.max(1e-6, (sigma[0][0] + sigma[1][1]) * (sigma[idxX][idxX] + sigma[idxY][idxY]))
      );
      landmarks[i].corrWithRobot = (sigma[0][idxX] + sigma[1][idxY]) / denom;
    }

    const estRobot = { x: mu[0], y: mu[1], theta: mu[2] };

    // Record breadcrumbs
    const trueTraj = [...currentState.trueTrajectory, { x: nextTrue.x, y: nextTrue.y }];
    const estTraj = [...currentState.estTrajectory, { x: estRobot.x, y: estRobot.y }];
    if (trueTraj.length > 100) trueTraj.shift();
    if (estTraj.length > 100) estTraj.shift();

    // Compute Trajectory ATE RMS
    let sumErrSq = 0;
    const len = Math.min(trueTraj.length, estTraj.length);
    for (let i = 0; i < len; i++) {
      const dx = estTraj[i].x - trueTraj[i].x;
      const dy = estTraj[i].y - trueTraj[i].y;
      sumErrSq += dx * dx + dy * dy;
    }
    const ateRms = Math.sqrt(sumErrSq / Math.max(1, len));

    // Robot 2-sigma radius
    const robotParams = getEllipseParameters([
      [sigma[0][0], sigma[0][1]],
      [sigma[1][0], sigma[1][1]]
    ]);
    const robotSigmaRadius = robotParams.r1;

    let loopFlash = Math.max(0, currentState.loopClosureFlash - 1);
    if (loopClosureHappened) {
      loopFlash = 12;
    }

    const totalLapLen = 871.24;
    const lapCount = Math.floor(currentDist / totalLapLen);
    const statusText =
      loopFlash > 0
        ? '⚡ Loop Closure! Cross-correlations collapsed landmark uncertainty'
        : lapCount > 0
        ? `Lap ${lapCount + 1}: Global covariance converged & aligned`
        : `Lap 1: Traversing track (Step ${step})`;

    return {
      nextState: {
        stepCount: step,
        lapCount,
        trueRobot: nextTrue,
        estRobot,
        mu,
        sigma,
        landmarks,
        trueTrajectory: trueTraj,
        estTrajectory: estTraj,
        activeRays,
        ateRms,
        robotSigmaRadius,
        loopClosureFlash: loopFlash,
        statusText,
        activeLine,
        draggedLandmarkIndex: currentState.draggedLandmarkIndex
      },
      isComplete: false,
      activePseudocodeLine: activeLine
    };
  },

  onPointerInteraction(
    currentState: EKFSLAMState,
    event: CanvasPointerEvent,
    _params: EKFSLAMParams
  ): EKFSLAMState {
    const landmarks = currentState.landmarks.map((lm) => ({ ...lm }));
    let dragged = currentState.draggedLandmarkIndex;

    if (!event.isDragging) {
      let nearestIdx = -1;
      let minDist = 35;
      for (let i = 0; i < landmarks.length; i++) {
        const d = Math.hypot(event.worldX - landmarks[i].trueX, event.worldY - landmarks[i].trueY);
        if (d < minDist) {
          minDist = d;
          nearestIdx = i;
        }
      }
      dragged = nearestIdx >= 0 ? nearestIdx : null;
    }

    if (dragged !== null && dragged >= 0 && dragged < landmarks.length) {
      landmarks[dragged].trueX = Math.max(-260, Math.min(260, event.worldX));
      landmarks[dragged].trueY = Math.max(-160, Math.min(160, event.worldY));
    }

    return {
      ...currentState,
      landmarks,
      draggedLandmarkIndex: dragged
    };
  },

  getMetrics(currentState: EKFSLAMState, params: EKFSLAMParams): MetricReadout[] {
    const observedCount = currentState.landmarks.filter((l) => l.observed).length;
    const ateColor =
      currentState.ateRms < 6.0
        ? 'emerald'
        : currentState.ateRms < 16.0
        ? 'amber'
        : 'rose';

    const covColor =
      currentState.robotSigmaRadius < 12.0
        ? 'emerald'
        : currentState.robotSigmaRadius < 26.0
        ? 'amber'
        : 'sky';

    return [
      {
        id: 'ate',
        label: 'Pose Error (ATE RMS)',
        value: currentState.ateRms.toFixed(2),
        unit: ' px',
        color: ateColor
      },
      {
        id: 'robotSigma',
        label: 'Robot Uncertainty (2σ)',
        value: currentState.robotSigmaRadius.toFixed(1),
        unit: ' px',
        color: covColor
      },
      {
        id: 'landmarks',
        label: 'Mapped Landmarks',
        value: `${observedCount} / ${NUM_LANDMARKS}`,
        color: observedCount === NUM_LANDMARKS ? 'emerald' : 'sky'
      },
      {
        id: 'status',
        label: 'Filter Status',
        value:
          currentState.loopClosureFlash > 0
            ? 'Loop Closure Snapped!'
            : currentState.lapCount > 0
            ? 'Loop Closed & Correlated'
            : 'Accumulating Drift',
        color:
          currentState.loopClosureFlash > 0
            ? 'emerald'
            : currentState.lapCount > 0
            ? 'sky'
            : 'amber'
      }
    ];
  },

  render(
    ctx: CanvasRenderingContext2D,
    state: EKFSLAMState,
    params: EKFSLAMParams,
    viewport: { width: number; height: number; dpr: number }
  ): void {
    const cx = viewport.width / 2;
    const cy = viewport.height / 2;

    ctx.save();
    ctx.translate(cx, cy);

    // 1. Draw Stadium Track Boundary (Subtle Slate Outline)
    ctx.beginPath();
    const halfStraight = 100;
    const radius = 75;
    // Top straight
    ctx.moveTo(-halfStraight, -radius);
    ctx.lineTo(halfStraight, -radius);
    // Right semicircle
    ctx.arc(halfStraight, 0, radius, -Math.PI / 2, Math.PI / 2);
    // Bottom straight
    ctx.lineTo(-halfStraight, radius);
    // Left semicircle
    ctx.arc(-halfStraight, 0, radius, Math.PI / 2, (3 * Math.PI) / 2);
    ctx.closePath();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 14;
    ctx.stroke();

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Off-Diagonal Covariance Cross-Correlations (Subtle Sky Blue / Amber Links)
    if (params.showCorrelations) {
      for (let i = 0; i < NUM_LANDMARKS; i++) {
        for (let j = i + 1; j < NUM_LANDMARKS; j++) {
          const lm1 = state.landmarks[i];
          const lm2 = state.landmarks[j];
          if (lm1.observed && lm2.observed) {
            const idx1 = 3 + 2 * i;
            const idx2 = 3 + 2 * j;
            const covVal = state.sigma[idx1][idx2] + state.sigma[idx1 + 1][idx2 + 1];
            const var1 = state.sigma[idx1][idx1] + state.sigma[idx1 + 1][idx1 + 1];
            const var2 = state.sigma[idx2][idx2] + state.sigma[idx2 + 1][idx2 + 1];
            const corr = Math.abs(covVal) / Math.sqrt(Math.max(1e-6, var1 * var2));

            if (corr > 0.22) {
              ctx.beginPath();
              ctx.moveTo(lm1.estX, lm1.estY);
              ctx.lineTo(lm2.estX, lm2.estY);
              ctx.strokeStyle = `rgba(56, 189, 248, ${Math.min(0.65, corr * 0.7)})`;
              ctx.lineWidth = 1.0 + corr * 2.0;
              ctx.stroke();
            }
          }
        }
      }
    }

    // 3. Sensor Field & Range-Bearing Rays (Amber #f59e0b)
    ctx.beginPath();
    ctx.arc(state.estRobot.x, state.estRobot.y, params.sensorRange, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.03)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.2)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    for (const ray of state.activeRays) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(ray.fromX, ray.fromY);
      ctx.lineTo(ray.toX, ray.toY);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.8;
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 8;
      ctx.stroke();

      // Sensor distance badge
      const midX = (ray.fromX + ray.toX) / 2;
      const midY = (ray.fromY + ray.toY) / 2;
      ctx.font = '600 9px monospace';
      ctx.fillStyle = '#fbbf24';
      ctx.textAlign = 'center';
      ctx.fillText(`r=${ray.range.toFixed(0)}`, midX, midY - 4);
      ctx.restore();
    }

    // 4. Trajectories (Emerald Ground Truth vs Sky Blue EKF Estimate)
    if (state.trueTrajectory.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.trueTrajectory[0].x, state.trueTrajectory[0].y);
      for (let i = 1; i < state.trueTrajectory.length; i++) {
        ctx.lineTo(state.trueTrajectory[i].x, state.trueTrajectory[i].y);
      }
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
      ctx.lineWidth = 2.0;
      ctx.stroke();
    }

    if (state.estTrajectory.length > 1) {
      ctx.beginPath();
      ctx.moveTo(state.estTrajectory[0].x, state.estTrajectory[0].y);
      for (let i = 1; i < state.estTrajectory.length; i++) {
        ctx.lineTo(state.estTrajectory[i].x, state.estTrajectory[i].y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }

    // 5. Landmark Covariance Ellipses and Markers
    for (const lm of state.landmarks) {
      const isDragged = lm.id === state.draggedLandmarkIndex;

      // Ground Truth Landmark (Emerald #10b981)
      ctx.beginPath();
      ctx.arc(lm.trueX, lm.trueY, isDragged ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = isDragged ? '#f43f5e' : '#10b981';
      ctx.fill();
      ctx.strokeStyle = isDragged ? '#fda4af' : '#a7f3d0';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label
      ctx.font = '600 10px Inter, sans-serif';
      ctx.fillStyle = '#a7f3d0';
      ctx.textAlign = 'center';
      ctx.fillText(`L${lm.id}`, lm.trueX, lm.trueY - 9);

      if (lm.observed) {
        // Estimated Landmark Uncertainty 2-sigma Ellipse (Amber #f59e0b)
        const ellipse = getEllipseParameters(lm.cov);
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(
          lm.estX,
          lm.estY,
          Math.min(90, Math.max(3, ellipse.r1)),
          Math.min(90, Math.max(3, ellipse.r2)),
          ellipse.angle,
          0,
          Math.PI * 2
        );
        ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
        ctx.fill();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.4;
        ctx.stroke();

        // Estimated position crosshair
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(lm.estX, lm.estY, 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Residual drift line to ground truth
        const err = Math.hypot(lm.estX - lm.trueX, lm.estY - lm.trueY);
        if (err > 2.0) {
          ctx.beginPath();
          ctx.moveTo(lm.trueX, lm.trueY);
          ctx.lineTo(lm.estX, lm.estY);
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.55)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([2, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.restore();
      }
    }

    // 6. Robot Covariance Ellipse (Sky Blue #38bdf8)
    const robotCov: [[number, number], [number, number]] = [
      [state.sigma[0][0], state.sigma[0][1]],
      [state.sigma[1][0], state.sigma[1][1]]
    ];
    const robotEllipse = getEllipseParameters(robotCov);

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(
      state.estRobot.x,
      state.estRobot.y,
      Math.min(120, Math.max(4, robotEllipse.r1)),
      Math.min(120, Math.max(4, robotEllipse.r2)),
      robotEllipse.angle,
      0,
      Math.PI * 2
    );
    ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.restore();

    // Ground Truth Robot Marker (Emerald Triangle #10b981)
    ctx.save();
    ctx.translate(state.trueRobot.x, state.trueRobot.y);
    ctx.rotate(state.trueRobot.theta);
    ctx.beginPath();
    ctx.moveTo(9, 0);
    ctx.lineTo(-6, -5);
    ctx.lineTo(-4, 0);
    ctx.lineTo(-6, 5);
    ctx.closePath();
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#a7f3d0';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();

    // Estimated Robot Marker (Sky Blue Triangle #38bdf8)
    ctx.save();
    ctx.translate(state.estRobot.x, state.estRobot.y);
    ctx.rotate(state.estRobot.theta);
    ctx.beginPath();
    ctx.moveTo(11, 0);
    ctx.lineTo(-7, -6);
    ctx.lineTo(-5, 0);
    ctx.lineTo(-7, 6);
    ctx.closePath();
    ctx.fillStyle = '#0284c7';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.0;
    ctx.stroke();
    ctx.restore();

    // 7. Miniature Live Covariance Heatmap (Top-Right Corner)
    const mapSize = 95;
    const mapX = 175;
    const mapY = -165;
    const cellSize = mapSize / STATE_DIM;

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(mapX - 4, mapY - 18, mapSize + 8, mapSize + 24, 6);
    ctx.fill();
    ctx.stroke();

    ctx.font = '600 9px Inter, sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.textAlign = 'left';
    ctx.fillText('Σ Covariance Heatmap', mapX, mapY - 6);

    for (let r = 0; r < STATE_DIM; r++) {
      for (let c = 0; c < STATE_DIM; c++) {
        const val = state.sigma[r][c];
        const varR = Math.max(1e-4, state.sigma[r][r]);
        const varC = Math.max(1e-4, state.sigma[c][c]);
        const corr = val / Math.sqrt(varR * varC);

        let cellColor = 'rgba(30, 41, 59, 0.6)';
        if (r === c) {
          cellColor = '#38bdf8';
        } else if (corr > 0.05) {
          cellColor = `rgba(56, 189, 248, ${Math.min(0.95, corr * 1.2)})`;
        } else if (corr < -0.05) {
          cellColor = `rgba(244, 63, 94, ${Math.min(0.95, Math.abs(corr) * 1.2)})`;
        }

        ctx.fillStyle = cellColor;
        ctx.fillRect(mapX + c * cellSize, mapY + r * cellSize, cellSize - 0.5, cellSize - 0.5);
      }
    }

    // Grid divider line separating robot block (3x3) and landmarks
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mapX, mapY + 3 * cellSize);
    ctx.lineTo(mapX + mapSize, mapY + 3 * cellSize);
    ctx.moveTo(mapX + 3 * cellSize, mapY);
    ctx.lineTo(mapX + 3 * cellSize, mapY + mapSize);
    ctx.stroke();
    ctx.restore();

    // 8. Loop Closure Flash Glow & Status Banner
    if (state.loopClosureFlash > 0) {
      ctx.save();
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 20;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(state.estRobot.x, state.estRobot.y, state.robotSigmaRadius + 14, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Top Status Banner
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillStyle = state.loopClosureFlash > 0 ? '#10b981' : '#38bdf8';
    ctx.textAlign = 'center';
    ctx.fillText(state.statusText, 0, -148);

    // Help Footer
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(
      'Emerald: Ground Truth | Sky Blue: EKF Estimate & 2σ Ellipse | Amber: Range-Bearing Rays | Drag landmarks to test re-convergence',
      0,
      158
    );

    ctx.restore();
  }
};

export default EKFSLAMPlugin;
