// src/registry/index.ts
import { AlgorithmPlugin, AlgorithmCategory } from '../types/plugin';
import { KalmanFilterPlugin } from '../modules/perception-kalman';
import { ParticleFilterPlugin } from '../modules/perception-particle-filter';
import { ICPPlugin } from '../modules/perception-icp';
import { OccupancyGridPlugin } from '../modules/perception-occupancy-grid';
import { PoseGraphSLAMPlugin } from '../modules/perception-pose-graph-slam';
import { EKFSLAMPlugin } from '../modules/perception-ekf-slam';

import { AStarPathfindingPlugin } from '../modules/pathfinding-astar';
import { RRTStarPlugin } from '../modules/motion-planning-rrt-star';
import { PRMPlannerPlugin } from '../modules/motion-planning-prm';
import { PotentialFieldsPlugin } from '../modules/motion-planning-potential-fields';
import { DWAPlugin } from '../modules/motion-planning-dwa';
import { HybridAStarPlugin } from '../modules/motion-planning-hybrid-astar';

import { KinematicsIKPlugin } from '../modules/kinematics-ik';
import { HyperRedundantIKPlugin } from '../modules/kinematics-jacobian-ik';
import { PIDControlPlugin } from '../modules/control-pid';
import { PurePursuitPlugin } from '../modules/control-pure-pursuit';
import { LQRControlPlugin } from '../modules/control-lqr';
import { MPCControlPlugin } from '../modules/control-mpc';

import { RLValueIterationPlugin } from '../modules/rl-value-iteration';
import { QLearningPlugin } from '../modules/rl-q-learning';
import { CartPolePolicyPlugin } from '../modules/rl-cartpole-policy';
import { GMMClusteringPlugin } from '../modules/ai-kmeans-gmm';
import { SelfAttentionPlugin } from '../modules/ai-self-attention';
import { MLPBackpropPlugin } from '../modules/ai-mlp-backprop';

export const CATEGORY_LABELS: Record<AlgorithmCategory, string> = {
  'perception-slam': '1. Sense (Perception & SLAM)',
  'motion-planning': '2. Plan (Search & Motion)',
  'kinematics-control': '3. Act (Kinematics & Control)',
  'reinforcement-ai': '4. Learn (RL & AI)'
};

export const PLUGIN_REGISTRY: AlgorithmPlugin<any, any>[] = [
  // Pillar 1: Sense (Perception & SLAM)
  KalmanFilterPlugin,
  ParticleFilterPlugin,
  ICPPlugin,
  OccupancyGridPlugin,
  EKFSLAMPlugin,
  PoseGraphSLAMPlugin,

  // Pillar 2: Plan (Search & Motion)
  AStarPathfindingPlugin,
  RRTStarPlugin,
  PRMPlannerPlugin,
  PotentialFieldsPlugin,
  DWAPlugin,
  HybridAStarPlugin,

  // Pillar 3: Act (Kinematics & Control)
  KinematicsIKPlugin,
  HyperRedundantIKPlugin,
  PIDControlPlugin,
  PurePursuitPlugin,
  LQRControlPlugin,
  MPCControlPlugin,

  // Pillar 4: Learn (RL & AI)
  RLValueIterationPlugin,
  QLearningPlugin,
  CartPolePolicyPlugin,
  GMMClusteringPlugin,
  SelfAttentionPlugin,
  MLPBackpropPlugin
];
