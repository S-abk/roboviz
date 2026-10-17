// src/registry/index.ts
import { AlgorithmPlugin, AlgorithmCategory } from '../types/plugin';
import { KalmanFilterPlugin } from '../modules/perception-kalman';
import { ParticleFilterPlugin } from '../modules/perception-particle-filter';
import { ICPPlugin } from '../modules/perception-icp';
import { OccupancyGridPlugin } from '../modules/perception-occupancy-grid';
import { AStarPathfindingPlugin } from '../modules/pathfinding-astar';
import { RRTStarPlugin } from '../modules/motion-planning-rrt-star';
import { PotentialFieldsPlugin } from '../modules/motion-planning-potential-fields';
import { DWAPlugin } from '../modules/motion-planning-dwa';
import { KinematicsIKPlugin } from '../modules/kinematics-ik';
import { PIDControlPlugin } from '../modules/control-pid';
import { HyperRedundantIKPlugin } from '../modules/kinematics-jacobian-ik';
import { PurePursuitPlugin } from '../modules/control-pure-pursuit';
import { RLValueIterationPlugin } from '../modules/rl-value-iteration';
import { QLearningPlugin } from '../modules/rl-q-learning';
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

  // Pillar 2: Plan (Search & Motion)
  AStarPathfindingPlugin,
  RRTStarPlugin,
  PotentialFieldsPlugin,
  DWAPlugin,

  // Pillar 3: Act (Kinematics & Control)
  KinematicsIKPlugin,
  HyperRedundantIKPlugin,
  PIDControlPlugin,
  PurePursuitPlugin,

  // Pillar 4: Learn (RL & AI)
  RLValueIterationPlugin,
  QLearningPlugin,
  SelfAttentionPlugin,
  MLPBackpropPlugin
];
