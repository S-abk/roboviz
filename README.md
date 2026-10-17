# RoboViz Atlas — Interactive Visual Encyclopedia of Robotics & AI Algorithms

**Live Demo:** [https://s-abk.github.io/roboviz/](https://s-abk.github.io/roboviz/)

**RoboViz Atlas** is an open-source, zero-backend interactive visual learning platform for foundational **Robotics** and **Artificial Intelligence** algorithms, built with **TypeScript**, **HTML5 2D Canvas**, **KaTeX**, and **Vite**.

Inspired by the tactile interactivity of *Red Blob Games*, the visual intuition of *3Blue1Brown*, and the mathematical rigor of *Probabilistic Robotics*, every module lets you click, drag, perturb, and time-travel through live algorithms while watching the governing equations and pseudocode react in real time.

---

## Color-Synchronized Pedagogy (Canvas $\leftrightarrow$ KaTeX)

Every variable in the rendered KaTeX governing equation is color-synchronized with its exact geometric counterpart on the 2D Canvas:

| Color | Hex | Semantic Meaning Across All 16 Modules |
| :--- | :--- | :--- |
| **Sky Blue** | `#38bdf8` / `#0ea5e9` | Robot pose, posterior state estimate, explored nodes, or kinematic links |
| **Emerald** | `#10b981` | Goal states, ground-truth pose, optimal paths, or positive reward signals |
| **Amber** | `#f59e0b` | Heuristics, active frontiers, sensor rays/measurements, or attention weights |
| **Rose** | `#f43f5e` | Obstacles, collisions, hazards, residual error vectors, or tracking drift |

---

## Complete 20-Module Curriculum

### 1. Sense (Perception & SLAM)
- **1D Kalman Filter (Linear Gaussian State Estimation)** (`#perception-kalman`)
- **Monte Carlo Localization (Particle Filter)** (`#perception-particle-filter`) — $N$ weighted particles, Gaussian sensor likelihoods, low-variance resampling, and interactive robot kidnapping.
- **Iterative Closest Point (ICP) Scan Matching** (`#perception-icp`) — Nearest-neighbor point cloud correspondences and SVD rigid-body alignment.
- **Log-Odds Occupancy Grid Mapping** (`#perception-occupancy-grid`) — 2D grid raycasting with recursive Bayesian log-odds updates and interactive wall painting.
- **Pose-Graph SLAM & Loop Closure** (`#perception-pose-graph-slam`) — Odometry drift accumulation along a loop trajectory corrected globally via loop-closure spring constraints and Gauss-Newton relaxation.

### 2. Plan (Search & Motion Planning)
- **A\* & Weighted Dijkstra Grid Search** (`#motion-planning-astar`) — Configurable heuristic weight $w \in [0, 3]$ and interactive maze drawing.
- **RRT & RRT\* Continuous Motion Planner** (`#motion-planning-rrt-star`) — Sampling-based tree expansion with local neighborhood rewiring and asymptotic optimality.
- **Artificial Potential Fields & Local Minima** (`#motion-planning-potential-fields`) — Attractive/repulsive gradient fields and custom U-shaped local minimum traps.
- **Dynamic Window Approach (DWA) Local Planner** (`#motion-planning-dwa`) — Velocity-space $(v, \omega)$ circular arc rollouts and multi-objective trajectory scoring.
- **Hybrid A\* Non-Holonomic Car Planner** (`#motion-planning-hybrid-astar`) — Continuous $(x, y, \theta)$ Ackermann motion-primitive search respecting minimum turning radius and forward/reverse gear costs.

### 3. Act (Kinematics & Control)
- **2-Link Analytical Inverse Kinematics** (`#kinematics-2d-ik`) — Law of Cosines elbow-up/elbow-down analytical solver with workspace reachability rings.
- **N-Link Hyper-Redundant Arm (CCD & Jacobian IK)** (`#kinematics-jacobian-ik`) — 3-to-8 DOF tentacle manipulator using Cyclic Coordinate Descent.
- **PID Controller & Disturbance Rejection** (`#control-pid`) — Split-screen quadrotor altitude controller with live scrolling oscilloscope and P/I/D effort gauges.
- **Pure Pursuit Autonomous Path Tracking** (`#control-pure-pursuit`) — Kinematic Ackermann bicycle model tracking closed-loop splines via a dynamic lookahead circle $L_d$.
- **Model Predictive Control (MPC) Horizon Planner** (`#control-mpc`) — Receding-horizon trajectory optimization over $N$ future steps with proactive obstacle swerving.

### 4. Learn (Reinforcement Learning & AI)
- **MDP Value Iteration (Bellman Optimality)** (`#rl-value-iteration`) — Synchronous dynamic programming over a stochastic grid-world.
- **Tabular Q-Learning & Cliff-Walking Agent** (`#rl-q-learning`) — Directional $Q(s,a)$ action-value triangles and live $\epsilon$-greedy Temporal-Difference updates.
- **Policy Gradient & Cart-Pole Balancing** (`#rl-cartpole-policy`) — Continuous-state Inverted Pendulum stabilized via online policy gradients with a live $(\theta, \dot{\theta})$ phase-portrait plot.
- **Transformer Scaled Dot-Product Self-Attention** (`#ai-self-attention`) — Interactive 6-token Query-Key-Value attention arcs, temperature scaling, causal masking, and $6 \times 6$ attention matrix.
- **Neural Network Decision Boundary & Backpropagation** (`#ai-mlp-backprop`) — Live 2-layer tanh MLP warping a 2D probability field via gradient descent as you click to add training points.

---

## Keyboard Shortcuts & Controls

- **`Space`**: Play / Pause simulation
- **`←` / `→`**: Step backward / forward in the Time-Travel Scrubber
- **`R`**: Reset simulation state
- **Speed Selector (`0.25x` – `4x`)**: Slow down individual math steps or accelerate convergence
- **Scenario Presets**: One-click buttons at the top of the parameter panel to trigger classic engineering failure modes and tuned benchmarks

---

## Local Development

```bash
npm install
npm run dev
```

To build for production:

```bash
npm run build
npm run preview
```

## Architecture & Adding New Modules

Every module is a pure, self-contained TypeScript file in `src/modules/<module-name>/index.ts` implementing the `AlgorithmPlugin<TState, TParams>` interface from `src/types/plugin.ts`, and registered in `src/registry/index.ts`.
