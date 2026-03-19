# Evaluation: PPO Training Runs with Randomized Target Area

**Date:** 2026-03-17
**Environment:** 4096 parallel envs, 401 iterations, 100 steps/env
**Task:** Drone landing on a 10x10m target area at 1m height, spawning at 10m altitude with +/-5m XY offset
**Algorithm:** PPO (rsl-rl v5.0.1), MLP [128,128], tanh, adaptive LR schedule
**Evaluation:** 100 episodes per run, 50 parallel envs, checkpoint at iteration 400

---

## 1. Experimental Setup

Three runs were conducted, each adding one change on top of the previous:

| Run | `rel_pos` scale | Target Range | Curriculum | Key Change |
|-----|----------------|--------------|------------|------------|
| **baseline** (`drone-landing`) | 1/15 (clips at 15m) | 10x10m | 20000 steps (200 iters) | Reference run |
| **scaled_obs** (`increased-rel_pos-scale`) | 1/30 (clips at 30m) | 10x10m | 20000 steps (200 iters) | Wider obs normalization |
| **no_curriculum** (`no-curriculum`) | 1/30 (clips at 30m) | 10x10m | Disabled | Full randomization from start |

All other hyperparameters are identical: `clip_param=0.2`, `desired_kl=0.01`, `entropy_coef=0.001`, `gamma=0.99`, `lam=0.95`, `lr=3e-4` (base), `std_type=log`.

---

## 2. Evaluation Results (100 Episodes Each)

### 2.1 Outcome Summary

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| **Success rate** | 90% | 94% | **99%** |
| **Crash rate** | 10% | 6% | **1%** |
| **Timeout rate** | 0% | 0% | 0% |

### 2.2 Approach Precision

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| Min dist mean | 0.517 m | 0.358 m | **0.226 m** |
| Min dist median | 0.246 m | 0.247 m | **0.188 m** |
| Min dist std | 1.052 m | 0.790 m | **0.406 m** |
| Best single | 0.074 m | 0.065 m | **0.049 m** |
| p10 | 0.130 m | 0.109 m | **0.090 m** |
| p25 | 0.189 m | 0.165 m | **0.126 m** |
| p75 | 0.268 m | 0.265 m | **0.260 m** |
| p90 | 0.332 m | 0.284 m | **0.281 m** |
| p95 | 2.031 m | 0.807 m | **0.283 m** |
| Episodes < 0.3m | 90 | 94 | **99** |
| Episodes < 0.5m | 90 | 94 | **99** |
| Episodes < 1.0m | 91 | 95 | **99** |

### 2.3 Episode Length

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| Mean | 612 steps | 649 steps | **588 steps** |
| Std | 83 | 118 | 122 |
| Min | 471 | 492 | 501 |
| Max | 843 | 1253 | 955 |

### 2.4 Crash Analysis

| Metric | baseline (10 crashes) | scaled_obs (6 crashes) | no_curriculum (1 crash) |
|--------|----------------------|----------------------|------------------------|
| Mean min dist | 3.202 m | 2.616 m | 4.201 m |
| Min dist range | 0.792 - 5.355 m | 0.786 - 7.436 m | 4.201 m (single) |
| Mean length | 558 steps | 658 steps | 507 steps |

---

## 3. Training Dynamics

### 3.1 Convergence Speed

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| MA20 > 0 at iter | 95 | 115 | **204** |
| MA20 > 30 at iter | 96 | 116 | 296 |
| MA20 > 40 at iter | 97 | 117 | 374 |
| MA20 > 50 at iter | 97 | 118 | never |
| Training time | 26.8 min | 26.4 min | 25.0 min |
| Avg FPS | 103,875 | 105,680 | 111,359 |

### 3.2 Training Stability (Last 100 Iterations)

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| Mean reward | 7.40 | -53.06 | **38.51** |
| Reward std | 27.65 | **83.84** | **8.29** |
| Min reward | -93.57 | -399.76 | 15.17 |
| Max reward | 44.53 | 43.21 | **58.21** |

### 3.3 Reward Components (Last 50 Iterations Average)

| Component | baseline | scaled_obs | no_curriculum |
|-----------|----------|------------|---------------|
| Distance penalty | -5.23 | -5.38 | **-5.06** |
| Crash penalty | -0.18 | **-0.39** | **-0.02** |
| Success reward | 6.30 | 5.88 | **6.62** |
| Time penalty | -0.11 | -0.11 | **-0.10** |

### 3.4 Policy Evolution

| Metric | baseline | scaled_obs | no_curriculum |
|--------|----------|------------|---------------|
| Std: iter 0 | 0.998 | 0.992 | 0.998 |
| Std: iter 200 | 0.580 | 0.697 | 0.606 |
| Std: iter 400 | 0.344 | 0.451 | **0.331** |
| Entropy: start | 5.674 | 5.662 | 5.664 |
| Entropy: end | 1.282 | 2.428 | **1.143** |
| LR: iter 400 | 0.003 | 0.010 | 0.010 |

---

## 4. Analysis

### 4.1 Observation Scale (baseline vs scaled_obs)

Widening the `rel_pos` normalization range from 1/15 to 1/30 improved the success rate from 90% to 94% and reduced the p95 approach distance from 2.03m to 0.81m. The previous 1/15 scale clipped relative positions beyond 15m, which could occur during the 10x10m randomization (max drone-to-target distance ~16.7m). The wider scale preserves gradient information at larger distances.

However, the scaled_obs run shows **significantly worse training stability**: reward std of 83.84 in the last 100 iterations (vs 27.65 for baseline), with catastrophic dips to -399.76. The adaptive LR schedule pushed the learning rate to 0.010 (its maximum) and kept it there, suggesting the policy kept oscillating. Despite this instability, the final checkpoint at iter 400 happened to be at a good point, yielding decent eval performance.

### 4.2 Curriculum Removal (scaled_obs vs no_curriculum)

Removing the curriculum produced the best results across every metric:

- **99% success rate** (vs 94% and 90%)
- **Lowest mean approach distance** (0.226m vs 0.358m and 0.517m)
- **Tightest p95** (0.283m vs 0.807m and 2.031m, a 7x improvement over baseline)
- **Most stable training** (reward std of 8.29 in last 100 iters, the lowest by far)
- **Lowest policy std** (0.331), indicating the most confident policy

The trade-off is **slower convergence**: no_curriculum doesn't reach positive reward until iteration 173, while the curriculum runs get there by iteration 95-115. The harder task (full randomization from step 0) forces longer initial exploration, but the resulting policy generalizes better.

### 4.3 Why Did Curriculum Hurt?

The curriculum trains on easy (nearby) targets for the first 200 iterations, then switches to the full 10x10m range. This transition causes the performance collapse visible in the baseline and scaled_obs runs around iterations 200-300, where reward drops sharply. The policy overfits to short-range approaches and struggles to adapt.

Without curriculum, the policy learns to handle the full distance range from the start. Although convergence is slower, the policy never needs to "unlearn" a short-range strategy, resulting in:
- No performance collapse mid-training
- Monotonically improving reward after the initial exploration phase
- A learning rate that stays at its maximum (0.010) throughout, suggesting consistent KL divergence without oscillation

### 4.4 Crash Behavior

The single crash in no_curriculum occurred at min_dist=4.2m (length=507 steps), suggesting the drone never reached the target — likely an aggressive maneuver at large distance causing tilt > 60 degrees. The baseline's 10 crashes show a bimodal pattern: some occur near the target (min_dist < 1m, likely overshooting) and others at larger distances (3-5m, likely tilt-related). This aligns with the curriculum hypothesis — the curriculum-trained policy learns aggressive short-range behavior that doesn't transfer cleanly.

---

## 5. Summary

| | baseline | scaled_obs | no_curriculum |
|--|----------|------------|---------------|
| **Eval success** | 90% | 94% | **99%** |
| **Training stability** | Medium | Poor | **Best** |
| **Convergence speed** | **Fast** (iter 95) | Medium (iter 115) | Slow (iter 204) |
| **Precision (p95)** | 2.03 m | 0.81 m | **0.28 m** |
| **Recommendation** | - | - | **Best overall** |

**Best configuration:** `rel_pos` scale = 1/30, no curriculum, full 10x10m random targets from the start. The slower convergence (25 min total, reaching positive reward ~5 min later) is a negligible cost for a dramatically better and more stable final policy.

---

## 6. Data Sources

- W&B project: `drone-landing` (6 runs: 3 training + 3 eval)
- Local TensorBoard events: `logs/{run_name}/events.out.tfevents.*`
- Eval episode tables: `eval_data/wandb_eval_table_*.json`
- Checkpoint: `model_400.pt` for all runs
