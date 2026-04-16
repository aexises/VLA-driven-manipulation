# Agent Instructions: Dense Reward GRPO for VLA Manipulation

## Project Overview

This project investigates **integrating dense rewards into Group Relative Policy Optimization (GRPO)** for training Vision-Language-Action (VLA) models on robotic manipulation tasks.

### Primary References
- `README.md` — Original proposal for the three-way reward comparison
- `hypothesis.md` — Canonical mathematical formulation, algorithm pseudocode, experimental plan, and hypothesis testing framework

### Core Hypothesis
Replacing binary (sparse) rewards in GRPO-based VLA training with dense rewards (from ReinboT) or clipped dense rewards will:
1. Improve **sample efficiency** (`hypothesis.md`, Гипотеза 1)
2. Reduce **variance of gradient estimates** via group normalization (`hypothesis.md`, Гипотеза 3)
3. Enable **better temporal credit assignment** (`hypothesis.md`, Гипотеза 1)
4. Dense reward component weights affect trajectory quality and learning speed (`hypothesis.md`, Гипотеза 2)

### Reward Types Under Comparison
This project compares exactly **three** GRPO reward formulations:

1. **Binary (Sparse) Reward** — Original GRPO from SimpleVLA-RL (arXiv:2509.09674). Terminal reward only: $R(\tau) \in \{0, 1\}$ at episode end, zero on all intermediate steps. No shaping, no intermediate signal.

2. **Dense Reward** — Dense reward formulation from ReinboT (Zhang et al., 2025), as specified in `hypothesis.md`:
   $r_t = w_1 r^{\rm sub}_t + w_2 r^{\rm prog}_t + w_3 r^{\rm smooth}_t + w_4 r^{\rm final}_T$
   Provides intermediate step-level signal: sub-goal progress, task progress, action smoothness, and terminal success. Full formulation is in `hypothesis.md` under "Определение плотного вознаграждения" and "Сравнение вариантов плотного вознаграждения".

3. **Clipped Dense Reward** — Same dense reward as (2), but with low values clipped:
   $r^{\rm clipped}_t = \max(r_t, \tau_{\rm clip})$
   where $\tau_{\rm clip} \geq 0$ is a clipping threshold determined experimentally. Candidate values: $\{0, 0.1, 0.2, 0.5\}$.

**Rationale for clipped dense reward:** Clipping low and negative rewards may reduce penalty noise during exploration and stabilize early training. However, it breaks strict potential-based shaping invariance, so empirical comparison against full dense reward is required to assess trade-offs.

---

## Scientific Method Protocol

All work on this project MUST follow this protocol:

### 1. Hypothesis-Driven Development
- Every change must be traceable to a **specific hypothesis** from `hypothesis.md` (Section: "Формулировка тестируемой гипотезы")
- Do NOT implement features or optimizations without linking them to a testable prediction
- Document which hypothesis each experiment validates

### 2. Controlled Experimentation
- **Three-way comparison:** Binary vs Dense vs Clipped Dense — all three MUST be compared under identical conditions
- **Control variables:** Keep hyperparameters, environment, model architecture, and random seeds identical across conditions
- **Multiple runs:** Minimum 5 seeds per condition for statistical significance

### 3. Phased Experimental Approach

#### Phase 1: Validation on Classic RL Environments (MANDATORY FIRST STEP)
**Before any VLA-specific experiments, all three reward types MUST be validated on simple, well-understood RL benchmarks.**

Purpose:
- Prove effectiveness of dense and clipped-dense rewards in a controlled, reproducible setting
- Establish baseline behavior without the complexity of vision-language-action inputs
- Debug algorithm implementation before scaling to VLA tasks

**CRITICAL REQUIREMENT:** Each Phase 1 environment MUST support ALL THREE reward types (Binary, Dense, Clipped Dense) with clearly defined reward functions.

#### Phase 1 Environments and Reward Definitions

| Environment | Binary Reward | Dense Reward | Clipped Dense |
|-------------|---------------|--------------|---------------|
| **CartPole-v1** | +1 if pole upright (|θ| < 12°), 0 otherwise (episode end) | $r_t = +1$ per step alive + $w_1(-\|θ_t\|)$ (angle penalty) + $w_2(-\|x_t\|)$ (cart position penalty) | Same dense, clipped at $\tau_{\rm clip}$ |
| **MountainCar-v0** | +1 if flag reached (x ≥ 0.5), 0 otherwise (episode end) | $r_t = -(0.5 - x_t)$ (distance to goal) + $w_1 \cdot v_t$ (velocity progress) | Same dense, clipped at $\tau_{\rm clip}$ |
| **Acrobot-v1** | +1 if tip reaches target height, 0 otherwise (episode end) | $r_t = -(y_{\rm target} - y_{\rm tip})$ (height progress) + $w_1(-\|θ_t - θ_{\rm target}\|)$ (angle-to-goal) | Same dense, clipped at $\tau_{\rm clip}$ |
| **HalfCheetah-v4** | +1 if forward velocity ≥ threshold, 0 otherwise (episode end) | $r_t = v_t$ (forward velocity) + $w_1(-\|a_t - a_{t-1}\|^2)$ (action smoothness) | Same dense, clipped at $\tau_{\rm clip}$ |
| **Ant-v4** | +1 if forward velocity ≥ threshold, 0 otherwise (episode end) | $r_t = v_t$ (forward velocity) + $w_1(-\|a_t - a_{t-1}\|^2)$ (smoothness) + $w_2(-\|\text{z-coordinate deviation}\|)$ (stability) | Same dense, clipped at $\tau_{\rm clip}$ |

**Notes:**
- Binary reward is always terminal-only (0/1 based on task completion or threshold achievement)
- Dense reward uses ReinboT-style decomposition: progress + smoothness + terminal
- Clipped dense uses same dense formula with $r^{\rm clipped}_t = \max(r_t, \tau_{\rm clip})$, threshold determined experimentally.
- Component weights $w_i$ should be tuned so that magnitudes are comparable across environments

Success criteria for Phase 1:
- Dense reward shows ≥15% improvement in sample efficiency over sparse (measured in episodes to target reward)
- Clipped dense reward demonstrates either: (a) faster convergence than full dense, or (b) comparable final performance with lower gradient variance
- Results are statistically significant (p < 0.05, ≥5 seeds)

#### Phase 2: VLA Manipulation Tasks
Only after Phase 1 success criteria are met:
- LIBERO suite (Spatial, Object, Goal, Long)
- RoboTwin tasks
- CALVIN benchmark tasks
- **SFT only** baseline (supervised fine-tuning without RL) is included for VLA tasks

### 4. Metric Definitions
Track the following metrics for every experiment:
| Metric | Definition |
|--------|------------|
| Success Rate | % of episodes where task goal is achieved (terminal reward = 1) |
| Cumulative Reward | $\mathbb{E}[\sum_t r_t]$ over test episodes |
| Sample Efficiency | Episodes/steps to reach target success threshold (e.g., 80%) |
| Gradient Variance | Variance of advantage estimates across trajectories |
| Trajectory Smoothness | $\mathbb{E}[\|a_t - a_{t-1}\|^2]$ (action jitter) |
| Negative Reward Ratio | % of timesteps where $r_t < 0$ (for analyzing clipped dense effect) |

### 5. Ablation Requirements
The core experiment compares exactly three reward types. Additional ablations:

- **Clipping threshold sweep:** $\tau_{\rm clip} \in \{0, 0.1, 0.2, 0.5\}$ for Clipped Dense only. Select best-performing value for final three-way comparison.
- **Discounted advantage:** With vs without γ-discount in the advantage sum (for all three reward types).
- **Dense reward components:** Toggle individual components from ReinboT Eq. (8): $w_1$ (sub-goal), $w_2$ (progress), $w_3$ (smoothness), $w_4$ (terminal). Test which components contribute most.
- **Group normalization:** With vs without (for Dense and Clipped Dense only; Binary uses standard GRPO normalization).
- **SFT only:** For Phase 2 (VLA tasks) only, include supervised fine-tuning baseline without RL.

### 6. Validation Criteria
A result is considered **significant** only if:
- p-value < 0.05 (t-test or Mann-Whitney U)
- Effect size is practically meaningful (e.g., >10% improvement in sample efficiency)
- Results are reproducible across ≥3 random seeds

---

## Working with Code

### File Structure (Expected)
```
├── README.md                      # Original proposal
├── hypothesis.md                  # Canonical mathematical formulation
├── agents.md                      # This file
└── src/                           # (To be created)
    ├── envs/
    │   ├── classic/               # Classic RL environments (CartPole, MountainCar, etc.)
    │   └── vla/                   # VLA manipulation environments (LIBERO, RoboTwin, etc.)
    ├── models/                    # VLA policy models
    ├── algorithms/
    │   ├── grpo_sparse.py         # Baseline GRPO (from SimpleVLA-RL)
    │   ├── grpo_dense.py          # Proposed Dense-GRPO (ReinboT-style)
    │   ├── grpo_clipped_dense.py  # Clipped Dense GRPO variant (configurable τ_clip)
    │   └── reward_shaping.py      # Dense reward component functions (ReinboT Eq. 8)
    ├── experiments/
    │   ├── run_experiment.py      # Main experiment runner
    │   ├── phase1_classic.py      # Phase 1: Classic RL environment experiments
    │   ├── phase2_vla.py          # Phase 2: VLA manipulation experiments
    │   └── configs/               # YAML configs for each experiment
    └── analysis/
        ├── metrics.py             # Metric computation
        └── plot_results.py        # Visualization
```

### Local and Cloud Data Storage Requirements

**ALL experiment data MUST be stored locally for Phase 1.** For Phase 2+, data MUST be uploaded to cloud storage (local copies are optional but recommended):

- **Phase 1 (classic RL):** Local storage only. Cloud upload optional.
- **Phase 2+ (VLA tasks):** Cloud storage required (e.g., OpenML, HuggingFace Hub, S3). Local copies are optional but recommended for debugging and quick access.

#### Required Logging Infrastructure

1. **TensorBoard** — Primary tool for training visualization. Each experiment run MUST produce its own TensorBoard log directory:
   ```
   runs/
   └── {experiment_name}/
       └── {run_id}_{timestamp}/
           ├── hparams/             # Hyperparameters used
           ├── metrics/             # Scalar metrics (loss, reward, success rate, etc.)
           ├── histograms/          # Weight, gradient, advantage distributions
           └── images/              # Optional: trajectory visualizations, environment screenshots
   ```

2. **CSV/JSON Logs** — Raw metric data stored alongside TensorBoard logs:
   ```
   results/
   └── {experiment_name}/
       ├── config.yaml              # Full experiment configuration
       ├── metrics.csv              # Per-episode/step metrics (flat, parseable)
       ├── summary.json             # Final results: success rate, sample efficiency, statistical tests
       └── raw_trajectories.json    # Optional: full trajectory data for post-hoc analysis
   ```

#### Metrics to Log (Every Experiment)

| Category | Metrics |
|----------|---------|
| **Training** | Loss per step, gradient norm, learning rate, KL divergence |
| **Rewards** | Per-step reward, cumulative episode reward, min/max/mean reward, negative reward ratio |
| **Performance** | Success rate, episode length, sample efficiency (steps to target) |
| **Policy** | Entropy, action distribution stats, trajectory smoothness |
| **Advantages** | Mean/variance of A_t, normalization stats (μ, σ) |
| **System** | Training time, memory usage, GPU utilization |

#### TensorBoard Logging — Required Scalars

At minimum, every experiment MUST log these scalars per step/episode:
- `reward/step_mean`, `reward/episode_cumulative`, `reward/negative_ratio`
- `loss/policy`, `loss/kl`, `loss/total`
- `metrics/success_rate`, `metrics/episode_length`
- `grad/norm`, `grad/advantage_mean`, `grad/advantage_std`
- `hparams/learning_rate`, `hparams/clip_threshold`, `hparams/beta_kl`
- `hparams/reward_type` (binary/dense/clipped_dense), `hparams/tau_clip`

#### Reproducibility Requirements

- **Seed logging:** Random seeds for Python, NumPy, PyTorch/JAX MUST be recorded in `config.yaml`
- **Git hash:** Commit hash of the code used for each experiment MUST be logged
- **Environment snapshot:** Gym/Gymnasium environment version, key dependency versions (saved automatically via `config.yaml`)
- **Determinism check:** Running the same config twice with the same seed MUST produce identical results (within floating-point tolerance)

#### TensorBoard Usage

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"runs/{experiment_name}/{run_id}_{timestamp}")

# Log scalars
writer.add_scalar("reward/episode_cumulative", cum_reward, step=episode)
writer.add_scalar("metrics/success_rate", success_rate, step=episode)
writer.add_scalar("grad/advantage_mean", adv_mean, step=global_step)

# Log hyperparameters
writer.add_hparams(hparam_dict, metric_dict)

writer.close()
```

#### Data Retention

- DO NOT delete experiment logs or results without explicit user approval
- Old runs may be compressed (`tar -czf`) but must remain accessible
- Maintain a `results/EXPERIMENTS.md` index listing all completed experiments with links to their logs
- **Phase 2+ cloud upload:** After each VLA experiment, upload `config.yaml`, `metrics.csv`, `summary.json` to the designated cloud repository. Upload must include metadata: experiment name, phase, reward type, seed, git hash, and timestamp. Verify upload by downloading and comparing checksums.

### Implementation Guidelines

#### 1. Dense Reward Formulation
Follow `hypothesis.md` Equations (1)–(5):
```
R_{i,t} = Σ_{t'=t}^{T_i} γ^{t'-t} r_{i,t'}          (cumulative return)
Â_{i,t} = Σ_{t'=t}^{T_i} R̂_{i,t'}                    (advantage from normalized returns)
```

**Discounted advantage variant:** Optionally, when γ < 1, apply discount to the advantage sum:
```
Â_{i,t} = Σ_{t'=t}^{T_i} γ^{t'-t} R̂_{i,t'}
```
Both versions are tested in ablations.

```
∇J(θ) = E[Σ (A_{i,t} + β(π_ref/π_old - 1)) ∇log π_θ] (policy gradient)
```

#### 2. Clipped Dense Reward Implementation
```
r^{clipped}_t = max(r_t, τ_clip)
```
where:
- $r_t$ is the dense reward from ReinboT (Zhang et al., 2025), as formalized in `hypothesis.md`
- $\tau_{clip}$ is the clipping threshold (configurable; determined experimentally)
- Candidate values: $\{0, 0.1, 0.2, 0.5\}$

**Important:** Clipping breaks strict PBRS policy invariance guarantee. Document this limitation and rely on empirical comparison.

#### 3. Group Normalization (from `hypothesis.md`)
```
μ = mean({R_{j,u}}), σ = std({R_{j,u}})
R̂_{i,t} = (R_{i,t} - μ) / σ
```
Normalize across ALL trajectories in the batch (group size G).

#### 4. GRPO Update
Use clipped surrogate objective (PPO-style) with KL regularization:
```
J(θ) = E[ min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t) - β·D_KL(π_θ || π_ref) ]
```
where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$

---

## Documentation Requirements

### Every Experiment Must Record
1. **Hypothesis tested** (link to specific section in `hypothesis.md`)
2. **Configuration:** hyperparameters, environment, seeds, model architecture
3. **Results:** all metrics (Section 3 above), statistical tests, effect sizes
4. **Observations:** qualitative notes on training behavior, failure modes
5. **Conclusion:** hypothesis supported or rejected, with evidence

### Code Documentation
- Every function computing rewards, advantages, or gradients MUST include the corresponding equation number from `hypothesis.md` or `README.md`
- Example: `# Computes Eq. (1) from hypothesis.md: R_{i,t} = Σ γ^{t'-t} r_{i,t'}`

---

## Decision Framework

When making implementation or experimental decisions:

### Priority Order
1. **Theoretical correctness** — Does this preserve policy invariance? (see README.md §3.3, §6; `hypothesis.md` discussion of bias/invariance)
2. **Empirical validity** — Is this supported by ablation results?
3. **Reproducibility** — Can another agent replicate this exactly?
4. **Simplicity** — Is this the minimal implementation that tests the hypothesis?

### When in Doubt
- Refer to `hypothesis.md` mathematical formulation
- Default to the simplest version that tests the core hypothesis
- Document assumptions explicitly
- Ask the user before making architectural decisions

---

## Quality Checklist

Before committing any work:
- [ ] All formulas in documentation reference equation numbers from source documents
- [ ] Experiments include proper baselines and controls (all THREE reward types compared)
- [ ] Statistical tests are applied correctly (check assumptions: normality, independence)
- [ ] Results are logged with seed, config, and raw metrics
- [ ] TensorBoard logs written to `runs/{experiment_name}/{run_id}/`
- [ ] CSV/JSON results written to `results/{experiment_name}/`
- [ ] Phase 2+ results uploaded to cloud (OpenML / HF Hub / S3) with checksum verification
- [ ] Required scalars logged (reward, loss, metrics, grad stats, hparams)
- [ ] Git commit hash and dependency versions recorded in config
- [ ] Code is reproducible: `python run_experiment.py --config=config.yaml` produces identical results
- [ ] No hardcoded hyperparameters; all configs in YAML/JSON
- [ ] Clipping threshold τ_clip is explicitly documented for Clipped Dense experiments
- [ ] Policy invariance NOT assumed for Dense reward (ReinboT weighted sum ≠ PBRS)
- [ ] Phase 1 (classic RL) experiments completed and validated BEFORE Phase 2 (VLA)
- [ ] Negative reward ratio metric tracked for Clipped Dense analysis
- [ ] `results/EXPERIMENTS.md` index updated with new experiment entry

---

## Key References

| Concept | Source |
|---------|--------|
| Binary reward (GRPO baseline) | SimpleVLA-RL (arXiv:2509.09674) |
| Dense reward formulation | ReinboT — Zhang et al., 2025; `hypothesis.md` |
| Clipped dense reward | agents.md (this file) — clipping threshold determined experimentally |
| GRPO algorithm | `hypothesis.md` "Алгоритм GRPO с плотным вознаграждением" |
| Group normalization | `hypothesis.md` Eqs. (1)–(3) |
| GRPO objective & gradient | `hypothesis.md` Eqs. (4)–(5); Shao et al., 2024 |
| Discounted advantage variant | `hypothesis.md` (advantage with γ-discount option) |
| Phase 1 classic RL validation | agents.md "Phased Experimental Approach" (this file) |
| SFT baseline (Phase 2) | agents.md (this file) |
