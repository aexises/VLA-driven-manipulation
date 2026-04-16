# Proposal: Comparing Binary, Dense, and Clipped Dense Rewards for GRPO-based VLA Training

## 1. Motivation

The VLA training framework in **SimpleVLA-RL** (arXiv:2509.09674) relies on **binary (sparse) rewards** — a single 0/1 signal at episode end. This introduces:

- High variance in policy gradient estimates
- Poor credit assignment over long-horizon vision-language-action trajectories
- Sample inefficiency

**ReinboT** (Zhang et al., 2025) demonstrates that **dense reward signals** — providing step-level feedback on sub-goal progress, task progress, smoothness, and terminal success — can substantially improve learning speed and final performance.

This project compares exactly **three** reward formulations for GRPO-based VLA training.

---

## 2. Hypothesis

**Replacing binary rewards in GRPO-based VLA training with dense rewards (from ReinboT) or clipped dense rewards will:**

1. Improve **sample efficiency**
2. Reduce **variance of gradient estimates**
3. Enable **better temporal credit assignment**

---

## 3. Three Reward Types Under Comparison

### 3.1 Binary (Sparse) Reward — SimpleVLA-RL Baseline

The original GRPO setup from SimpleVLA-RL:

$$R(\tau) \in \{0, 1\}$$

Reward is given only at episode end: 1 if task succeeded, 0 otherwise. All intermediate steps receive zero reward.

**Advantages:** Simple, no reward design needed, preserves original optimal policy.
**Disadvantages:** High variance, poor credit assignment, sample inefficient.

---

### 3.2 Dense Reward — ReinboT Formulation

Dense reward from ReinboT (Zhang et al., 2025), as formalized in `hypothesis.md`:

$$r_t = w_1 r^{\rm sub}_t + w_2 r^{\rm prog}_t + w_3 r^{\rm smooth}_t + w_4 r^{\rm final}_T$$

| Component | Description |
|-----------|-------------|
| $r^{\rm sub}_t$ | Sub-goal reward: proximity to intermediate targets |
| $r^{\rm prog}_t$ | Task progress: weighted signal based on task stage completion |
| $r^{\rm smooth}_t$ | Action smoothness: penalty on abrupt action changes ($-\|a_t - a_{t-1}\|^2$) |
| $r^{\rm final}_T$ | Terminal success: binary 0/1 at episode end |

Weights $w_i$ are tuned so that component magnitudes are comparable (see ReinboT).

**Advantages:** Rich step-level signal, better credit assignment, faster learning.
**Disadvantages:** Requires reward design; negative components may introduce noise during exploration.

---

### 3.3 Clipped Dense Reward — Proposed Variant

Same dense reward as 3.2, but with low values clipped:

$$r^{\rm clipped}_t = \max(r_t, \tau_{\rm clip})$$

where $\tau_{\rm clip} \geq 0$ is a clipping threshold determined experimentally. Candidate values: $\{0, 0.1, 0.2, 0.5\}$.

**Rationale:** Clipping low and negative rewards may reduce penalty noise during exploration and stabilize early training.
**Trade-off:** Breaks strict potential-based shaping invariance; empirical comparison required.

---

## 4. GRPO Objective

All three reward types use the same GRPO framework. Cumulative returns:

$$R_{i,t} = \sum_{t'=t}^{T_i} \gamma^{t'-t} r_{i,t'}$$

Group normalization:

$$\hat{R}_{i,t} = \frac{R_{i,t} - \mu}{\sigma}, \quad \mu = \text{mean}(\{R_{j,u}\}), \ \sigma = \text{std}(\{R_{j,u}\})$$

Advantage computation:

$$A_{i,t} = \sum_{t'=t}^{T_i} \hat{R}_{i,t'}$$

**Discounted variant:** When $\gamma < 1$, advantages may optionally include discounting in the sum:

$$A_{i,t} = \sum_{t'=t}^{T_i} \gamma^{t'-t} \hat{R}_{i,t'}$$

This option is available for all three reward types and will be tested in ablations.

Policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{T_i}\sum_{t=1}^{T_i}\left(A_{i,t}+\beta\left(\frac{\pi_{\rm ref}}{\pi_{\rm old}}-1\right)\right)\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\right]$$

Full derivation in `hypothesis.md` Eqs. (1)–(5).

---

## 5. Experimental Plan

### Phase 1: Classic RL Environments (MANDATORY FIRST STEP)

Before VLA tasks, validate all three reward types on simple benchmarks. Each environment supports all three reward types (Binary, Dense, Clipped Dense) for direct comparison:

- **CartPole-v1** — Binary: survival 0/1; Dense: angle + position penalties; Clipped Dense
- **MountainCar-v0** — Binary: reach flag 0/1; Dense: distance to goal + velocity; Clipped Dense
- **Acrobot-v1** — Binary: reach height 0/1; Dense: height progress + angle; Clipped Dense
- **HalfCheetah-v4 / Ant-v4** — Binary: velocity threshold 0/1; Dense: velocity + smoothness; Clipped Dense

**Success criteria:**
- Dense reward shows ≥15% improvement in sample efficiency over binary
- Clipped dense demonstrates either: (a) faster convergence than dense, or (b) comparable final performance with lower gradient variance
- Results statistically significant (p < 0.05, ≥5 seeds)

### Phase 2: VLA Manipulation Tasks

Only after Phase 1 success:
- LIBERO suite (Spatial, Object, Goal, Long)
- RoboTwin tasks
- CALVIN benchmark tasks
- **SFT only** baseline (supervised fine-tuning without RL) will also be included for VLA tasks

---

## 6. Expected Outcomes

| Property | Binary (SimpleVLA-RL) | Dense (ReinboT) | Clipped Dense |
|----------|----------------------|-----------------|---------------|
| Sample efficiency | Low | High | Medium–High |
| Gradient variance | High | Lower | Lowest (potentially) |
| Credit assignment | Poor | Fine-grained | Fine-grained |
| Policy invariance | ✓ | ✗ (ReinboT sum ≠ PBRS) | ✗ (empirical) |
| Implementation complexity | Minimal | Moderate | Moderate |

---

## 7. Key Risks

1. Poorly tuned dense reward weights → slow learning or suboptimal policy
2. Clipping threshold too high → loss of useful negative feedback
3. Computational overhead from dense reward computation
4. Overfitting to shaping signal rather than true task objective

---

## 8. Conclusion

This project systematically compares **three** GRPO reward formulations:

| # | Type | Source |
|---|------|--------|
| 1 | Binary (Sparse) | SimpleVLA-RL (arXiv:2509.09674) |
| 2 | Dense | ReinboT (Zhang et al., 2025) |
| 3 | Clipped Dense | This project — clipping threshold determined experimentally |

Validation proceeds in two phases: classic RL environments first, then VLA manipulation tasks. Full mathematical formulation and algorithm pseudocode are documented in `hypothesis.md`. Agent instructions live in `agents.md`.
