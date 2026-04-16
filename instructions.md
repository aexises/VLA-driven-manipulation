# Phase 1 Instructions

This file explains how to run **Phase 1 only** and how to test the hypotheses from [hypothesis.md](/Users/daeron/VLA-driven-manipulation/hypothesis.md).

Important runtime note:
- In this repo, use `python`, not `python3`, if your installed ML stack lives in the active Conda environment.

## 1. Goal of Phase 1

Phase 1 validates the three reward variants on classic RL tasks before any VLA work:
- `binary`
- `dense`
- `clipped_dense`

Phase 1 is split into:
- **Phase 1A**: `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`
- **Phase 1B**: `HalfCheetah-v4`, `Ant-v4`

## 2. Environment Setup

Install the repo in your active environment:

```bash
python -m pip install -e .
```

If you want MuJoCo support for Phase 1B:

```bash
python -m pip install -e ".[mujoco]"
```

If you also want the test dependency:

```bash
python -m pip install -e ".[dev]"
```

## 3. Verify Setup

Check whether the Phase 1 environments are available:

```bash
python -m src.experiments.check_phase1_setup
```

Expected behavior:
- Phase 1A environments should report `ok`
- Phase 1B may fail until MuJoCo support is installed correctly

## 4. Smoke Run

Generate a small Phase 1A smoke matrix:

```bash
python -m src.experiments.phase1_classic \
  --config src/experiments/configs/phase1a_smoke.yaml \
  --mode generate-configs \
  --output-dir results/generated-configs/phase1a_smoke
```

Run one specific smoke config:

```bash
python -m src.experiments.run_experiment \
  --config results/generated-configs/phase1a_smoke/phase1a_cartpole_v1_dense_seed11.yaml
```

This should create:
- TensorBoard-style logs under `runs/...`
- flat results under `results/...`
- `config.yaml`
- `metrics.csv`
- `summary.json`
- `training_curves.png`

## 5. Full Phase 1 Workflow

### Phase 1A

Generate all Phase 1A single-run configs:

```bash
python -m src.experiments.phase1_classic \
  --config src/experiments/configs/phase1a.yaml \
  --mode generate-configs \
  --output-dir results/generated-configs/phase1a
```

Run the whole Phase 1A matrix:

```bash
python -m src.experiments.phase1_classic \
  --config src/experiments/configs/phase1a.yaml \
  --mode run-matrix
```

Build the aggregate Phase 1A/Phase 1 report:

```bash
python -m src.experiments.phase1_classic \
  --mode analyze-results \
  --results-root results \
  --output-dir results/phase1_report
```

### Phase 1B

After Phase 1A and `tau_clip` selection, run Phase 1B:

```bash
python -m src.experiments.phase1_classic \
  --config src/experiments/configs/phase1b.yaml \
  --mode run-matrix
```

Then regenerate the aggregate report:

```bash
python -m src.experiments.phase1_classic \
  --mode analyze-results \
  --results-root results \
  --output-dir results/phase1_report
```

## 6. How to Test Each Hypothesis

### Hypothesis 1

Claim:
- Dense reward improves sample efficiency over sparse reward
- Clipped dense may stabilize early learning

Use:
- `results/*/*/metrics.csv`
- `results/*/*/summary.json`
- `results/phase1_report/phase1_report.json`
- `results/phase1_report/plots/*.png`

Primary metrics:
- `metrics_success_rate`
- `reward_episode_cumulative`
- `sample_efficiency_iteration`

How to evaluate:
1. Run the full matrix with identical seeds across reward types.
2. Compare `dense` vs `binary` per environment.
3. In `phase1_report.json`, inspect:
   - `sample_efficiency_iteration_mean`
   - `final_success_rate_mean`
   - pairwise Mann-Whitney statistics
4. Accept Hypothesis 1 only if:
   - dense beats binary on sample efficiency by the pre-registered criterion
   - the corrected p-value is below `0.05`

### Hypothesis 2

Claim:
- Dense reward component weights affect behavior and learning speed

Current support in repo:
- reward weights are configurable in `RewardConfig`
- per-run configs are written as YAML-like files and can be edited for ablations

Recommended ablation process:
1. Copy a generated dense config.
2. Modify one weight at a time:
   - `main_progress`
   - `auxiliary_progress`
   - `smoothness`
   - `terminal_success`
3. Re-run the modified config with the same seed set.
4. Compare:
   - `final_success_rate`
   - `mean_cumulative_reward`
   - `mean_training_negative_ratio`
   - `trajectory smoothness` proxy through the dense reward design and training curves

### Hypothesis 3

Claim:
- Group normalization lowers gradient variance

Current support in repo:
- `GRPOConfig.use_group_normalization`
- `GRPOConfig.discounted_advantage`

How to evaluate:
1. Duplicate dense and clipped-dense configs.
2. Set `use_group_normalization: false`.
3. Re-run with the same seeds.
4. Compare:
   - `mean_gradient_variance_proxy`
   - `grad_advantage_std` in `metrics.csv`
   - stability of success-rate curves in `training_curves.png`

## 7. How to Select `tau_clip`

The clipped-dense threshold must be chosen on Phase 1A first.

Candidate values:
- `0.0`
- `0.1`
- `0.2`
- `0.5`

Recommended procedure:
1. Generate clipped-dense configs for each `tau_clip` value.
2. Run the full Phase 1A seed set for each candidate.
3. Rank candidates by:
   - sample efficiency first
   - gradient-variance reduction second
4. Freeze the winning `tau_clip` before Phase 1B.

## 8. Plots

### Is there code for plots?

Yes now.

Per-run plot generation is implemented in:
- [src/analysis/plot_results.py](/Users/daeron/VLA-driven-manipulation/src/analysis/plot_results.py)

Automatic run-level plot saving is triggered from:
- [src/analysis/logging.py](/Users/daeron/VLA-driven-manipulation/src/analysis/logging.py)

Aggregate Phase 1 comparison plots are generated by:
- [src/analysis/phase1_report.py](/Users/daeron/VLA-driven-manipulation/src/analysis/phase1_report.py)

Saved outputs:
- per-run: `results/<experiment>/<run_id>/training_curves.png`
- aggregate: `results/phase1_report/plots/*_comparison.png`

## 9. What “Ready to Run” Means in This Repo

For Phase 1, the repo now supports:
- single-run training from config
- matrix generation for Phase 1A and Phase 1B
- per-run metrics and summaries
- automatic per-run plots
- aggregate Phase 1 report generation
- pairwise Mann-Whitney tests with Holm correction

What still depends on your local environment:
- MuJoCo installation for `HalfCheetah-v4` and `Ant-v4`
- optional `pytest` if you want to run tests
- optional TensorBoard package if you want native TensorBoard writers instead of the no-op fallback

## 10. Recommended Minimal Command Sequence

```bash
python -m pip install -e .
python -m src.experiments.check_phase1_setup
python -m src.experiments.run_experiment --config src/experiments/configs/cartpole_dense.yaml
python -m src.experiments.phase1_classic --config src/experiments/configs/phase1a.yaml --mode run-matrix
python -m src.experiments.phase1_classic --mode analyze-results --results-root results --output-dir results/phase1_report
```
