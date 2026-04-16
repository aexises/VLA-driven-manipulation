# Instructions: Running Reward Comparison for OpenVLA-OFT

This guide explains how to use the current implementation as intended.

The reward-comparison runtime lives in [SimpleVLA-RL](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL). The main supported comparison path is:

- `LIBERO`: `binary`, `dense`, `clipped_dense`
- `Robotwin`: `binary` only

## 1. What Is Implemented

The code supports three reward modes for `OpenVLA-OFT` RL training:

- `binary`
  Terminal-only reward, matching the original SimpleVLA-RL baseline behavior.
- `dense`
  LIBERO-only dense reward using:
  - progress: native `env.step(...)` reward
  - smoothness: `-mean((a_t - a_{t-1})^2)`
  - terminal: `1.0` on successful completion
  - subgoal: `0.0` in the current version
- `clipped_dense`
  Same as `dense`, then clipped with `max(r_t, tau_clip)`.

Current limitation:

- Robotwin rejects `dense` and `clipped_dense` by design in this version.

## 2. Prerequisites

You need a working `SimpleVLA-RL` environment with:

- `veRL`
- `OpenVLA-OFT`
- `LIBERO`
- optional `Robotwin` if you want binary-only Robotwin experiments
- a valid `align.json`
- a valid SFT checkpoint for `OpenVLA-OFT`

Use [SimpleVLA-RL/SETUP.md](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/SETUP.md) for environment setup.

You also need:

- `WANDB_API_KEY`
- a writable checkpoint root directory

## 3. Main Launch Scripts

Use these scripts from inside [SimpleVLA-RL](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL):

- [run_openvla_oft_rl_libero_binary.sh](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/examples/run_openvla_oft_rl_libero_binary.sh)
- [run_openvla_oft_rl_libero_dense.sh](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/examples/run_openvla_oft_rl_libero_dense.sh)
- [run_openvla_oft_rl_libero_clipped_dense.sh](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/examples/run_openvla_oft_rl_libero_clipped_dense.sh)
- [run_openvla_oft_rl_libero_tau_sweep.sh](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/examples/run_openvla_oft_rl_libero_tau_sweep.sh)

Recommended workflow:

1. Run `binary`
2. Run `dense`
3. Run `clipped_dense`
4. Run the `tau` sweep for clipped dense

## 4. Required Environment Variables

Each LIBERO launcher accepts environment overrides. At minimum set:

```bash
export WANDB_API_KEY=...
export PROJECT_NAME=SimpleVLA-RL
export EXPERIMENT_NAME=libero10_dense_seed0
export SFT_MODEL_PATH=/abs/path/to/openvla-oft-sft
export CKPT_PATH=/abs/path/to/output_root
export DATASET_NAME=libero_10
export ALIGN_PATH=/abs/path/to/SimpleVLA-RL/align.json
export NUM_GPUS=8
export NUM_NODES=1
export SEED=0
```

For clipped dense also set:

```bash
export TAU_CLIP=0.1
```

Supported LIBERO `DATASET_NAME` values in the current scripts:

- `libero_10`
- `libero_90`
- `libero_spatial`
- `libero_object`
- `libero_goal`

## 5. Example Commands

Run binary:

```bash
cd /Users/daeron/VLA-driven-manipulation/SimpleVLA-RL
bash examples/run_openvla_oft_rl_libero_binary.sh
```

Run dense:

```bash
cd /Users/daeron/VLA-driven-manipulation/SimpleVLA-RL
bash examples/run_openvla_oft_rl_libero_dense.sh
```

Run clipped dense:

```bash
cd /Users/daeron/VLA-driven-manipulation/SimpleVLA-RL
TAU_CLIP=0.1 bash examples/run_openvla_oft_rl_libero_clipped_dense.sh
```

Run `tau` sweep:

```bash
cd /Users/daeron/VLA-driven-manipulation/SimpleVLA-RL
BASE_EXPERIMENT_PREFIX=libero10_tau_sweep bash examples/run_openvla_oft_rl_libero_tau_sweep.sh
```

## 6. Reward Config Surface

The active Hydra reward config lives in [ppo_trainer.yaml](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/verl/trainer/config/ppo_trainer.yaml).

Relevant runtime keys:

- `reward.type=binary|dense|clipped_dense`
- `reward.impl=baseline_terminal|libero_native_dense`
- `reward.tau_clip=<float>`
- `reward.weights.subgoal=<float>`
- `reward.weights.progress=<float>`
- `reward.weights.smoothness=<float>`
- `reward.weights.terminal=<float>`
- `trainer.resume_mode=none|last|best`
- `actor_rollout_ref.model.resume=True|False`

Current intended defaults:

- binary:
  - `reward.type=binary`
  - `reward.impl=baseline_terminal`
- dense:
  - `reward.type=dense`
  - `reward.impl=libero_native_dense`
- clipped dense:
  - `reward.type=clipped_dense`
  - `reward.impl=libero_native_dense`
  - `reward.tau_clip in {0.0, 0.1, 0.2, 0.5}`

Default dense weights:

- `subgoal=0.0`
- `progress=1.0`
- `smoothness=0.05`
- `terminal=1.0`

## 7. Outputs Per Run

Each run writes under:

- `trainer.default_local_dir`

Expected files:

- `resolved_config.yaml`
- `metrics.jsonl`
- `summary.json`
- `actor/global_step_{step}/`
- `actor/last/`
- `actor/best/`
- `metadata/checkpoints.json`
- `metadata/best_checkpoint.json`

What they mean:

- `metrics.jsonl`
  Step-by-step metrics logged during training and validation.
- `summary.json`
  Final run summary including reward type, `tau_clip`, reward stats, and best/last checkpoint paths.
- `actor/last/`
  Most recent periodic checkpoint.
- `actor/best/`
  Best checkpoint using validation score when available.

## 8. Resume Behavior

Resume is controlled by:

- `actor_rollout_ref.model.resume=True`
- `trainer.resume_mode=last|best`

Example:

```bash
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  actor_rollout_ref.model.resume=True \
  trainer.resume_mode=best \
  ...
```

Resume rules:

- `last` loads from `actor/last/`
- `best` loads from `actor/best/`
- the run fails fast if the checkpoint does not exist
- the run also fails fast if checkpoint metadata does not match:
  - reward type
  - task suite
  - action token settings
  - VLA model family

## 9. Analysis

Use the local report script from the repo root:

[simplevla_reward_report.py](/Users/daeron/VLA-driven-manipulation/src/analysis/simplevla_reward_report.py)

Example:

```bash
cd /Users/daeron/VLA-driven-manipulation
python src/analysis/simplevla_reward_report.py \
  --results-root . \
  --output-dir results/simplevla_reward_report
```

Outputs include:

- `reward_report.json`
- `reward_summary_table.csv`
- `reward_comparison_table.csv`
- `tau_sweep_table.csv` when clipped runs exist
- comparison plots per LIBERO suite

## 10. Intended Experiment Order

Use this order for a clean comparison:

1. Pick one LIBERO suite and one SFT checkpoint.
2. Run `binary` with seed `0`.
3. Run `dense` with seed `0`.
4. Run `clipped_dense` with seed `0` for a few `tau_clip` values.
5. Select the best `tau_clip`.
6. Re-run all three reward types with matched seeds.
7. Aggregate results with the report script.

Keep these fixed across conditions:

- `SFT_MODEL_PATH`
- `DATASET_NAME`
- `NUM_GPUS`
- `NUM_NODES`
- `n_samples`
- actor/ref hyperparameters
- total epochs
- validation cadence
- seed list

## 11. Known Constraints

- LIBERO is the only benchmark with full reward-comparison support right now.
- Robotwin is binary-only right now.
- Dense reward currently uses LIBERO native step reward as the progress term.
- Best checkpoint selection uses validation score if present; otherwise it falls back to training verification score.
- The analysis script is meant for local result folders produced by this implementation, not arbitrary legacy SimpleVLA-RL runs.

## 12. Suggested Minimal Smoke Run

Before long runs, do a short smoke test by temporarily lowering the values inside the launcher you plan to use.

Recommended temporary edits for a smoke run:

- `trainer.total_epochs=1`
- `trainer.save_freq=1`
- `trainer.test_freq=1`
- `data.num_trials_per_task=2`
- optionally reduce `data.train_batch_size` and `data.val_batch_size`

Example file to edit first:

- [run_openvla_oft_rl_libero_dense.sh](/Users/daeron/VLA-driven-manipulation/SimpleVLA-RL/examples/run_openvla_oft_rl_libero_dense.sh)

Then run:

```bash
cd /Users/daeron/VLA-driven-manipulation/SimpleVLA-RL
EXPERIMENT_NAME=libero10_dense_smoke SEED=0 \
bash examples/run_openvla_oft_rl_libero_dense.sh
```

If you want, I can also add a second section to this guide with copy-paste commands for:

- 3-way LIBERO comparison
- `tau_clip` sweep
- resume from `best`
- exportable result table generation
