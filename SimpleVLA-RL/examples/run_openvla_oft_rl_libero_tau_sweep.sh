set -x

BASE_EXPERIMENT_PREFIX='MODIFIED YOURSELF e.g. libero10_tau_sweep'
for TAU_CLIP in 0.0 0.1 0.2 0.5; do
  EXPERIMENT_NAME="${BASE_EXPERIMENT_PREFIX}_tau${TAU_CLIP}"
  WANDB_API_KEY="${WANDB_API_KEY:-YOUR WANDB KEY}" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  TAU_CLIP="$TAU_CLIP" \
  bash examples/run_openvla_oft_rl_libero_clipped_dense.sh
done
