#!/bin/bash
# Submit every MA Red-Blue-Button method in one go: all three Active
# Inference paradigms, both OPSRL variants (cold-start + pretrained-to-
# convergence), and both MAPPO variants (pretrained + online/cold-start).
#
# This is a plain submit-all wrapper, not an SBATCH job itself -- run it
# directly on a CC login node from the repo root:
#   bash cc_scripts/redbluebutton/submit_all_ma.sh
#
# All seven jobs use the same 30-seed pool (0..29), the same 100-episode/
# 20-per-config/max_steps=50 protocol, and -- confirmed directly against
# each script's own generate_random_config -- the identical button-position
# map sequence per seed, so results are genuinely comparable seed-for-seed.
# Each writes to its own distinct destination folder (verified no collisions
# across the whole cc_scripts/redbluebutton/ launcher set), so all seven can
# run concurrently with no risk of overwriting each other's logs.
#
# Env-var overrides (MAX_STEPS, PRETRAIN_MAX_EPISODES, BUDGETS, etc.) are
# NOT threaded through this wrapper -- if you need one, submit that single
# launcher directly instead, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_aif_independent.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "================================================================"
echo "Submitting all MA Red-Blue-Button jobs"
echo "================================================================"

declare -a JOB_IDS=()
declare -a JOB_NAMES=()

submit() {
    local name="$1"
    shift
    echo ""
    echo "--- $name ---"
    local out
    if out=$("$@" 2>&1); then
        echo "$out"
        JOB_IDS+=("$(echo "$out" | grep -oE '[0-9]+' | tail -1)")
        JOB_NAMES+=("$name")
    else
        echo "FAILED to submit $name:"
        echo "$out"
    fi
}

submit "Independent AIF"            sbatch two_aif_independent.sh
submit "Fully Collective AIF"       sbatch two_aif_fully_collective.sh
submit "Individually Collective AIF" sbatch two_aif_individually_collective.sh
submit "OPSRL (cold-start)"         sbatch two_opsrl.sh
submit "OPSRL (pretrained)"         sbatch two_opsrl_pretrained.sh
submit "MAPPO (pretrained)"         sbatch mappo_checkpoint_curve.sh
# `env MODE=online sbatch ...` (not `MODE=online submit ...`) -- setting the
# var directly on the exec'd command via env is unambiguous; prefixing it to
# the submit() function call instead would rely on bash's function-scoped
# export-propagation semantics, which is fragile and easy to get wrong here.
submit "MAPPO (online/cold-start)"  env MODE=online sbatch mappo_checkpoint_curve.sh

echo ""
echo "================================================================"
echo "Submitted ${#JOB_IDS[@]}/7 jobs:"
for i in "${!JOB_IDS[@]}"; do
    printf "  %-32s job %s\n" "${JOB_NAMES[$i]}" "${JOB_IDS[$i]}"
done
echo "================================================================"
echo "Check status with: squeue -u \$USER"
