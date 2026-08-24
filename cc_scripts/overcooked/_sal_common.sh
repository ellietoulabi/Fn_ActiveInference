# Shared helpers for ind/ic/fc semantic-action-level SLURM scripts.
# Source from the repo root after: cd .../Fn_ActiveInference

sal_setup_pythonpath() {
    # Prepend (not overwrite) -- a hard overwrite here previously destroyed
    # whatever `module load gcc arrow` put on PYTHONPATH to make pyarrow
    # importable (Alliance's pip pyarrow is a dummy wheel that always fails;
    # the real package only comes from that module). Confirmed via a real CC
    # job: `pyarrow OK` succeeded right after the module load, then
    # `ModuleNotFoundError: No module named 'pyarrow'` appeared the moment
    # this function first ran (inside sal_verify_imports) and stayed broken
    # for the rest of the job, since every later call just re-applied the
    # same arrow-free value. Repo-local paths still come first, so nothing
    # about import priority for this project's own modules changes.
    export PYTHONPATH="$PWD:$PWD/run_scripts_overcooked:$PWD/environments/overcooked_ai/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONIOENCODING=utf-8
}

# Alliance: scipy-stack packages are not always visible once PYTHONPATH is set on a venv.
# Install wheels into the venv so imports match runtime (overcooked needs scipy.sparse).
sal_ensure_venv_runtime_deps() {
    echo "Ensuring numpy+scipy in venv (Compute Canada scipy-stack workaround)..."
    pip install --no-input --ignore-installed \
        'numpy>=1.20.0' \
        'scipy>=1.7.0' || {
        echo "ERROR: pip install numpy/scipy into venv failed."
        return 1
    }
}

# Back-compat alias for scripts that have not been updated yet.
sal_ensure_venv_numpy() {
    sal_ensure_venv_runtime_deps
}

sal_verify_imports() {
    sal_setup_pythonpath
    python -c "
import numpy
import scipy.sparse
import gymnasium
import dill
print('numpy:', numpy.__file__)
print('scipy:', scipy.__file__)
print('import check OK (with PYTHONPATH)')
" || {
        echo "ERROR: import check failed with PYTHONPATH set (numpy/scipy/gymnasium/dill)."
        return 1
    }
}

sal_preflight() {
    # ic/fc now check the CollisionFix stacks specifically, since those are the
    # only variants carrying the collision-blindness fix (ai/02-debug.md J.8)
    # and PROGRESS_SUCCESS_PROB fix (J.9) -- the plain/base stacks are known to
    # produce zero-delivery runs without them. FC's check points at the
    # rebuilt-from-scratch FullyCollective* packages (section K); the old,
    # discarded IND-derived FC package this used to check no longer exists.
    local paradigm="${1:?paradigm required: ind|ic|fc}"
    echo "Preflight (${paradigm}): checking imports..."
    sal_setup_pythonpath
    python -c "
import numpy as np  # noqa: F401 — before any repo imports
import sal_step_csv_log  # noqa: F401
import run_independent_semantic_action_level as ind  # noqa: F401
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv  # noqa: F401
paradigm = '${paradigm}'
if paradigm == 'ic':
    import run_individually_collective_policy_semantic_action_level_optimized_collision_fix as ric  # noqa: F401
    from agents.ActiveInferenceFixedPoliciesOptimizedCollisionFix.agent import Agent  # noqa: F401
    from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPoliciesActionLevelCollisionFix import model_init  # noqa: F401
    assert hasattr(model_init, 'PROGRESS_SUCCESS_PROB'), 'IC CollisionFix model_init missing PROGRESS_SUCCESS_PROB'
if paradigm == 'fc':
    import run_fully_collective_policy_semantic_action_level_optimized_collision_fix as rfc  # noqa: F401
    from agents.FullyCollectiveFixedPoliciesOptimizedCollisionFix.agent import Agent  # noqa: F401
    from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.FullyCollectiveWithSemanticPoliciesActionLevelCollisionFix import model_init  # noqa: F401
    assert hasattr(model_init, 'PROGRESS_SUCCESS_PROB'), 'FC CollisionFix model_init missing PROGRESS_SUCCESS_PROB'
print('Preflight OK:', paradigm)
" || {
        echo "ERROR: preflight imports failed for ${paradigm} (see traceback above)."
        return 1
    }
}

sal_report_failure() {
    local log_file="${1:-}"
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "========== last 100 lines of ${log_file} =========="
        tail -n 100 "$log_file"
        echo "========== end log tail =========="
    else
        echo "ERROR: log file not found${log_file:+: $log_file}"
    fi
}

sal_copy_artifacts() {
    local dest_base="${1:?dest}"
    local log_file="${2:?log}"
    local csv_dir="${3:?csv_dir}"
    echo "Copying logs and step CSVs to ${dest_base}..."
    mkdir -p "$dest_base"
    cp "$log_file" "$dest_base/" 2>/dev/null || echo "Warning: log file not found: $log_file"
    if compgen -G "${csv_dir}"/*.csv > /dev/null 2>&1; then
        cp "${csv_dir}"/*.csv "$dest_base/" 2>/dev/null || echo "Warning: CSV copy failed"
    else
        echo "Warning: no step CSV files in ${csv_dir}"
    fi
    if compgen -G "${csv_dir}"/*.jsonl > /dev/null 2>&1; then
        cp "${csv_dir}"/*.jsonl "$dest_base/" 2>/dev/null || echo "Warning: JSONL copy failed"
    fi
    echo "Copy done"
}
