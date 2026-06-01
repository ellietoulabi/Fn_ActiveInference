# Shared helpers for ind/ic/fc semantic-action-level SLURM scripts.
# Source from the repo root after: cd .../Fn_ActiveInference

sal_setup_pythonpath() {
    export PYTHONPATH="$PWD:$PWD/run_scripts_overcooked:$PWD/environments/overcooked_ai/src"
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
    local paradigm="${1:?paradigm required: ind|ic|fc}"
    echo "Preflight (${paradigm}): checking imports..."
    sal_setup_pythonpath
    python -c "
import numpy as np  # noqa: F401 — before any repo imports
import sal_step_csv_log  # noqa: F401
import run_independent_semantic_action_level as ind  # noqa: F401
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv  # noqa: F401
paradigm = '${paradigm}'
if paradigm in ('ind', 'ic'):
    import run_individually_collective_policy_semantic_action_level as ric  # noqa: F401
if paradigm == 'ic':
    from agents.ActiveInferenceFixedPolicies.agent import Agent  # noqa: F401
if paradigm == 'fc':
    from agents.IndependentActiveInferenceWithDynamicPolicies.agent import Agent  # noqa: F401
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
