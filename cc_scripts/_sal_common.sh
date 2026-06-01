# Shared helpers for ind/ic/fc semantic-action-level SLURM scripts.
# Source from the repo root after: cd .../Fn_ActiveInference

sal_setup_pythonpath() {
    export PYTHONPATH="$PWD:$PWD/run_scripts_overcooked:$PWD/environments/overcooked_ai/src"
    export PYTHONIOENCODING=utf-8
}

sal_preflight() {
    local paradigm="${1:?paradigm required: ind|ic|fc}"
    echo "Preflight (${paradigm}): checking imports..."
    python -c "
import sys
from pathlib import Path
root = Path('.').resolve()
sys.path.insert(0, str(root / 'run_scripts_overcooked'))
sys.path.insert(0, str(root / 'environments' / 'overcooked_ai' / 'src'))
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
    echo "Copy done"
}
