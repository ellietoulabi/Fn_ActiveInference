"""Calculate exact number of A_fn and B_fn calls per agent step."""

from generative_models.SA_ActiveInference.RedBlueButton import model_init

print("=" * 80)
print("CALL COUNT ANALYSIS - Per Agent Step")
print("=" * 80)

# Agent parameters
policy_len = 1  # For quick test
inference_horizon = 1
num_actions = 6  # UP, DOWN, LEFT, RIGHT, OPEN, NOOP
num_policies = num_actions ** policy_len
print(f"\nAgent Config:")
print(f"  policy_len: {policy_len}")
print(f"  inference_horizon: {inference_horizon}")
print(f"  num_policies: {num_policies}")

# State factors and their dependencies
SKIP_MODALITIES = {'button_just_pressed'}
all_deps = set()
for modality, deps in model_init.observation_state_dependencies.items():
    if modality not in SKIP_MODALITIES:
        all_deps.update(deps)

dep_list = sorted(all_deps)
print(f"\n  Unique state factors in dependencies (excl. button_just_pressed): {dep_list}")

# Calculate state space for these dependencies
num_state_combos = 1
for dep in dep_list:
    size = len(model_init.states[dep])
    num_state_combos *= size
    print(f"    {dep}: {size} states")

print(f"  → Total unique state combos to enumerate: {num_state_combos}")

print("\n" + "=" * 80)
print("STEP 1: STATE INFERENCE (infer_states)")
print("=" * 80)
print("Uses vanilla_fpi_update_posterior_states")
print("  → Does NOT call A_fn (uses stored observation)")
print("  → Does NOT call B_fn")
print()
print("A_fn calls: 0")
print("B_fn calls: 0")

print("\n" + "=" * 80)
print("STEP 2: POLICY INFERENCE (infer_policies)")
print("=" * 80)
print(f"Evaluates {num_policies} policies\n")

# For each policy:
print(f"For EACH of {num_policies} policies:")
print(f"  1. get_expected_states (roll out policy)")
print(f"       → Calls B_fn {policy_len} times (once per action in policy)")
print(f"       → B_fn calls per policy: {policy_len}")
print()
print(f"  2. get_expected_obs_sequence (predict observations)")
print(f"       → Calls get_expected_obs_from_beliefs {policy_len} times (once per timestep)")
print(f"       → Each get_expected_obs_from_beliefs:")
print(f"           - Enumerates {num_state_combos} state combinations ONCE")
print(f"           - Calls A_fn {num_state_combos} times")
print(f"           - Reuses these {num_state_combos} results across all 6 modalities")
print(f"       → A_fn calls per policy (for expected obs): {num_state_combos * policy_len}")
print()
print(f"  3. calc_expected_utility (compute pragmatic value)")
print(f"       → Uses cached observations, no A_fn or B_fn calls")
print()
print(f"  4. calc_states_info_gain (compute epistemic value)")
print(f"       → Calls calc_surprise_functional {policy_len} times (once per timestep)")
print(f"       → Each calc_surprise_functional:")
print(f"           - Enumerates {num_state_combos} state combinations ONCE")
print(f"           - Calls A_fn {num_state_combos} times")
print(f"           - Reuses these {num_state_combos} results across all 6 modalities")
print(f"       → A_fn calls per policy (for info gain): {num_state_combos * policy_len}")
print()

b_calls_per_policy = policy_len
a_calls_per_policy = num_state_combos * policy_len * 2  # *2 for expected_obs + info_gain

print(f"TOTAL per policy:")
print(f"  B_fn calls: {b_calls_per_policy}")
print(f"  A_fn calls: {a_calls_per_policy}")
print()

total_b_calls = b_calls_per_policy * num_policies
total_a_calls = a_calls_per_policy * num_policies

print(f"TOTAL across all {num_policies} policies:")
print(f"  B_fn calls: {total_b_calls}")
print(f"  A_fn calls: {total_a_calls}")

print("\n" + "=" * 80)
print("TOTAL PER AGENT STEP")
print("=" * 80)
print(f"B_fn calls: {total_b_calls}")
print(f"A_fn calls: {total_a_calls}")

# Estimate time
a_fn_time_ms = 0.115  # from profile
b_fn_time_ms = 0.01   # estimated (very fast)

estimated_time_ms = total_a_calls * a_fn_time_ms + total_b_calls * b_fn_time_ms
print(f"\nEstimated time (at {a_fn_time_ms}ms per A_fn, {b_fn_time_ms}ms per B_fn):")
print(f"  A_fn time: {total_a_calls * a_fn_time_ms:.1f}ms")
print(f"  B_fn time: {total_b_calls * b_fn_time_ms:.1f}ms")
print(f"  TOTAL: ~{estimated_time_ms:.1f}ms")

print("\n" + "=" * 80)
print("WHY SO MANY A_fn CALLS?")
print("=" * 80)
print(f"1. We have {num_state_combos} unique state combinations to enumerate")
print(f"   (all combos of: {dep_list})")
print(f"2. We call A_fn TWICE per policy:")
print(f"   - Once in get_expected_obs_from_beliefs (to predict observations)")
print(f"   - Once in calc_surprise_functional (to compute info gain)")
print(f"3. We evaluate {num_policies} policies")
print(f"4. Result: {num_state_combos} × 2 × {num_policies} = {total_a_calls} A_fn calls")
print()
print("WITH OPTIMIZATION:")
print(f"  - We enumerate {num_state_combos} states ONCE per call to get_expected_obs_from_beliefs")
print(f"  - We REUSE those {num_state_combos} A_fn results across all 6 modalities")
print(f"  - WITHOUT this: would be {num_state_combos * 6} A_fn calls per modality computation!")


