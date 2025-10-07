"""Analyze optimized complexity after skipping button_just_pressed."""

from generative_models.SA_ActiveInference.RedBlueButton import model_init

print("Optimized Marginalization (skipping button_just_pressed)")
print("=" * 70)

SKIP_MODALITIES = {'button_just_pressed'}

total_a_calls_per_timestep = 0
for modality, deps in model_init.observation_state_dependencies.items():
    if modality in SKIP_MODALITIES:
        print(f"{modality:25s}: SKIPPED (would be {9*9*9*2*2:6d} combos)")
        continue
        
    num_combos = 1
    for dep in deps:
        num_combos *= len(model_init.states[dep])
    total_a_calls_per_timestep += num_combos
    deps_str = ", ".join(deps)
    print(f"{modality:25s}: {num_combos:6d} combos  (deps: {deps_str})")

print()
print(f"A_fn calls per timestep: {total_a_calls_per_timestep}")
print()

num_policies = 6  # policy_len=1
policy_len = 1

print(f"For {num_policies} policies, policy_len={policy_len}:")
print(f"  Expected obs: {total_a_calls_per_timestep * policy_len} calls/policy")
print(f"  Info gain: {total_a_calls_per_timestep * policy_len} calls/policy")
print(f"  Total per policy: {total_a_calls_per_timestep * policy_len * 2}")
print(f"  TOTAL: {total_a_calls_per_timestep * policy_len * 2 * num_policies:,} A_fn calls")


