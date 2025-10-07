"""Analyze computational complexity of dependency-based marginalization."""

from generative_models.SA_ActiveInference.RedBlueButton import model_init

print("Dependency-based Marginalization Analysis")
print("=" * 70)
print()

total_full_joint = 1
for factor, vals in model_init.states.items():
    size = len(vals)
    total_full_joint *= size
    print(f"{factor:20s}: {size:4d} states")

print(f"\n{'FULL JOINT':20s}: {total_full_joint:4d} states (if we enumerate all)\n")

print("=" * 70)
print("Per-modality dependency complexity:")
print("=" * 70)

for modality, deps in model_init.observation_state_dependencies.items():
    num_combos = 1
    for dep in deps:
        dep_size = len(model_init.states[dep])
        num_combos *= dep_size
    
    deps_str = ", ".join(deps)
    print(f"{modality:25s}: {num_combos:6d} combos  (deps: {deps_str})")

print()
print("=" * 70)
print("Policy evaluation complexity:")
print("=" * 70)

num_actions = 6
policy_len = 3
num_policies = num_actions ** policy_len

print(f"Actions: {num_actions}")
print(f"Policy length: {policy_len}")
print(f"Number of policies: {num_policies}")
print()

# For each policy, we need to:
# - Roll out states (policy_len steps)
# - For each timestep, compute expected obs (7 modalities)
# - Compute utility and info gain

print("Per policy evaluation:")
print(f"  - State rollouts: {policy_len} steps")
print(f"  - Expected obs computations: {policy_len} timesteps")
print(f"  - Info gain computations: {policy_len} timesteps")
print()

# Sum up A_fn calls per timestep
total_a_calls_per_timestep = 0
for modality, deps in model_init.observation_state_dependencies.items():
    num_combos = 1
    for dep in deps:
        num_combos *= len(model_init.states[dep])
    total_a_calls_per_timestep += num_combos

print(f"A_fn calls per timestep (for all modalities): {total_a_calls_per_timestep}")
print(f"A_fn calls for expected obs over {policy_len} steps: {total_a_calls_per_timestep * policy_len}")
print(f"A_fn calls for info gain over {policy_len} steps: {total_a_calls_per_timestep * policy_len}")
print(f"Total A_fn calls per policy: {total_a_calls_per_timestep * policy_len * 2}")
print()
print(f"TOTAL A_fn calls for all {num_policies} policies: {total_a_calls_per_timestep * policy_len * 2 * num_policies:,}")


