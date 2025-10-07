"""Check what policies are being evaluated."""

from agents.ActiveInference import utils

# Construct policies for policy_len=3, 6 actions
policies = utils.construct_policies(
    actions=[0, 1, 2, 3, 4, 5],  # UP, DOWN, LEFT, RIGHT, OPEN, NOOP
    policy_len=3
)

print(f"Total policies: {len(policies)}")
print(f"Expected: 6^3 = {6**3}")

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']

# Find the DOWN-DOWN-OPEN policy
target = [1, 1, 4]
print(f"\nLooking for policy DOWN-DOWN-OPEN: {target}")

found = False
for idx, policy in enumerate(policies):
    if list(policy) == target:
        print(f"  Found at index {idx}!")
        found = True
        break

if not found:
    print(f"  NOT FOUND!")

# Show first 10 policies
print(f"\nFirst 20 policies:")
for idx in range(min(20, len(policies))):
    policy = policies[idx]
    policy_str = ' → '.join([action_names[a] for a in policy])
    print(f"  {idx:3d}: {policy_str}")


