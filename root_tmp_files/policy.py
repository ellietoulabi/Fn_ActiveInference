"""
Ego-First Interleaved Policy Generator

Policy horizon = 3

Step 1:
    actor = SELF only
    action ∈ {NORTH, SOUTH, EAST, WEST, STAY, INTERACT}

Step 2–3:
    actor ∈ {SELF, OTHER}
    action ∈ {NORTH, SOUTH, EAST, WEST, STAY, INTERACT}

Total policies:
    6 × 12 × 12 = 864

Policy representation:
    policy = [
        (actor, action),
        (actor, action),
        (actor, action)
    ]
"""

import csv
from itertools import product
from typing import List, Tuple

# -------------------------------------------------
# Actor encoding
# -------------------------------------------------

SELF = 0
OTHER = 1

ACTOR_NAMES = {
    SELF: "SELF",
    OTHER: "OTHER"
}

# -------------------------------------------------
# Primitive actions
# -------------------------------------------------

NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
STAY = 4
INTERACT = 5

ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY, INTERACT]

ACTION_NAMES = {
    NORTH: "NORTH",
    SOUTH: "SOUTH",
    EAST: "EAST",
    WEST: "WEST",
    STAY: "STAY",
    INTERACT: "INTERACT"
}

# -------------------------------------------------
# Policy generator
# -------------------------------------------------

def generate_ego_first_interleaved_policies() -> List[List[Tuple[int, int]]]:
    """
    Generate all ego-first interleaved policies.

    Returns
    -------
    policies : list
        List of policies where each policy is
        [
            (actor, action),
            (actor, action),
            (actor, action)
        ]
    """

    policies = []

    # step1: SELF only
    step1_options = [(SELF, a) for a in ACTIONS]

    # step2 and step3: SELF or OTHER
    step_later_options = (
        [(SELF, a) for a in ACTIONS] +
        [(OTHER, a) for a in ACTIONS]
    )

    for step1 in step1_options:
        for step2, step3 in product(step_later_options, repeat=2):

            policy = [
                step1,
                step2,
                step3
            ]

            policies.append(policy)

    return policies


# -------------------------------------------------
# Save policies to CSV
# -------------------------------------------------

def save_policies_csv(policies, filepath):
    """
    Save policies to CSV file.

    CSV columns:
        policy_id
        s1_actor
        s1_action
        s2_actor
        s2_action
        s3_actor
        s3_action
    """

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "policy_id",
            "s1_actor", "s1_action",
            "s2_actor", "s2_action",
            "s3_actor", "s3_action"
        ])

        for pid, policy in enumerate(policies):

            row = [
                pid,
                policy[0][0], policy[0][1],
                policy[1][0], policy[1][1],
                policy[2][0], policy[2][1],
            ]

            writer.writerow(row)


# -------------------------------------------------
# Load policies from CSV
# -------------------------------------------------

def load_policies_csv(filepath) -> List[List[Tuple[int, int]]]:
    """
    Load policies from CSV file.

    Returns
    -------
    policies : list
        List of policies in format:
        [
            (actor, action),
            (actor, action),
            (actor, action)
        ]
    """

    policies = []

    with open(filepath, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            policy = [

                (int(row["s1_actor"]), int(row["s1_action"])),
                (int(row["s2_actor"]), int(row["s2_action"])),
                (int(row["s3_actor"]), int(row["s3_action"]))

            ]

            policies.append(policy)

    return policies


# -------------------------------------------------
# Policy validation
# -------------------------------------------------

def validate_policies(policies):
    """
    Basic sanity checks.
    """

    assert len(policies) == 864, "Policy count should be 864"

    for p in policies:

        assert len(p) == 3

        # step1 must be SELF
        assert p[0][0] == SELF

        for actor, action in p:

            assert actor in [SELF, OTHER]
            assert action in ACTIONS


# -------------------------------------------------
# Pretty print
# -------------------------------------------------

def policy_to_string(policy):

    parts = []

    for actor, action in policy:

        parts.append(
            f"{ACTOR_NAMES[actor]}:{ACTION_NAMES[action]}"
        )

    return " -> ".join(parts)


# -------------------------------------------------
# Example usage
# -------------------------------------------------

if __name__ == "__main__":

    policies = generate_ego_first_interleaved_policies()

    print("Generated:", len(policies))

    validate_policies(policies)

    save_policies_csv(policies, "ego_first_interleaved_policies.csv")

    loaded = load_policies_csv("ego_first_interleaved_policies.csv")

    print("Loaded:", len(loaded))

    print("Example policy:")
    print(policy_to_string(loaded[0]))