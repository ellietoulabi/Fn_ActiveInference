"""
Sanity test: mappo_simple.py's AIFObsOvercookedMAEnv DOES teleport -- the
opposite regression test from test_no_teleport.py, which still covers
mappo.py's separate SemanticAIFObsOvercookedRLlibEnv (unchanged, still walks
primitive-by-primitive).

Deliberate design change, 2026-08-14 (ai/02-debug.md, "MAPPO teleportation",
explicit request): selecting a semantic (destination, mode) option now moves
the agent directly to its planned standing tile/facing in one step, with the
real engine's own transition function only invoked once (to resolve INTERACT,
if requested, from the teleported position) -- not walked one primitive at a
time. This is an intentional, one-sided advantage given to MAPPO to test
whether its earlier zero-delivery training result was really about the
combinatorial burden of primitive-level exploration; AIF's own paradigms are
untouched and still walk normally.

Checks:
  - A faraway destination CAN produce a position change of more than 1
    Manhattan cell in a single env.step() (proves teleport happened, not just
    "happened to reach an adjacent tile").
  - state.timestep advances by exactly 1 per env.step() regardless of how far
    either agent teleported (the whole point: walking is free, decision
    budget isn't spent on it).
  - Both agents' final positions are never equal to each other after a
    simultaneous teleport (regression test for the overlapping-players bug
    found and fixed the same day: independently-planned teleport targets can
    coincide since planning only ever sees the partner's PRE-decision
    position; _fast_forward_joint resolves both agents on one shared scratch
    state so the engine's own collision handling applies).
  - INTERACT-mode options actually pick up an onion from a teleported
    position (proves the interact resolves against the teleported state, not
    some stale one).

Run:
    .venv/bin/python -m agents.PPO.MA_PPO.test_teleport
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.IndependentActiveInferenceWithDynamicPolicies import utils as dyn_utils
from agents.PPO.MA_PPO.mappo_simple import AIFObsOvercookedMAEnv


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def test_far_teleport_and_fixed_timestep_cost():
    env = AIFObsOvercookedMAEnv({"layout": "cramped_room", "horizon": 400})
    obs, infos = env.reset(seed=0)
    state = infos["agent_0"]["state"]
    pre_positions = [p.position for p in state.players]
    pre_timestep = state.timestep

    n_modes = len(dyn_utils.MODES)
    opt0 = dyn_utils.DESTINATIONS.index("cntr4") * n_modes + dyn_utils.MODES.index("stay")
    opt1 = dyn_utils.DESTINATIONS.index("cntr1") * n_modes + dyn_utils.MODES.index("stay")

    obs, rew, term, trunc, infos = env.step({"agent_0": opt0, "agent_1": opt1})
    state = infos["agent_0"]["state"]
    post_positions = [p.position for p in state.players]

    jumps = [_manhattan(pre_positions[i], post_positions[i]) for i in range(2)]
    assert any(j > 1 for j in jumps), (
        f"Expected at least one agent to jump more than 1 cell (teleport), got jumps={jumps}"
    )
    assert state.timestep == pre_timestep + 1, (
        f"Expected exactly one real env tick consumed regardless of teleport "
        f"distance, got timestep {pre_timestep} -> {state.timestep}"
    )
    assert post_positions[0] != post_positions[1], (
        f"Agents ended up on the same tile: {post_positions}"
    )
    print(f"OK: far teleport (jumps={jumps}), timestep advanced by exactly 1, no overlap.")


def test_joint_teleport_no_overlap_when_targets_would_collide():
    """
    Regression test for the specific bug found and fixed the same day:
    independently-planned teleport targets landing on the same tile when both
    agents pick a destination near each other / near a shared approach tile.
    """
    env = AIFObsOvercookedMAEnv({"layout": "cramped_room", "horizon": 400})
    obs, infos = env.reset(seed=0)

    n_modes = len(dyn_utils.MODES)
    # Both agents target the SAME destination -- a real historical repro case.
    opt = dyn_utils.DESTINATIONS.index("onion1") * n_modes + dyn_utils.MODES.index("interact")

    # Should not raise (previously: AssertionError: Overlapping players or objects).
    obs, rew, term, trunc, infos = env.step({"agent_0": opt, "agent_1": opt})
    state = infos["agent_0"]["state"]
    positions = [p.position for p in state.players]
    assert positions[0] != positions[1], f"Agents overlapped: {positions}"
    print(f"OK: both agents targeting the same destination did not overlap ({positions}).")


def test_interact_resolves_from_teleported_position():
    env = AIFObsOvercookedMAEnv({"layout": "cramped_room", "horizon": 400})
    obs, infos = env.reset(seed=0)

    n_modes = len(dyn_utils.MODES)
    opt = dyn_utils.DESTINATIONS.index("onion1") * n_modes + dyn_utils.MODES.index("interact")
    obs, rew, term, trunc, infos = env.step({"agent_0": opt, "agent_1": dyn_utils.DESTINATIONS.index("cntr4") * n_modes + dyn_utils.MODES.index("stay")})
    state = infos["agent_0"]["state"]
    assert state.players[0].held_object is not None, (
        "Expected agent_0 to be holding an onion after teleporting to onion1 and interacting."
    )
    print(f"OK: interact resolved from teleported position, held={state.players[0].held_object}.")


def main():
    test_far_teleport_and_fixed_timestep_cost()
    test_joint_teleport_no_overlap_when_targets_would_collide()
    test_interact_resolves_from_teleported_position()
    print("")
    print("ALL OK: AIFObsOvercookedMAEnv teleports correctly, with no inter-agent overlap.")


if __name__ == "__main__":
    main()
