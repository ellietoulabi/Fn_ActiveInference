"""
Sanity test: selecting a faraway semantic target produces ONE primitive step,
never a teleport.

We pick a semantic option whose destination is far from the agent and step
the env once. Then we assert:
  - The agent did not jump to the target tile in one step.
  - The (x, y) position changed by at most 1 in Manhattan distance.

Runs the check against BOTH semantic envs:
  - agents.PPO.MA_PPO.mappo.SemanticAIFObsOvercookedRLlibEnv
  - agents.PPO.MA_PPO.mappo_simple.AIFObsOvercookedMAEnv

Run:
    .venv/bin/python -m agents.PPO.MA_PPO.test_no_teleport
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.IndependentActiveInferenceWithDynamicPolicies import utils as dyn_utils
from agents.PPO.MA_PPO.mappo import SemanticAIFObsOvercookedRLlibEnv
from agents.PPO.MA_PPO.mappo_simple import AIFObsOvercookedMAEnv as SimpleSemanticEnv


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _pick_far_option(env, agent_idx: int):
    """
    Return (option_idx, destination, target_tile_xy, current_xy) for a destination
    that is far from the agent (Manhattan >= 2 grid cells).
    """
    state = env.state
    player = state.players[agent_idx]
    cur_xy = (int(player.position[0]), int(player.position[1]))

    _policies_env, metadata = env._build_dynamic_action_options(agent_idx)
    best_idx = None
    best_dist = -1
    best_dest = None
    best_target = None
    for i, m in enumerate(metadata):
        dest = m.get("destination", "?")
        if dest == "noop":
            continue
        target_rc = dyn_utils.DESTINATION_TO_TILE.get(dest)
        if target_rc is None:
            continue
        target_xy = (int(target_rc[1]), int(target_rc[0]))
        d = _manhattan(cur_xy, target_xy)
        if d > best_dist:
            best_dist = d
            best_idx = i
            best_dest = dest
            best_target = target_xy
    assert best_idx is not None, "Could not find any non-noop destination."
    assert best_dist >= 2, f"Destination not far enough (manhattan={best_dist})."
    return best_idx, best_dest, best_target, cur_xy


def _run_one(label: str, env_cls):
    env = env_cls({"layout": "cramped_room", "horizon": 400})
    env.reset(seed=0)

    pre_state = env.state
    p0_pre = (int(pre_state.players[0].position[0]), int(pre_state.players[0].position[1]))
    p1_pre = (int(pre_state.players[1].position[0]), int(pre_state.players[1].position[1]))

    opt_a0, dest_a0, target_a0, _ = _pick_far_option(env, agent_idx=0)
    opt_a1, dest_a1, target_a1, _ = _pick_far_option(env, agent_idx=1)

    print("")
    print(f"==> {label}")
    print(f"agent_0 pre xy={p0_pre} | chose dest={dest_a0!r} target_xy={target_a0}")
    print(f"agent_1 pre xy={p1_pre} | chose dest={dest_a1!r} target_xy={target_a1}")

    _, _, _, _, infos = env.step({"agent_0": opt_a0, "agent_1": opt_a1})

    post_state = env.state
    p0_post = (int(post_state.players[0].position[0]), int(post_state.players[0].position[1]))
    p1_post = (int(post_state.players[1].position[0]), int(post_state.players[1].position[1]))
    meta0 = infos["agent_0"]["semantic"]
    meta1 = infos["agent_1"]["semantic"]
    print(f"agent_0 post xy={p0_post} | primitive={meta0['primitive']}")
    print(f"agent_1 post xy={p1_post} | primitive={meta1['primitive']}")

    d0 = _manhattan(p0_pre, p0_post)
    d1 = _manhattan(p1_pre, p1_post)
    print(f"agent_0 manhattan delta = {d0}")
    print(f"agent_1 manhattan delta = {d1}")

    assert d0 <= 1, f"[{label}] AGENT 0 TELEPORTED! pre={p0_pre} post={p0_post}"
    assert d1 <= 1, f"[{label}] AGENT 1 TELEPORTED! pre={p1_pre} post={p1_post}"
    assert p0_post != target_a0, (
        f"[{label}] AGENT 0 jumped directly to target_xy={target_a0} in one step"
    )
    assert p1_post != target_a1, (
        f"[{label}] AGENT 1 jumped directly to target_xy={target_a1} in one step"
    )
    print(f"OK [{label}]: no teleporting; both agents moved at most one grid cell.")


def main():
    _run_one("mappo.SemanticAIFObsOvercookedRLlibEnv", SemanticAIFObsOvercookedRLlibEnv)
    _run_one("mappo_simple.AIFObsOvercookedMAEnv", SimpleSemanticEnv)
    print("")
    print("ALL OK: no teleporting in either env.")


if __name__ == "__main__":
    main()
