"""
Utility functions for Active Inference agent with functional generative model.

This module provides helper functions for:
- Dynamic semantic policy generation for Overcooked
- Path finding on the fixed map
- Orientation-aware policy compilation
- Policy deduplication / validation / padding
- Action sampling
- Observation formatting
- State belief conversion

Design choices
--------------
1. Primitive actions are encoded as:
       UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4, INTERACT=5
2. Semantic destinations are:
       onion1, onion2, dish, serve, pot, cntr1..cntr5, noop
3. Semantic modes are:
       stay, interact
4. Destination tiles are non-walkable object/counter tiles.
5. For EVERY semantic (destination, mode) pair, we generate EXACTLY ONE policy.
   We do NOT drop policies just because an interaction is unavailable or useless
   in the current state.
6. Policies are compiled to:
       move to an adjacent walkable support tile
       end facing the destination
       then either stop or INTERACT depending on mode
7. Movement actions are assumed to set orientation to the move direction.
8. There is no separate turn action. compile_semantic_policy uses a two-phase
   plan: shortest position-only walk to an adjacent support tile (empty if
   already there), then facing correction on that tile. If the required facing
   points into a map obstacle (non-walkable tile or out of bounds), a single
   press in that direction is treated as reorientation without moving (as when
   bumping a counter in Overcooked). Otherwise, oriented BFS may step off and
   return to fix facing.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Dict, List, Tuple, Optional, Sequence, Any

import numpy as np


# =============================================================================
# Primitive action constants
# =============================================================================

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4
INTERACT = 5

ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    STAY: "STAY",
    INTERACT: "INTERACT",
}

ORIENT_TO_ACTION = {
    "NORTH": UP,
    "SOUTH": DOWN,
    "WEST": LEFT,
    "EAST": RIGHT,
}

ACTION_TO_ORIENT = {
    UP: "NORTH",
    DOWN: "SOUTH",
    LEFT: "WEST",
    RIGHT: "EAST",
}


# =============================================================================
# Semantic space
# =============================================================================

DESTINATIONS = [
    "onion1",
    "onion2",
    "dish",
    "serve",
    "pot",
    "cntr1",
    "cntr2",
    "cntr3",
    "cntr4",
    "cntr5",
    "noop",
]

MODES = ["stay", "interact"]


# =============================================================================
# Fixed layout
# =============================================================================

LAYOUT_ROWS = [
    "XXPXX",
    "O1  O",
    "X  2X",
    "XDXSX",
]

GRID_H = len(LAYOUT_ROWS)
GRID_W = len(LAYOUT_ROWS[0])

# True walkable floor tiles only
WALKABLE_TILES = {
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
}

STATION_TILES = {
    "onion1": (1, 0),
    "onion2": (1, 4),
    "pot": (0, 2),
    "dish": (3, 1),
    "serve": (3, 3),
}

COUNTER_TILES = {
    "cntr1": (0, 1),
    "cntr2": (0, 3),
    "cntr3": (2, 0),
    "cntr4": (2, 4),
    "cntr5": (3, 2),
}

DESTINATION_TO_TILE = {**STATION_TILES, **COUNTER_TILES}


# =============================================================================
# Pretty / debug helpers
# =============================================================================

def action_seq_to_str(policy: Sequence[int]) -> str:
    return "[" + ", ".join(ACTION_NAMES.get(int(a), str(a)) for a in policy) + "]"


# =============================================================================
# Geometry helpers
# =============================================================================

def in_bounds(rc: Tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < GRID_H and 0 <= c < GRID_W


def is_walkable(rc: Tuple[int, int]) -> bool:
    return rc in WALKABLE_TILES


def neighbors(rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = rc
    cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [x for x in cand if in_bounds(x)]


def adjacent_walkable_tiles(target_tile: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [nb for nb in neighbors(target_tile) if is_walkable(nb)]


def required_facing_from_approach(
    approach_tile: Tuple[int, int],
    target_tile: Tuple[int, int],
) -> int:
    ar, ac = approach_tile
    tr, tc = target_tile
    dr, dc = tr - ar, tc - ac

    if (dr, dc) == (-1, 0):
        return UP
    if (dr, dc) == (1, 0):
        return DOWN
    if (dr, dc) == (0, -1):
        return LEFT
    if (dr, dc) == (0, 1):
        return RIGHT

    raise ValueError(f"{approach_tile} is not orthogonally adjacent to {target_tile}.")


def bfs_shortest_action_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked_tiles: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[List[int]]:
    """
    Return shortest primitive movement path from start to goal over WALKABLE tiles.

    Args:
        start: walkable tile
        goal: walkable tile
        blocked_tiles: additional temporarily blocked walkable tiles

    Returns:
        List of primitive actions, or None if unreachable
    """
    if not is_walkable(start) or not is_walkable(goal):
        return None

    if start == goal:
        return []

    blocked = set(blocked_tiles or [])
    if goal in blocked:
        blocked.remove(goal)

    transitions = [
        (UP, (-1, 0)),
        (DOWN, (1, 0)),
        (LEFT, (0, -1)),
        (RIGHT, (0, 1)),
    ]

    q = deque([start])
    parent = {start: None}
    parent_action: Dict[Tuple[int, int], int] = {}

    while q:
        cur = q.popleft()

        for action, (dr, dc) in transitions:
            nxt = (cur[0] + dr, cur[1] + dc)

            if not in_bounds(nxt):
                continue
            if not is_walkable(nxt):
                continue
            if nxt in blocked:
                continue
            if nxt in parent:
                continue

            parent[nxt] = cur
            parent_action[nxt] = action

            if nxt == goal:
                actions: List[int] = []
                node = nxt
                while parent[node] is not None:
                    actions.append(parent_action[node])
                    node = parent[node]
                actions.reverse()
                return actions

            q.append(nxt)

    return None


def all_shortest_action_paths(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked_tiles: Optional[Sequence[Tuple[int, int]]] = None,
    max_paths: int = 256,
) -> List[List[int]]:
    """
    Enumerate all shortest position-only action sequences from start to goal.

    Used so we can prefer a shortest walk that already ends with the required
    facing, instead of arbitrarily taking the first BFS reconstruction and then
    paying extra oriented-correction steps.
    """
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not is_walkable(start) or not is_walkable(goal):
        return []
    if start == goal:
        return [[]]

    blocked = set(blocked_tiles or [])
    if goal in blocked:
        blocked.remove(goal)

    transitions = [
        (UP, (-1, 0)),
        (DOWN, (1, 0)),
        (LEFT, (0, -1)),
        (RIGHT, (0, 1)),
    ]

    dist: Dict[Tuple[int, int], int] = {start: 0}
    preds: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int]]] = {start: []}

    q = deque([start])
    while q:
        cur = q.popleft()
        dcur = dist[cur]
        for action, (dr, dc) in transitions:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not in_bounds(nxt):
                continue
            if not is_walkable(nxt):
                continue
            if nxt in blocked:
                continue
            nd = dcur + 1
            if nxt not in dist:
                dist[nxt] = nd
                preds[nxt] = [(cur, action)]
                q.append(nxt)
            elif dist[nxt] == nd:
                preds[nxt].append((cur, action))

    if goal not in dist:
        return []

    paths: List[List[int]] = []

    def dfs_collect(node: Tuple[int, int], rev_actions: List[int]) -> None:
        if len(paths) >= max_paths:
            return
        if node == start:
            paths.append(list(reversed(rev_actions)))
            return
        for pred, act in preds[node]:
            rev_actions.append(act)
            dfs_collect(pred, rev_actions)
            rev_actions.pop()

    dfs_collect(goal, [])
    paths.sort(key=lambda p: tuple(p))
    return paths


def bfs_shortest_action_path_oriented(
    start_pos: Tuple[int, int],
    start_facing: int,
    goal_pos: Tuple[int, int],
    goal_facing: int,
    blocked_tiles: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[List[int]]:
    """
    Shortest path in augmented state space (position, facing).

    A move changes both:
    - position
    - facing (to the move direction)

    This lets the agent move away and come back in order to end on the desired
    support tile with the desired final facing when a single press toward the
    goal cannot be modeled as bump-turn (neighbor walkable).
    """
    if not is_walkable(start_pos) or not is_walkable(goal_pos):
        return None

    start_state = (start_pos, int(start_facing))
    goal_state = (goal_pos, int(goal_facing))

    if start_state == goal_state:
        return []

    blocked = set(blocked_tiles or [])
    if goal_pos in blocked:
        blocked.remove(goal_pos)

    transitions = [
        (UP, (-1, 0)),
        (DOWN, (1, 0)),
        (LEFT, (0, -1)),
        (RIGHT, (0, 1)),
    ]

    q = deque([start_state])
    parent = {start_state: None}
    parent_action: Dict[Tuple[Tuple[int, int], int], int] = {}

    while q:
        (cur_pos, cur_facing) = q.popleft()

        for action, (dr, dc) in transitions:
            nxt_pos = (cur_pos[0] + dr, cur_pos[1] + dc)
            nxt_facing = action
            nxt_state = (nxt_pos, nxt_facing)

            if not in_bounds(nxt_pos):
                continue
            if not is_walkable(nxt_pos):
                continue
            if nxt_pos in blocked:
                continue
            if nxt_state in parent:
                continue

            parent[nxt_state] = (cur_pos, cur_facing)
            parent_action[nxt_state] = action

            if nxt_state == goal_state:
                actions: List[int] = []
                node = nxt_state
                while parent[node] is not None:
                    actions.append(parent_action[node])
                    node = parent[node]
                actions.reverse()
                return actions

            q.append(nxt_state)

    return None


# =============================================================================
# State helpers
# =============================================================================

def _state_get(state: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in state:
            return state[k]
    return default


def get_self_pos(state: Dict[str, Any]) -> Tuple[int, int]:
    pos = _state_get(state, "self_pos", "agent_pos", "my_pos")
    if pos is None:
        raise KeyError("State must contain self_pos / agent_pos / my_pos.")
    return tuple(pos)


def get_other_pos(state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    pos = _state_get(state, "other_pos", "partner_pos", "other_agent_pos")
    return None if pos is None else tuple(pos)


def get_self_orient(state: Dict[str, Any]) -> str:
    orient = _state_get(state, "self_orient", "agent_orient", "my_orient", default="NORTH")
    return str(orient).upper()


def get_self_held(state: Dict[str, Any]) -> str:
    held = _state_get(state, "self_held", "agent_held", "my_held", default="nothing")
    return str(held).lower()


def get_pot_onions(state: Dict[str, Any]) -> int:
    val = _state_get(state, "pot_onions", default=None)
    if val is not None:
        return int(val)

    pot_state = str(_state_get(state, "pot_state", default="")).lower()
    mapping = {
        "empty": 0,
        "idle": 0,
        "one_onion": 1,
        "1": 1,
        "two_onions": 2,
        "2": 2,
        "three_onions": 3,
        "full": 3,
        "ready": 3,
        "cooking": 3,
    }
    return mapping.get(pot_state, 0)


def soup_ready(state: Dict[str, Any]) -> bool:
    val = _state_get(state, "soup_ready", default=None)
    if val is not None:
        return bool(val)

    pot_state = str(_state_get(state, "pot_state", default="")).lower()
    return pot_state in {"ready", "cooked", "soup_ready"}


def get_counter_contents(state: Dict[str, Any]) -> Dict[str, str]:
    raw = _state_get(state, "counter_contents", default=None)
    if raw is None:
        return {name: "empty" for name in COUNTER_TILES.keys()}

    out = {}
    for name in COUNTER_TILES.keys():
        out[name] = str(raw.get(name, "empty")).lower()
    return out


# =============================================================================
# Semantic validity rules
# =============================================================================

def interaction_is_valid(state: Dict[str, Any], destination: str) -> bool:
    """
    Kept as a diagnostic helper only.

    IMPORTANT:
    This function is NOT used to prune the generated semantic policy set.
    We always generate one policy per semantic (destination, mode) pair.
    """
    held = get_self_held(state)
    counters = get_counter_contents(state)
    pot_n = get_pot_onions(state)
    ready = soup_ready(state)

    if destination == "noop":
        return True

    if destination in ("onion1", "onion2"):
        return held == "nothing"

    if destination == "dish":
        return held == "nothing"

    if destination == "serve":
        return held == "soup"

    if destination == "pot":
        if held == "onion":
            return pot_n < 3 and not ready
        if held == "dish":
            return ready
        return False

    if destination.startswith("cntr"):
        content = counters[destination]
        if held == "nothing":
            return content in {"onion", "dish", "soup"}
        return content == "empty"

    return False


# =============================================================================
# Orientation-aware compilation
# =============================================================================

def _facing_after_primitive_path(start_facing: int, path: Sequence[int]) -> int:
    """Facing after executing movement primitives (each move sets facing to that action)."""
    f = int(start_facing)
    for a in path:
        f = int(a)
    return f


def _neighbor_cell(rc: Tuple[int, int], move_action: int) -> Tuple[int, int]:
    r, c = int(rc[0]), int(rc[1])
    a = int(move_action)
    if a == UP:
        return (r - 1, c)
    if a == DOWN:
        return (r + 1, c)
    if a == LEFT:
        return (r, c - 1)
    if a == RIGHT:
        return (r, c + 1)
    return (r, c)


def _map_blocks_walk_from(rc: Tuple[int, int], move_action: int) -> bool:
    """True if a move in move_action would not enter a walkable map cell (wall/station/oob)."""
    nxt = _neighbor_cell(rc, move_action)
    if not in_bounds(nxt):
        return True
    if not is_walkable(nxt):
        return True
    return False


def _facing_fix_press_into_map_obstacle(
    approach: Tuple[int, int],
    req_facing: int,
) -> Optional[List[int]]:
    """
    One press toward required_facing when that direction is blocked by map layout.

    Models bumping a counter/station: position stays on approach, facing becomes
    req_facing. Only used when the neighbor in req_facing is not walkable floor.
    """
    rf = int(req_facing)
    if rf in (STAY, INTERACT):
        return None
    ap = (int(approach[0]), int(approach[1]))
    if _map_blocks_walk_from(ap, rf):
        return [rf]
    return None


def _plan_two_phase_approach_path(
    self_pos: Tuple[int, int],
    start_facing: int,
    approach: Tuple[int, int],
    req_facing: int,
    blocked_tiles: Optional[Sequence[Tuple[int, int]]],
) -> Optional[List[int]]:
    """
    1) Shortest position-only walk to the approach tile (empty if already there).
       Among all shortest walks, prefer one whose final facing already matches
       req_facing (avoids spurious oriented correction from BFS tie-breaking).
    2) If no shortest walk ends with req_facing, append shortest oriented correction
       (approach,·)→(approach, req).
    """
    self_pos = (int(self_pos[0]), int(self_pos[1]))
    approach = (int(approach[0]), int(approach[1]))
    start_facing = int(start_facing)
    req_facing = int(req_facing)

    if self_pos == approach:
        path_geo: List[int] = []
        f_after = start_facing
    else:
        geo_candidates = all_shortest_action_paths(self_pos, approach, blocked_tiles)
        if not geo_candidates:
            return None

        best_geo: Optional[List[int]] = None
        best_face: Optional[List[int]] = None
        best_total = None

        for path_geo in geo_candidates:
            f_after = _facing_after_primitive_path(start_facing, path_geo)
            if f_after == req_facing:
                path_face: List[int] = []
            else:
                bump = _facing_fix_press_into_map_obstacle(approach, req_facing)
                if bump is not None:
                    pf = bump
                else:
                    pf = bfs_shortest_action_path_oriented(
                        approach,
                        f_after,
                        approach,
                        req_facing,
                        blocked_tiles,
                    )
                if pf is None:
                    continue
                path_face = list(pf)

            total = len(path_geo) + len(path_face)
            cand = (total, len(path_face), tuple(path_geo), path_geo, path_face)
            if best_total is None or cand[:3] < best_total[:3]:
                best_total = cand
                best_geo = list(path_geo)
                best_face = list(path_face)

        if best_geo is None:
            return None
        path_geo = best_geo
        path_face = best_face if best_face is not None else []

    if self_pos == approach:
        if f_after == req_facing:
            return path_geo
        bump = _facing_fix_press_into_map_obstacle(approach, req_facing)
        if bump is not None:
            path_face = bump
        else:
            path_face = bfs_shortest_action_path_oriented(
                approach,
                f_after,
                approach,
                req_facing,
                blocked_tiles,
            )
        if path_face is None:
            return None
        return path_geo + list(path_face)

    return path_geo + path_face


def _compile_actions_for_mode(
    path_to_approach: List[int],
    mode: str,
) -> List[int]:
    if mode == "stay":
        return list(path_to_approach) if len(path_to_approach) > 0 else [STAY]

    if mode == "interact":
        return list(path_to_approach) + [INTERACT] if len(path_to_approach) > 0 else [INTERACT]

    raise ValueError(f"Unknown mode: {mode}")


def compile_semantic_policy(
    state: Dict[str, Any],
    destination: str,
    mode: str,
    block_other_agent: bool = True,
) -> Dict[str, Any]:
    """
    Compile one semantic policy into a primitive action sequence.

    IMPORTANT:
    This function always returns exactly one policy dict for every valid
    semantic (destination, mode) pair in the library.

    We do NOT prune based on current state usefulness.
    """
    self_pos = get_self_pos(state)

    if not is_walkable(self_pos):
        raise ValueError(f"Self position must be on a walkable tile, got {self_pos}.")

    if destination == "noop":
        noop_action = STAY if mode == "stay" else INTERACT
        return {
            "destination": destination,
            "mode": mode,
            "target_tile": None,
            "approach_tile": self_pos,
            "required_facing": None,
            "final_facing": ORIENT_TO_ACTION[get_self_orient(state)],
            "path": [],
            "actions": [noop_action],
            "valid": True,
            "reason": "semantic noop",
        }

    if destination not in DESTINATION_TO_TILE:
        raise ValueError(f"Unknown destination: {destination}")

    target_tile = DESTINATION_TO_TILE[destination]
    candidate_approaches = adjacent_walkable_tiles(target_tile)

    if len(candidate_approaches) == 0:
        raise ValueError(f"No adjacent walkable approach tiles for destination {destination} at {target_tile}.")

    start_facing = ORIENT_TO_ACTION[get_self_orient(state)]

    blocked_tiles: List[Tuple[int, int]] = []
    if block_other_agent:
        other_pos = get_other_pos(state)
        if other_pos is not None and is_walkable(other_pos) and other_pos != self_pos:
            blocked_tiles.append(other_pos)

    best = None

    # First try with the other agent treated as blocked.
    # Two-phase: shortest walk to support tile, then oriented fix for facing only.
    for approach in sorted(candidate_approaches):
        req_facing = required_facing_from_approach(approach, target_tile)
        path = _plan_two_phase_approach_path(
            self_pos=self_pos,
            start_facing=start_facing,
            approach=approach,
            req_facing=req_facing,
            blocked_tiles=blocked_tiles,
        )
        if path is None:
            continue

        candidate = (
            len(path),
            approach,
            path,
            req_facing,
        )
        if best is None or candidate < best:
            best = candidate

    # If blocked planning fails, retry without temporarily blocking the other agent.
    if best is None and blocked_tiles:
        for approach in sorted(candidate_approaches):
            req_facing = required_facing_from_approach(approach, target_tile)
            path = _plan_two_phase_approach_path(
                self_pos=self_pos,
                start_facing=start_facing,
                approach=approach,
                req_facing=req_facing,
                blocked_tiles=None,
            )
            if path is None:
                continue

            candidate = (
                len(path),
                approach,
                path,
                req_facing,
            )
            if best is None or candidate < best:
                best = candidate

    # Absolute fallback to preserve semantic cardinality.
    # This should be very rare, but ensures one policy per semantic pair.
    if best is None:
        fallback_approach = sorted(candidate_approaches)[0]
        fallback_req = required_facing_from_approach(fallback_approach, target_tile)
        fallback_actions = [STAY] if mode == "stay" else [INTERACT]
        return {
            "destination": destination,
            "mode": mode,
            "target_tile": target_tile,
            "approach_tile": fallback_approach,
            "required_facing": fallback_req,
            "final_facing": start_facing,
            "path": [],
            "actions": fallback_actions,
            "valid": False,
            "reason": "fallback policy: no oriented path found",
        }

    _, best_approach, best_path, req_facing = best
    actions = _compile_actions_for_mode(best_path, mode)

    return {
        "destination": destination,
        "mode": mode,
        "target_tile": target_tile,
        "approach_tile": best_approach,
        "required_facing": req_facing,
        "final_facing": req_facing,
        "path": list(best_path),
        "actions": list(actions),
        "valid": True,
        "reason": (
            "two-phase: shortest walk to approach tile, then facing fix if needed; "
            + ("interact" if mode == "interact" else "stop")
        ),
    }


def generate_semantic_policy_metadata(
    state: Dict[str, Any],
    destinations: Optional[Sequence[str]] = None,
    modes: Optional[Sequence[str]] = None,
    block_other_agent: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate metadata for ALL semantic (destination, mode) pairs in stable order.
    """
    destinations = list(destinations) if destinations is not None else list(DESTINATIONS)
    modes = list(modes) if modes is not None else list(MODES)

    out = []
    for dest in destinations:
        for mode in modes:
            entry = compile_semantic_policy(
                state=state,
                destination=dest,
                mode=mode,
                block_other_agent=block_other_agent,
            )
            out.append(entry)

    return out


def generate_policies_from_state(
    state: Dict[str, Any],
    destinations: Optional[Sequence[str]] = None,
    modes: Optional[Sequence[str]] = None,
    block_other_agent: bool = True,
    deduplicate: bool = False,
    pad: bool = False,
    pad_action: int = STAY,
    return_metadata: bool = False,
):
    """
    Main entry point for dynamic policy generation.

    IMPORTANT:
    - By default, deduplicate=False so semantic cardinality is preserved.
    - With the default full semantic library, this returns exactly
      len(DESTINATIONS) * len(MODES) policies.
    """
    metadata = generate_semantic_policy_metadata(
        state=state,
        destinations=destinations,
        modes=modes,
        block_other_agent=block_other_agent,
    )

    policies = [list(m["actions"]) for m in metadata]

    if deduplicate:
        policies = deduplicate_policies(policies)

        seen = set()
        filtered_metadata = []
        for m in metadata:
            key = tuple(int(a) for a in m["actions"])
            if key not in seen:
                seen.add(key)
                filtered_metadata.append(m)
        metadata = filtered_metadata

    validate_policies(policies)

    if pad:
        padded_policies, lengths = pad_policies(policies, pad_action=pad_action)
        if return_metadata:
            return padded_policies, lengths, metadata
        return padded_policies, lengths

    if return_metadata:
        return policies, metadata
    return policies


# =============================================================================
# Dynamic Policy Utilities
# =============================================================================

def construct_policies(actions, policy_len):
    """
    Backward-compatible exhaustive constructor.
    """
    if policy_len == 1:
        return [[int(action)] for action in actions]

    policies = list(itertools.product(actions, repeat=policy_len))
    return [[int(a) for a in policy] for policy in policies]


def validate_policies(policies):
    if policies is None:
        raise ValueError("Policies cannot be None.")

    if not isinstance(policies, (list, tuple)):
        raise ValueError("Policies must be a list or tuple of policies.")

    if len(policies) == 0:
        raise ValueError("Policies cannot be empty.")

    for i, policy in enumerate(policies):
        if not isinstance(policy, (list, tuple, np.ndarray)):
            raise ValueError(f"Policy at index {i} must be a sequence, got {type(policy)}.")

        if len(policy) == 0:
            raise ValueError(f"Policy at index {i} is empty.")

        for t, action in enumerate(policy):
            if not isinstance(action, (int, np.integer)):
                raise ValueError(
                    f"Action at policies[{i}][{t}] must be an int, "
                    f"got {type(action)} with value {action!r}."
                )

    return True


def deduplicate_policies(policies):
    validate_policies(policies)

    seen = set()
    unique_policies = []

    for policy in policies:
        key = tuple(int(a) for a in policy)
        if key not in seen:
            seen.add(key)
            unique_policies.append([int(a) for a in policy])

    return unique_policies


def pad_policies(policies, pad_action):
    validate_policies(policies)

    if not isinstance(pad_action, (int, np.integer)):
        raise ValueError(f"pad_action must be an int, got {type(pad_action)} with value {pad_action!r}.")

    policy_lengths = [len(policy) for policy in policies]
    max_len = max(policy_lengths)

    padded_policies = []
    for policy in policies:
        padded = [int(a) for a in policy] + [int(pad_action)] * (max_len - len(policy))
        padded_policies.append(padded)

    return padded_policies, policy_lengths


# =============================================================================
# Action / Policy Sampling
# =============================================================================

def sample_action(q_pi, policies, action_selection="deterministic", alpha=16.0, actions=None):
    validate_policies(policies)

    q_pi = np.asarray(q_pi, dtype=np.float64)
    if q_pi.ndim != 1:
        raise ValueError("q_pi must be a 1D array.")
    if len(q_pi) != len(policies):
        raise ValueError("Length of q_pi must match number of policies.")

    if actions is not None:
        num_actions = len(actions)
    else:
        num_actions = max(max(int(a) for a in policy) for policy in policies) + 1

    action_marginals = np.zeros(num_actions, dtype=np.float64)
    for pol_idx, policy in enumerate(policies):
        first_action = int(policy[0])
        action_marginals[first_action] += q_pi[pol_idx]

    total = np.sum(action_marginals)
    if total <= 0.0:
        raise ValueError("Action marginals sum to zero; invalid q_pi or policies.")
    action_marginals = action_marginals / total

    if action_selection == "deterministic":
        selected_action = np.argmax(action_marginals)
    elif action_selection == "stochastic":
        log_marginals = log_stable(action_marginals)
        p_actions = softmax(log_marginals * alpha)
        selected_action = np.random.choice(num_actions, p=p_actions)
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")

    return int(selected_action)


def sample_policy(q_pi, policies, action_selection="deterministic", alpha=16.0):
    validate_policies(policies)

    q_pi = np.asarray(q_pi, dtype=np.float64)
    if q_pi.ndim != 1:
        raise ValueError("q_pi must be a 1D array.")
    if len(q_pi) != len(policies):
        raise ValueError("Length of q_pi must match number of policies.")

    if action_selection == "deterministic":
        policy_idx = np.argmax(q_pi)
    elif action_selection == "stochastic":
        log_q_pi = log_stable(q_pi)
        p_policies = softmax(log_q_pi * alpha)
        policy_idx = np.random.choice(len(policies), p=p_policies)
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")

    return int(policies[policy_idx][0])


# =============================================================================
# Numerical Stability
# =============================================================================

def log_stable(x, eps=1e-16):
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(x, eps))


def softmax(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# State Belief Format Conversion
# =============================================================================

def qs_dict_to_list(qs_dict, factor_order=None):
    if factor_order is None:
        factor_order = sorted(qs_dict.keys())
    return [qs_dict[factor] for factor in factor_order]


def qs_list_to_dict(qs_list, factor_order):
    return {factor: qs_list[i] for i, factor in enumerate(factor_order)}


# =============================================================================
# Observation Formatting
# =============================================================================

def format_observation(obs, obs_labels):
    return obs


def observation_to_one_hot(obs_idx, num_obs):
    one_hot = np.zeros(num_obs, dtype=np.float64)
    one_hot[int(obs_idx)] = 1.0
    return one_hot


def observations_to_one_hot(obs_dict, observation_labels):
    one_hot_dict = {}
    for modality, obs_idx in obs_dict.items():
        num_obs = len(observation_labels[modality])
        one_hot_dict[modality] = observation_to_one_hot(obs_idx, num_obs)
    return one_hot_dict


# =============================================================================
# Generic categorical sampling
# =============================================================================

def sample(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return np.random.choice(len(probs), p=probs)







WALKABLE_INDEX_TO_RC = {
    0: (1, 1),
    1: (1, 2),
    2: (1, 3),
    3: (2, 1),
    4: (2, 2),
    5: (2, 3),
}

def walkable_idx_to_rc(idx):
    idx = int(idx)
    if idx not in WALKABLE_INDEX_TO_RC:
        raise ValueError(f"Unknown walkable index {idx}")
    return WALKABLE_INDEX_TO_RC[idx]


def normalize_orientation(ori):
    mapping = {
        0: "NORTH",
        1: "SOUTH",
        2: "WEST",
        3: "EAST",
        "NORTH": "NORTH",
        "SOUTH": "SOUTH",
        "WEST": "WEST",
        "EAST": "EAST",
    }
    return mapping.get(ori, "NORTH")


def normalize_held(x):
    if x is None:
        return "nothing"
    s = str(x).lower()
    if s in {"none", "nothing", "empty", "0"}:
        return "nothing"
    if "onion" in s:
        return "onion"
    if "dish" in s or "plate" in s:
        return "dish"
    if "soup" in s:
        return "soup"
    return s


def build_policy_state(
    *,
    self_pos,
    self_orient,
    self_held,
    other_pos,
    other_orient,
    other_held,
    pot_state="empty",
    pot_onions=0,
    soup_ready=False,
    counter_contents=None,
):
    if counter_contents is None:
        counter_contents = {
            "cntr1": "empty",
            "cntr2": "empty",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "empty",
        }

    return {
        "self_pos": walkable_idx_to_rc(self_pos) if isinstance(self_pos, (int, np.integer)) else tuple(self_pos),
        "self_orient": normalize_orientation(self_orient),
        "self_held": normalize_held(self_held),
        "other_pos": walkable_idx_to_rc(other_pos) if isinstance(other_pos, (int, np.integer)) else tuple(other_pos),
        "other_orient": normalize_orientation(other_orient),
        "other_held": normalize_held(other_held),
        "pot_state": pot_state,
        "pot_onions": int(pot_onions),
        "soup_ready": bool(soup_ready),
        "counter_contents": dict(counter_contents),
    }