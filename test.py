"""
Comprehensive tests for IndividuallyCollective B.py (full transition model).

Covers: D (build_D, D_fn), env_utils, normalize, B_self_pos, B_self_orientation, B_self_held,
B_pot_state, B_checkboxes, B_fn.

Run from project root: python test.py
"""

import numpy as np

import generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective.model_init as model_init
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective.B import (
    normalize,
    B_self_pos,
    B_self_orientation,
    B_self_held,
    B_pot_state,
    B_checkboxes,
    B_fn,
)
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective.D import (
    build_D,
    D_fn,
    DEFAULT_START_GRID_XY,
)

# Layout: pos_w=1 ori=NORTH(0) -> POT; pos_w=5 ori=SOUTH(1) -> SERVE
# pos_w=0 ori=WEST(3) -> ONION; pos_w=3 ori=SOUTH(1) -> DISH
POS_AT_POT, ORI_FACE_POT = 1, 0
POS_AT_SERVE, ORI_FACE_SERVE = 5, 1
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective import env_utils as ind_col_env_utils
POS_AT_ONION, ORI_FACE_ONION = 0, 3
POS_AT_DISH, ORI_FACE_DISH = 3, 1


# =============================================================================
# D (prior beliefs): build_D, D_fn
# =============================================================================
def test_build_D_returns_all_state_factors():
    print("\n--- test_build_D_returns_all_state_factors ---")
    D = build_D()
    assert set(D.keys()) == set(model_init.states.keys())
    for factor in model_init.states:
        assert factor in D
        v = D[factor]
        assert isinstance(v, np.ndarray) and v.dtype == float
        if v.size > 0:
            assert np.allclose(v.sum(), 1.0), f"{factor} should sum to 1"
    print("test_build_D_returns_all_state_factors OK")


def test_build_D_default_priors():
    print("\n--- test_build_D_default_priors ---")
    D = build_D()
    assert np.argmax(D["self_orientation"]) == 0  # NORTH
    assert np.argmax(D["self_held"]) == model_init.HELD_NONE
    assert np.argmax(D["pot_state"]) == model_init.POT_0
    for ck in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
        assert D[ck][0] == 1.0 and D[ck][1] == 0.0
    # Default grid (1,2) -> walkable index 3
    expected_walkable = model_init.grid_idx_to_walkable_idx(model_init.xy_to_index(*DEFAULT_START_GRID_XY))
    assert np.argmax(D["self_pos"]) == (expected_walkable if expected_walkable is not None else 0)
    print("test_build_D_default_priors OK")


def test_build_D_custom_start_pos():
    print("\n--- test_build_D_custom_start_pos ---")
    D = build_D(self_start_pos=0)
    assert np.argmax(D["self_pos"]) == 0
    D = build_D(self_start_pos=5)
    assert np.argmax(D["self_pos"]) == 5
    print("test_build_D_custom_start_pos OK")


def test_build_D_custom_start_ori():
    print("\n--- test_build_D_custom_start_ori ---")
    for ori in range(model_init.N_DIRECTIONS):
        D = build_D(self_start_ori=ori)
        assert np.argmax(D["self_orientation"]) == ori
    print("test_build_D_custom_start_ori OK")


def test_build_D_out_of_range_fallback():
    print("\n--- test_build_D_out_of_range_fallback ---")
    D = build_D(self_start_pos=99)
    assert np.argmax(D["self_pos"]) == 0
    D = build_D(self_start_ori=-1)
    assert np.argmax(D["self_orientation"]) == 0
    print("test_build_D_out_of_range_fallback OK")


def test_D_fn_none_config():
    print("\n--- test_D_fn_none_config ---")
    D = D_fn(None)
    assert set(D.keys()) == set(model_init.states.keys())
    assert np.allclose(D["self_held"][model_init.HELD_NONE], 1.0)
    print("test_D_fn_none_config OK")


def test_D_fn_with_config():
    print("\n--- test_D_fn_with_config ---")
    D = D_fn({"self_start_pos": 0, "self_start_ori": 1})
    assert np.argmax(D["self_pos"]) == 0
    assert np.argmax(D["self_orientation"]) == 1
    D = D_fn({"self_start_ori": 2})
    assert np.argmax(D["self_orientation"]) == 2
    print("test_D_fn_with_config OK")


def test_build_D_shape_match_model_init():
    print("\n--- test_build_D_shape_match_model_init ---")
    D = build_D()
    assert D["self_pos"].shape == (model_init.N_WALKABLE,)
    assert D["self_orientation"].shape == (model_init.N_DIRECTIONS,)
    assert D["self_held"].shape == (model_init.N_HELD_TYPES,)
    assert D["pot_state"].shape == (model_init.N_POT_STATES,)
    for ck in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
        assert D[ck].shape == (2,)
    print("test_build_D_shape_match_model_init OK")

# =============================================================================
# env_utils (IndividuallyCollective)
# =============================================================================
def test_env_utils_action_roundtrip():
    """model_action_to_env_action and env_action_to_model_action roundtrip for 0..5."""
    print("\n--- test_env_utils_action_roundtrip ---")
    try:
        ind_col_env_utils.model_action_to_env_action(0)
    except (ImportError, ModuleNotFoundError) as e:
        print("  [SKIP] overcooked_ai_py not available:", e)
        return
    for a in range(model_init.N_ACTIONS):
        env_a = ind_col_env_utils.model_action_to_env_action(a)
        back = ind_col_env_utils.env_action_to_model_action(env_a)
        assert back == a, "action {} -> env -> {} (expected {})".format(a, back, a)
    print("test_env_utils_action_roundtrip OK")


def test_env_utils_observation_keys():
    """env_obs_to_model_obs returns keys matching model_init.observations."""
    print("\n--- test_env_utils_observation_keys ---")
    expected = set(model_init.observations.keys())
    # Use a minimal mock state so we don't depend on env
    class MockPlayer:
        def __init__(self, position, orientation, has_object=False, held_name=None):
            self.position = position
            self.orientation = orientation
            self._has_object = has_object
            self._held_name = held_name
        def has_object(self):
            return self._has_object
        @property
        def held_object(self):
            if not self._has_object:
                return None
            class Obj:
                pass
            o = Obj()
            o.name = self._held_name
            return o
    class MockState:
        def __init__(self):
            grid_idx = model_init.WALKABLE_INDICES[0]
            x, y = model_init.index_to_xy(grid_idx)
            self.players = [MockPlayer((x, y), model_init.DIRECTIONS[0], False, None)]
            self.objects = {}
    state = MockState()
    obs = ind_col_env_utils.env_obs_to_model_obs(state, 0)
    assert set(obs.keys()) == expected
    assert 0 <= obs["self_pos_obs"] < model_init.N_WALKABLE
    assert 0 <= obs["self_orientation_obs"] < model_init.N_DIRECTIONS
    assert 0 <= obs["self_held_obs"] < model_init.N_HELD_TYPES
    assert 0 <= obs["pot_state_obs"] < model_init.N_POT_STATES
    assert obs["soup_delivered_obs"] in (0, 1)
    print("test_env_utils_observation_keys OK")


def test_env_utils_get_D_config_keys():
    """get_D_config_from_state returns self_start_pos, self_start_ori."""
    print("\n--- test_env_utils_get_D_config_keys ---")
    class MockPlayer:
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation
    class MockState:
        def __init__(self):
            grid_idx = model_init.WALKABLE_INDICES[0]
            x, y = model_init.index_to_xy(grid_idx)
            self.players = [MockPlayer((x, y), model_init.DIRECTIONS[0])]
    state = MockState()
    config = ind_col_env_utils.get_D_config_from_state(state, 0)
    assert "self_start_pos" in config and "self_start_ori" in config
    assert 0 <= config["self_start_pos"] < model_init.N_WALKABLE
    assert 0 <= config["self_start_ori"] < model_init.N_DIRECTIONS
    D = build_D(**config)
    assert np.argmax(D["self_pos"]) == config["self_start_pos"]
    assert np.argmax(D["self_orientation"]) == config["self_start_ori"]
    print("test_env_utils_get_D_config_keys OK")


def test_env_utils_reward_info_soup_delivered():
    """soup_delivered_obs is 1 when reward_info indicates sparse reward for agent."""
    print("\n--- test_env_utils_reward_info_soup_delivered ---")
    class MockPlayer:
        def __init__(self):
            self.position = (1, 1)
            self.orientation = model_init.DIRECTIONS[0]
            self._has_object = False
        def has_object(self):
            return self._has_object
        @property
        def held_object(self):
            return None
    class MockState:
        def __init__(self):
            self.players = [MockPlayer()]
            self.objects = {}
    state = MockState()
    obs_none = ind_col_env_utils.env_obs_to_model_obs(state, 0, reward_info=None)
    assert obs_none["soup_delivered_obs"] == 0
    obs_zero = ind_col_env_utils.env_obs_to_model_obs(state, 0, reward_info={"sparse_reward_by_agent": [0]})
    assert obs_zero["soup_delivered_obs"] == 0
    obs_one = ind_col_env_utils.env_obs_to_model_obs(state, 0, reward_info={"sparse_reward_by_agent": [1]})
    assert obs_one["soup_delivered_obs"] == 1
    obs_fallback = ind_col_env_utils.env_obs_to_model_obs(state, 0, reward_info={"sparse_reward": 5})
    assert obs_fallback["soup_delivered_obs"] == 1
    print("test_env_utils_reward_info_soup_delivered OK")


def test_env_utils_against_env():
    """Against real Overcooked env: reset, get state, env_obs_to_model_obs and get_D_config_from_state."""
    print("\n--- test_env_utils_against_env ---")
    try:
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        print("  [SKIP] Overcooked env not available:", e)
        return
    try:
        env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": 100})
        obs, infos = env.reset(seed=42)
        state = infos["agent_0"]["state"]
    except Exception as e:
        print("  [SKIP] Env reset failed:", e)
        return
    model_obs = ind_col_env_utils.env_obs_to_model_obs(state, 0, reward_info=None)
    assert set(model_obs.keys()) == set(model_init.observations.keys())
    assert 0 <= model_obs["self_pos_obs"] < model_init.N_WALKABLE
    assert 0 <= model_obs["self_orientation_obs"] < model_init.N_DIRECTIONS
    assert 0 <= model_obs["self_held_obs"] < model_init.N_HELD_TYPES
    assert 0 <= model_obs["pot_state_obs"] < model_init.N_POT_STATES
    assert model_obs["soup_delivered_obs"] in (0, 1)
    config = ind_col_env_utils.get_D_config_from_state(state, 0)
    assert "self_start_pos" in config and "self_start_ori" in config
    assert 0 <= config["self_start_pos"] < model_init.N_WALKABLE
    assert 0 <= config["self_start_ori"] < model_init.N_DIRECTIONS
    D = build_D(**config)
    assert set(D.keys()) == set(model_init.states.keys())
    for a in range(model_init.N_ACTIONS):
        env_a = ind_col_env_utils.model_action_to_env_action(a)
        back = ind_col_env_utils.env_action_to_model_action(env_a)
        assert back == a
    print("test_env_utils_against_env OK")


def _parents(
    pos_one_hot=None,
    ori_one_hot=None,
    held_one_hot=None,
    pot_one_hot=None,
    ck_put1=(1.0, 0.0),
    ck_put2=(1.0, 0.0),
    ck_put3=(1.0, 0.0),
    ck_plated=(1.0, 0.0),
    ck_delivered=(1.0, 0.0),
):
    """Build parents dict with one-hot or uniform beliefs."""
    nw, nd, nh, npot = model_init.N_WALKABLE, model_init.N_DIRECTIONS, model_init.N_HELD_TYPES, model_init.N_POT_STATES
    if pos_one_hot is None:
        pos_one_hot = np.ones(nw) / nw
    else:
        pos_one_hot = np.asarray(pos_one_hot, dtype=float)
        pos_one_hot = pos_one_hot / pos_one_hot.sum()
    if ori_one_hot is None:
        ori_one_hot = np.ones(nd) / nd
    else:
        ori_one_hot = np.asarray(ori_one_hot, dtype=float)
        ori_one_hot = ori_one_hot / ori_one_hot.sum()
    if held_one_hot is None:
        held_one_hot = np.ones(nh) / nh
    else:
        held_one_hot = np.asarray(held_one_hot, dtype=float)
        held_one_hot = held_one_hot / held_one_hot.sum()
    if pot_one_hot is None:
        pot_one_hot = np.ones(npot) / npot
    else:
        pot_one_hot = np.asarray(pot_one_hot, dtype=float)
        pot_one_hot = pot_one_hot / pot_one_hot.sum()
    return {
        "self_pos": pos_one_hot,
        "self_orientation": ori_one_hot,
        "self_held": held_one_hot,
        "pot_state": pot_one_hot,
        "ck_put1": np.asarray(ck_put1, dtype=float),
        "ck_put2": np.asarray(ck_put2, dtype=float),
        "ck_put3": np.asarray(ck_put3, dtype=float),
        "ck_plated": np.asarray(ck_plated, dtype=float),
        "ck_delivered": np.asarray(ck_delivered, dtype=float),
    }


def _one_hot(i, n):
    a = np.zeros(n)
    a[i] = 1.0
    return a


ACTION_NAMES = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]


def _summarize_parents(parents):
    """Short summary of parent beliefs for logging."""
    parts = []
    pos = parents["self_pos"]
    if np.max(pos) >= 0.99:
        parts.append(f"pos=w{np.argmax(pos)}")
    else:
        parts.append("pos=mixed")
    ori = parents["self_orientation"]
    if np.max(ori) >= 0.99:
        parts.append(f"ori={np.argmax(ori)}")
    else:
        parts.append("ori=mixed")
    held = parents["self_held"]
    if np.max(held) >= 0.99:
        parts.append(f"held={np.argmax(held)}")
    else:
        parts.append("held=mixed")
    pot = parents["pot_state"]
    if np.max(pot) >= 0.99:
        parts.append(f"pot=POT_{np.argmax(pot)}")
    else:
        parts.append("pot=mixed")
    ck1, ck2, ck3 = parents["ck_put1"], parents["ck_put2"], parents["ck_put3"]
    parts.append(f"ck_put1=[{ck1[0]:.2f},{ck1[1]:.2f}] ck_put2=[{ck2[0]:.2f},{ck2[1]:.2f}] ck_put3=[{ck3[0]:.2f},{ck3[1]:.2f}]")
    plat, deliv = parents["ck_plated"], parents["ck_delivered"]
    parts.append(f"ck_plated=[{plat[0]:.2f},{plat[1]:.2f}] ck_delivered=[{deliv[0]:.2f},{deliv[1]:.2f}]")
    return " ".join(parts)


def _print_out(label, out):
    for k in ["ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"]:
        v = out[k]
        print(f"    {label} {k}: [{v[0]:.6f}, {v[1]:.6f}] sum={v.sum():.6f}")


# =============================================================================
# NORMALIZE
# =============================================================================
def test_normalize():
    print("\n--- test_normalize ---")
    p = np.array([0.25, 0.25, 0.5])
    out = normalize(p)
    assert np.allclose(out.sum(), 1.0), "sum to 1"
    assert np.allclose(out, [0.25, 0.25, 0.5])
    z = np.array([0.0, 0.0])
    out_z = normalize(z)
    assert out_z.shape == (2,) and np.all(np.isfinite(out_z))
    print("test_normalize OK")


# =============================================================================
# B_self_pos
# =============================================================================
def test_B_self_pos_stay_interact_no_move():
    print("\n--- test_B_self_pos_stay_interact_no_move ---")
    for action in [model_init.STAY, model_init.INTERACT]:
        parents = {"self_pos": _one_hot(2, model_init.N_WALKABLE)}
        out = B_self_pos(parents, action)
        assert out.shape == (model_init.N_WALKABLE,) and np.allclose(out.sum(), 1.0)
        assert np.argmax(out) == 2
    print("test_B_self_pos_stay_interact_no_move OK")


def test_B_self_pos_north_moves():
    print("\n--- test_B_self_pos_north_moves ---")
    parents = {"self_pos": _one_hot(4, model_init.N_WALKABLE)}
    out = B_self_pos(parents, model_init.NORTH)
    assert np.argmax(out) == 1
    assert np.allclose(out.sum(), 1.0)
    print("test_B_self_pos_north_moves OK")


def test_B_self_pos_south_moves():
    print("\n--- test_B_self_pos_south_moves ---")
    parents = {"self_pos": _one_hot(1, model_init.N_WALKABLE)}
    out = B_self_pos(parents, model_init.SOUTH)
    assert np.argmax(out) == 4
    print("test_B_self_pos_south_moves OK")


# =============================================================================
# B_self_orientation
# =============================================================================
def test_B_self_orientation_directional_sets():
    print("\n--- test_B_self_orientation_directional_sets ---")
    for action, expected_ori in [(model_init.NORTH, 0), (model_init.SOUTH, 1), (model_init.EAST, 2), (model_init.WEST, 3)]:
        parents = {"self_orientation": _one_hot(2, model_init.N_DIRECTIONS)}
        out = B_self_orientation(parents, action)
        assert np.argmax(out) == expected_ori and np.allclose(out.sum(), 1.0)
    print("test_B_self_orientation_directional_sets OK")


def test_B_self_orientation_stay_keeps():
    print("\n--- test_B_self_orientation_stay_keeps ---")
    parents = {"self_orientation": _one_hot(3, model_init.N_DIRECTIONS)}
    out = B_self_orientation(parents, model_init.STAY)
    assert np.argmax(out) == 3
    print("test_B_self_orientation_stay_keeps OK")


# =============================================================================
# B_self_held
# =============================================================================
def test_B_self_held_pick_onion():
    print("\n--- test_B_self_held_pick_onion ---")
    parents = {
        "self_pos": _one_hot(POS_AT_ONION, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_ONION, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_NONE, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_0, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.INTERACT)
    assert out[model_init.HELD_ONION] > 0.8 and out[model_init.HELD_NONE] > 0.05
    assert np.allclose(out.sum(), 1.0)
    print("test_B_self_held_pick_onion OK")


def test_B_self_held_pick_dish():
    print("\n--- test_B_self_held_pick_dish ---")
    parents = {
        "self_pos": _one_hot(POS_AT_DISH, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_DISH, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_NONE, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_0, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.INTERACT)
    assert out[model_init.HELD_DISH] > 0.8
    print("test_B_self_held_pick_dish OK")


def test_B_self_held_put_onion():
    print("\n--- test_B_self_held_put_onion ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_ONION, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_0, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.INTERACT)
    assert out[model_init.HELD_NONE] > 0.8
    print("test_B_self_held_put_onion OK")


def test_B_self_held_take_soup():
    print("\n--- test_B_self_held_take_soup ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_DISH, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_3, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.INTERACT)
    assert out[model_init.HELD_SOUP] > 0.8
    print("test_B_self_held_take_soup OK")


def test_B_self_held_deliver():
    print("\n--- test_B_self_held_deliver ---")
    parents = {
        "self_pos": _one_hot(POS_AT_SERVE, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_SERVE, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_SOUP, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_0, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.INTERACT)
    assert out[model_init.HELD_NONE] > 0.8
    print("test_B_self_held_deliver OK")


def test_B_self_held_non_interact_unchanged():
    print("\n--- test_B_self_held_non_interact_unchanged ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_SOUP, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_3, model_init.N_POT_STATES),
    }
    out = B_self_held(parents, model_init.STAY)
    assert np.argmax(out) == model_init.HELD_SOUP
    print("test_B_self_held_non_interact_unchanged OK")


# =============================================================================
# B_pot_state
# =============================================================================
def test_B_pot_state_put_onion_POT_0_to_1():
    print("\n--- test_B_pot_state_put_onion_POT_0_to_1 ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_ONION, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_0, model_init.N_POT_STATES),
    }
    out = B_pot_state(parents, model_init.INTERACT)
    assert out[model_init.POT_1] > 0.8 and np.allclose(out.sum(), 1.0)
    print("test_B_pot_state_put_onion_POT_0_to_1 OK")


def test_B_pot_state_take_soup_POT_3_to_0():
    print("\n--- test_B_pot_state_take_soup_POT_3_to_0 ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_DISH, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_3, model_init.N_POT_STATES),
    }
    out = B_pot_state(parents, model_init.INTERACT)
    assert out[model_init.POT_0] > 0.8
    print("test_B_pot_state_take_soup_POT_3_to_0 OK")


def test_B_pot_state_non_interact_unchanged():
    print("\n--- test_B_pot_state_non_interact_unchanged ---")
    parents = {
        "self_pos": _one_hot(POS_AT_POT, model_init.N_WALKABLE),
        "self_orientation": _one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        "self_held": _one_hot(model_init.HELD_ONION, model_init.N_HELD_TYPES),
        "pot_state": _one_hot(model_init.POT_2, model_init.N_POT_STATES),
    }
    out = B_pot_state(parents, model_init.STAY)
    assert np.argmax(out) == model_init.POT_2
    print("test_B_pot_state_non_interact_unchanged OK")


# =============================================================================
# B_fn (full integration)
# =============================================================================
def test_B_fn_returns_all_factors():
    print("\n--- test_B_fn_returns_all_factors ---")
    expected_keys = {"self_pos", "self_orientation", "self_held", "pot_state", "ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"}
    qs = _parents()
    out = B_fn(qs, model_init.STAY)
    assert set(out.keys()) == expected_keys
    for k, v in out.items():
        assert isinstance(v, np.ndarray) and np.allclose(v.sum(), 1.0, atol=1e-5)
    print("test_B_fn_returns_all_factors OK")


def test_B_fn_shape_and_normalization():
    print("\n--- test_B_fn_shape_and_normalization ---")
    qs = _parents()
    for action in range(model_init.N_ACTIONS):
        out = B_fn(qs, action)
        assert out["self_pos"].shape == (model_init.N_WALKABLE,)
        assert out["self_orientation"].shape == (model_init.N_DIRECTIONS,)
        assert out["self_held"].shape == (model_init.N_HELD_TYPES,)
        assert out["pot_state"].shape == (model_init.N_POT_STATES,)
        for ck in ["ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"]:
            assert out[ck].shape == (2,)
        for k, v in out.items():
            assert np.allclose(v.sum(), 1.0, atol=1e-5)
    print("test_B_fn_shape_and_normalization OK")


def test_B_fn_noise_zero_deterministic():
    print("\n--- test_B_fn_noise_zero_deterministic ---")
    qs = _parents(
        pos_one_hot=_one_hot(0, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(0, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_NONE, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_0, model_init.N_POT_STATES),
    )
    out = B_fn(qs, model_init.STAY, B_NOISE_LEVEL=0.0)
    assert np.argmax(out["self_pos"]) == 0 and np.argmax(out["self_orientation"]) == 0
    assert np.argmax(out["self_held"]) == model_init.HELD_NONE and np.argmax(out["pot_state"]) == model_init.POT_0
    print("test_B_fn_noise_zero_deterministic OK")


def test_B_fn_noise_positive_spreads_mass():
    print("\n--- test_B_fn_noise_positive_spreads_mass ---")
    qs = _parents(
        pos_one_hot=_one_hot(2, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(1, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_SOUP, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_3, model_init.N_POT_STATES),
    )
    out_no_noise = B_fn(qs, model_init.STAY, B_NOISE_LEVEL=0.0)
    out_noise = B_fn(qs, model_init.STAY, B_NOISE_LEVEL=0.1)
    assert np.max(out_noise["self_pos"]) < np.max(out_no_noise["self_pos"])
    assert np.allclose(out_noise["self_pos"].sum(), 1.0)
    print("test_B_fn_noise_positive_spreads_mass OK")


# ---------------------------------------------------------------------------
# B_checkboxes: 1) Return shape and normalization
# ---------------------------------------------------------------------------
def test_return_shape_and_normalization():
    print("\n--- test_return_shape_and_normalization ---")
    print("  Step: build uniform parents, action=STAY.")
    parents = _parents()
    print("  Parents:", _summarize_parents(parents))
    print("  Step: call B_checkboxes(parents, STAY).")
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Step: assert keys are exactly ck_put1, ck_put2, ck_put3, ck_plated, ck_delivered.")
    assert set(out.keys()) == {"ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"}
    print("  Step: assert each value has shape (2,) and sums to 1 and is non-negative.")
    for k, v in out.items():
        assert v.shape == (2,), f"{k} shape"
        assert np.allclose(v.sum(), 1.0), f"{k} normalizes to 1"
        assert np.all(v >= -1e-9), f"{k} non-negative"
        print(f"    {k}: shape={v.shape} sum={v.sum():.6f} min={v.min():.6f}")
    print("test_return_shape_and_normalization OK")


# ---------------------------------------------------------------------------
# 2) Non-INTERACT: no delivery, no plating; put1/2/3 from pot only
# ---------------------------------------------------------------------------
def test_non_interact_no_delivery_no_plating():
    print("\n--- test_non_interact_no_delivery_no_plating ---")
    for action in [model_init.NORTH, model_init.SOUTH, model_init.STAY]:
        aname = ACTION_NAMES[action]
        print(f"  Step: action={aname} ({action}).")
        parents = _parents(
            pot_one_hot=_one_hot(model_init.POT_0, 4),
            ck_put1=(0.0, 1.0),
            ck_put2=(0.0, 1.0),
            ck_put3=(0.0, 1.0),
            ck_plated=(0.0, 1.0),
            ck_delivered=(1.0, 0.0),
        )
        print("  Parents:", _summarize_parents(parents))
        out = B_checkboxes(parents, action)
        print("  Output:")
        _print_out("out", out)
        print(f"  Assert: ck_delivered[1] == 0.0 (actual={out['ck_delivered'][1]})")
        assert out["ck_delivered"][1] == 0.0, "delivered is pulse 0 when no INTERACT deliver"
    print("test_non_interact_no_delivery_no_plating OK")


# ---------------------------------------------------------------------------
# 3) INTERACT at SERVE with HELD_SOUP -> delivery (pulse)
# ---------------------------------------------------------------------------
def test_delivery_pulse():
    print("\n--- test_delivery_pulse ---")
    print("  Step: pos=POS_AT_SERVE(5), ori=ORI_FACE_SERVE(1), held=HELD_SOUP, pot=POT_0, all checkboxes set (put1/2/3/plated=1, delivered=0).")
    parents = _parents(
        pos_one_hot=_one_hot(POS_AT_SERVE, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(ORI_FACE_SERVE, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_SOUP, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_0, 4),
        ck_put1=(0.0, 1.0),
        ck_put2=(0.0, 1.0),
        ck_put3=(0.0, 1.0),
        ck_plated=(0.0, 1.0),
        ck_delivered=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    print("  Step: call B_checkboxes(parents, INTERACT).")
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_delivered[1]==1, ck_delivered[0]==0; ck_put1/2/3 and ck_plated reset to [1,0].")
    assert out["ck_delivered"][1] == 1.0, "delivered pulse = 1 when INTERACT at serve with soup"
    assert out["ck_delivered"][0] == 0.0
    assert out["ck_put1"][0] == 1.0 and out["ck_put1"][1] == 0.0
    assert out["ck_put2"][0] == 1.0 and out["ck_put2"][1] == 0.0
    assert out["ck_put3"][0] == 1.0 and out["ck_put3"][1] == 0.0
    assert out["ck_plated"][0] == 1.0 and out["ck_plated"][1] == 0.0
    print("test_delivery_pulse OK")


# ---------------------------------------------------------------------------
# 4) INTERACT at POT with HELD_DISH and POT_3 -> plating
# ---------------------------------------------------------------------------
def test_plating_event():
    print("\n--- test_plating_event ---")
    print("  Step: pos=POS_AT_POT(1), ori=ORI_FACE_POT(0), held=HELD_DISH, pot=POT_3, ck_plated=[1,0], ck_delivered=[1,0].")
    parents = _parents(
        pos_one_hot=_one_hot(POS_AT_POT, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_DISH, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_3, 4),
        ck_plated=(1.0, 0.0),
        ck_delivered=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    print("  Step: call B_checkboxes(parents, INTERACT).")
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_plated[1]==1, ck_delivered[1]==0.")
    assert out["ck_plated"][1] == 1.0, "plated = 1 when INTERACT at pot with dish and POT_3"
    assert out["ck_delivered"][1] == 0.0, "no delivery"
    print("test_plating_event OK")


# ---------------------------------------------------------------------------
# 5) Pot-derived progress: POT_0 -> put1/2/3 all 0
# ---------------------------------------------------------------------------
def test_pot_0_put_progress():
    print("\n--- test_pot_0_put_progress ---")
    print("  Step: pot=POT_0, ck_put1/2/3 all [1,0], action=STAY.")
    parents = _parents(
        pot_one_hot=_one_hot(model_init.POT_0, 4),
        ck_put1=(1.0, 0.0),
        ck_put2=(1.0, 0.0),
        ck_put3=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_put1[1]==0, ck_put2[1]==0, ck_put3[1]==0.")
    assert out["ck_put1"][1] == 0.0
    assert out["ck_put2"][1] == 0.0
    assert out["ck_put3"][1] == 0.0
    print("test_pot_0_put_progress OK")


# ---------------------------------------------------------------------------
# 6) Pot-derived: POT_1 -> put1=1, put2/3=0
# ---------------------------------------------------------------------------
def test_pot_1_put_progress():
    print("\n--- test_pot_1_put_progress ---")
    print("  Step: pot=POT_1, ck_put1/2/3 all [1,0], action=STAY.")
    parents = _parents(
        pot_one_hot=_one_hot(model_init.POT_1, 4),
        ck_put1=(1.0, 0.0),
        ck_put2=(1.0, 0.0),
        ck_put3=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_put1[1]==1, ck_put2[1]==0, ck_put3[1]==0.")
    assert out["ck_put1"][1] == 1.0
    assert out["ck_put2"][1] == 0.0
    assert out["ck_put3"][1] == 0.0
    print("test_pot_1_put_progress OK")


# ---------------------------------------------------------------------------
# 7) Pot-derived: POT_2 -> put1=put2=1, put3=0
# ---------------------------------------------------------------------------
def test_pot_2_put_progress():
    print("\n--- test_pot_2_put_progress ---")
    print("  Step: pot=POT_2, ck_put1/2/3 all [1,0], action=STAY.")
    parents = _parents(
        pot_one_hot=_one_hot(model_init.POT_2, 4),
        ck_put1=(1.0, 0.0),
        ck_put2=(1.0, 0.0),
        ck_put3=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_put1[1]==1, ck_put2[1]==1, ck_put3[1]==0.")
    assert out["ck_put1"][1] == 1.0
    assert out["ck_put2"][1] == 1.0
    assert out["ck_put3"][1] == 0.0
    print("test_pot_2_put_progress OK")


# ---------------------------------------------------------------------------
# 8) Pot-derived: POT_3 -> put1=put2=put3=1
# ---------------------------------------------------------------------------
def test_pot_3_put_progress():
    print("\n--- test_pot_3_put_progress ---")
    print("  Step: pot=POT_3, ck_put1/2/3 all [1,0], action=STAY.")
    parents = _parents(
        pot_one_hot=_one_hot(model_init.POT_3, 4),
        ck_put1=(1.0, 0.0),
        ck_put2=(1.0, 0.0),
        ck_put3=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_put1[1]==1, ck_put2[1]==1, ck_put3[1]==1.")
    assert out["ck_put1"][1] == 1.0
    assert out["ck_put2"][1] == 1.0
    assert out["ck_put3"][1] == 1.0
    print("test_pot_3_put_progress OK")


# ---------------------------------------------------------------------------
# 9) Monotonic: once put1=1 it stays 1 (until delivery)
# ---------------------------------------------------------------------------
def test_monotonic_put1_persists():
    print("\n--- test_monotonic_put1_persists ---")
    print("  Step: pot=POT_1, ck_put1=[0,1] (already 1), ck_put2/3=[1,0], action=STAY.")
    parents = _parents(
        pot_one_hot=_one_hot(model_init.POT_1, 4),
        ck_put1=(0.0, 1.0),  # already 1
        ck_put2=(1.0, 0.0),
        ck_put3=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_put1[1]==1 (monotonic persistence).")
    assert out["ck_put1"][1] == 1.0
    print("test_monotonic_put1_persists OK")


# ---------------------------------------------------------------------------
# 10) No delivery when at SERVE but not holding soup
# ---------------------------------------------------------------------------
def test_at_serve_no_soup_no_delivery():
    print("\n--- test_at_serve_no_soup_no_delivery ---")
    print("  Step: pos=POS_AT_SERVE(5), ori=ORI_FACE_SERVE(1), held=HELD_NONE, action=INTERACT.")
    parents = _parents(
        pos_one_hot=_one_hot(POS_AT_SERVE, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(ORI_FACE_SERVE, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_NONE, model_init.N_HELD_TYPES),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_delivered[1]==0 (no soup -> no delivery).")
    assert out["ck_delivered"][1] == 0.0
    print("test_at_serve_no_soup_no_delivery OK")


# ---------------------------------------------------------------------------
# 11) No plating when at POT but not holding dish or pot not ready
# ---------------------------------------------------------------------------
def test_at_pot_no_dish_no_plating():
    print("\n--- test_at_pot_no_dish_no_plating ---")
    print("  Step: pos=POS_AT_POT(1), ori=ORI_FACE_POT(0), held=HELD_ONION, pot=POT_3, ck_plated=[1,0], action=INTERACT.")
    parents = _parents(
        pos_one_hot=_one_hot(POS_AT_POT, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_ONION, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_3, 4),
        ck_plated=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_plated[1]==0 (holding onion, not dish).")
    assert out["ck_plated"][1] == 0.0
    print("test_at_pot_no_dish_no_plating OK")


def test_at_pot_dish_but_pot_not_ready_no_plating():
    print("\n--- test_at_pot_dish_but_pot_not_ready_no_plating ---")
    print("  Step: pos=POS_AT_POT(1), ori=ORI_FACE_POT(0), held=HELD_DISH, pot=POT_0, ck_plated=[1,0], action=INTERACT.")
    parents = _parents(
        pos_one_hot=_one_hot(POS_AT_POT, model_init.N_WALKABLE),
        ori_one_hot=_one_hot(ORI_FACE_POT, model_init.N_DIRECTIONS),
        held_one_hot=_one_hot(model_init.HELD_DISH, model_init.N_HELD_TYPES),
        pot_one_hot=_one_hot(model_init.POT_0, 4),
        ck_plated=(1.0, 0.0),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: ck_plated[1]==0 (pot not ready).")
    assert out["ck_plated"][1] == 0.0
    print("test_at_pot_dish_but_pot_not_ready_no_plating OK")


# ---------------------------------------------------------------------------
# 12) Mixed belief: partial probability of delivery
# ---------------------------------------------------------------------------
def test_partial_delivery_probability():
    print("\n--- test_partial_delivery_probability ---")
    print("  Step: 50% pos=POS_AT_SERVE / 50% pos=0; 50% ori=ORI_FACE_SERVE / 50% ori=0; held=HELD_SOUP; action=INTERACT.")
    pos = np.zeros(model_init.N_WALKABLE)
    pos[POS_AT_SERVE] = 0.5
    pos[0] = 0.5
    ori = np.zeros(model_init.N_DIRECTIONS)
    ori[ORI_FACE_SERVE] = 0.5
    ori[0] = 0.5
    held = _one_hot(model_init.HELD_SOUP, model_init.N_HELD_TYPES)
    parents = _parents(pos_one_hot=pos, ori_one_hot=ori, held_one_hot=held)
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.INTERACT)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: 0.2 <= ck_delivered[1] <= 0.3 (p_deliver_now = 0.5*0.5*1 = 0.25). Actual:", out["ck_delivered"][1])
    assert 0.2 <= out["ck_delivered"][1] <= 0.3
    print("test_partial_delivery_probability OK")


# ---------------------------------------------------------------------------
# 13) All actions 0..5
# ---------------------------------------------------------------------------
def test_all_actions():
    print("\n--- test_all_actions ---")
    for action in range(model_init.N_ACTIONS):
        aname = ACTION_NAMES[action]
        print(f"  Step: action={aname} ({action}), uniform parents.")
        parents = _parents()
        print("  Parents:", _summarize_parents(parents))
        out = B_checkboxes(parents, action)
        print("  Output:")
        _print_out("out", out)
        print("  Assert: keys correct and each output sums to 1.")
        assert set(out.keys()) == {"ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"}
        for k, v in out.items():
            assert np.allclose(v.sum(), 1.0)
    print("test_all_actions OK")


# ---------------------------------------------------------------------------
# 14) Normalization after soft update
# ---------------------------------------------------------------------------
def test_normalization_after_soft_update():
    print("\n--- test_normalization_after_soft_update ---")
    print("  Step: pot=[0.2,0.3,0.25,0.25], all checkboxes [0.5,0.5], action=STAY.")
    parents = _parents(
        pot_one_hot=(0.2, 0.3, 0.25, 0.25),
        ck_put1=(0.5, 0.5),
        ck_put2=(0.5, 0.5),
        ck_put3=(0.5, 0.5),
        ck_plated=(0.5, 0.5),
        ck_delivered=(0.5, 0.5),
    )
    print("  Parents:", _summarize_parents(parents))
    out = B_checkboxes(parents, model_init.STAY)
    print("  Output:")
    _print_out("out", out)
    print("  Assert: each output key sums to 1.0 (within 1e-6).")
    for k, v in out.items():
        assert np.allclose(v.sum(), 1.0, atol=1e-6), f"{k} sum"
        print(f"    {k} sum={v.sum():.6f}")
    print("test_normalization_after_soft_update OK")


def run_all():
    print("=" * 60)
    print("IndividuallyCollective tests: D, env_utils, B (normalize, B_self_*, B_pot_state,")
    print("B_checkboxes, B_fn)")
    print("=" * 60)
    # --- D (prior beliefs) ---
    test_build_D_returns_all_state_factors()
    test_build_D_default_priors()
    test_build_D_custom_start_pos()
    test_build_D_custom_start_ori()
    test_build_D_out_of_range_fallback()
    test_D_fn_none_config()
    test_D_fn_with_config()
    test_build_D_shape_match_model_init()
    # --- env_utils ---
    test_env_utils_action_roundtrip()
    test_env_utils_observation_keys()
    test_env_utils_get_D_config_keys()
    test_env_utils_reward_info_soup_delivered()
    test_env_utils_against_env()
    # --- Utility & core B factors ---
    test_normalize()
    test_B_self_pos_stay_interact_no_move()
    test_B_self_pos_north_moves()
    test_B_self_pos_south_moves()
    test_B_self_orientation_directional_sets()
    test_B_self_orientation_stay_keeps()
    test_B_self_held_pick_onion()
    test_B_self_held_pick_dish()
    test_B_self_held_put_onion()
    test_B_self_held_take_soup()
    test_B_self_held_deliver()
    test_B_self_held_non_interact_unchanged()
    test_B_pot_state_put_onion_POT_0_to_1()
    test_B_pot_state_take_soup_POT_3_to_0()
    test_B_pot_state_non_interact_unchanged()
    test_B_fn_returns_all_factors()
    test_B_fn_shape_and_normalization()
    test_B_fn_noise_zero_deterministic()
    test_B_fn_noise_positive_spreads_mass()
    # --- B_checkboxes ---
    test_return_shape_and_normalization()
    test_non_interact_no_delivery_no_plating()
    test_delivery_pulse()
    test_plating_event()
    test_pot_0_put_progress()
    test_pot_1_put_progress()
    test_pot_2_put_progress()
    test_pot_3_put_progress()
    test_monotonic_put1_persists()
    test_at_serve_no_soup_no_delivery()
    test_at_pot_no_dish_no_plating()
    test_at_pot_dish_but_pot_not_ready_no_plating()
    test_partial_delivery_probability()
    test_all_actions()
    test_normalization_after_soft_update()
    print("\nAll D, env_utils, and B.py tests passed.")


if __name__ == "__main__":
    run_all()
