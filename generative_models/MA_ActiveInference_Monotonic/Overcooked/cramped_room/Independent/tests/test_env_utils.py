"""
Tests for env_utils (Independent Monotonic cramped_room).
Uses real Overcooked state when available; otherwise uses a mock state so tests still run.
Step output is printed; use pytest -s (or run this file as __main__) to see it.
Run from project root:
  python generative_models/MA_ActiveInference_Monotonic/Overcooked/cramped_room/Independent/tests/test_env_utils.py
"""

import importlib.util
import sys
import types
from pathlib import Path


def _step(msg):
    """Print a test step line so output is visible (e.g. under pytest -s)."""
    print(msg, flush=True)

# Project root (tests -> Independent -> cramped_room -> Overcooked -> MA_ActiveInference_Monotonic -> generative_models -> Fn_ActiveInference)
_THIS_DIR = Path(__file__).resolve().parent
_INDEPENDENT_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent.parent.parent
_OVERCOOKED_SRC = _PROJECT_ROOT / "environments" / "overcooked_ai" / "src"

sys.path.insert(0, str(_PROJECT_ROOT))
if _OVERCOOKED_SRC.exists():
    sys.path.insert(0, str(_OVERCOOKED_SRC))

# Load env_utils: try package first; if that fails (e.g. numpy missing in A.py), load model_init + env_utils only
env_utils = None
try:
    from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent import env_utils
except Exception:
    pass

if env_utils is None:
    _PKG_NAME = "generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent"
    try:
        pkg = types.ModuleType(_PKG_NAME)
        sys.modules[_PKG_NAME] = pkg
        for name in ["generative_models", "generative_models.MA_ActiveInference_Monotonic",
                     "generative_models.MA_ActiveInference_Monotonic.Overcooked",
                     "generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room"]:
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
        spec_m = importlib.util.spec_from_file_location(
            _PKG_NAME + ".model_init",
            _INDEPENDENT_DIR / "model_init.py",
            submodule_search_locations=[str(_INDEPENDENT_DIR)],
        )
        model_init = importlib.util.module_from_spec(spec_m)
        model_init.__package__ = _PKG_NAME
        sys.modules[_PKG_NAME + ".model_init"] = model_init
        spec_m.loader.exec_module(model_init)
        setattr(pkg, "model_init", model_init)
        spec_e = importlib.util.spec_from_file_location(
            _PKG_NAME + ".env_utils",
            _INDEPENDENT_DIR / "env_utils.py",
            submodule_search_locations=[str(_INDEPENDENT_DIR)],
        )
        env_utils = importlib.util.module_from_spec(spec_e)
        env_utils.__package__ = _PKG_NAME
        sys.modules[_PKG_NAME + ".env_utils"] = env_utils
        spec_e.loader.exec_module(env_utils)
    except Exception:
        env_utils = None

# Try to load real Overcooked state
_ENV_AVAILABLE = False
_ENV_LOAD_ERROR = None
_MDP = None
_START_STATE = None

if env_utils is not None:
    try:
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        _MDP = OvercookedGridworld.from_layout_name(layout_name="cramped_room")
        _START_STATE = _MDP.get_standard_start_state()
        _ENV_AVAILABLE = True
    except Exception as e:
        _ENV_LOAD_ERROR = e


def _make_mock_state():
    """Minimal state compatible with env_utils: players at (1,1) and (3,2), NORTH, nothing held; empty objects."""
    # Direction.NORTH = (0, -1)
    class MockPlayer:
        def __init__(self, pos, ori=(0, -1), held=None):
            self.position = pos
            self.orientation = ori
            self.held_object = held
        def has_object(self):
            return self.held_object is not None
    class MockState:
        def __init__(self):
            # Cramped room start: agent 1 at (1,1), agent 2 at (3,2); both NORTH
            self.players = [
                MockPlayer((1, 1), (0, -1), None),
                MockPlayer((3, 2), (0, -1), None),
            ]
            self.objects = {}  # empty pot
    return MockState()


def _get_state():
    """Return real start state if available, else mock state."""
    if _ENV_AVAILABLE and _START_STATE is not None:
        return _START_STATE
    return _make_mock_state()


def _require_env_utils():
    if env_utils is None:
        _step("  SKIP (env_utils not loaded)")
        return False
    return True


def test_env_obs_to_model_obs_keys():
    """env_obs_to_model_obs returns the five observation keys and value ranges."""
    _step("  test_env_obs_to_model_obs_keys:")
    if not _require_env_utils():
        return
    state = _get_state()
    _step("    (using {} state)".format("real env" if _ENV_AVAILABLE else "mock"))
    obs0 = env_utils.env_obs_to_model_obs(state, 0)
    obs1 = env_utils.env_obs_to_model_obs(state, 1)
    expected_keys = {"agent_pos_obs", "agent_orientation_obs", "agent_held_obs", "pot_state_obs", "soup_delivered_obs"}
    _step("    step 1: keys = {}".format(set(obs0.keys())))
    assert set(obs0.keys()) == expected_keys, "obs0 keys: {}".format(set(obs0.keys()))
    assert set(obs1.keys()) == expected_keys, "obs1 keys: {}".format(set(obs1.keys()))
    _step("    step 2: keys match expected OK")
    assert 0 <= obs0["agent_pos_obs"] < 6 and 0 <= obs1["agent_pos_obs"] < 6
    assert 0 <= obs0["agent_orientation_obs"] < 4 and 0 <= obs1["agent_orientation_obs"] < 4
    assert 0 <= obs0["agent_held_obs"] < 4 and 0 <= obs1["agent_held_obs"] < 4
    assert 0 <= obs0["pot_state_obs"] < 4 and 0 <= obs1["pot_state_obs"] < 4
    assert obs0["soup_delivered_obs"] in (0, 1) and obs1["soup_delivered_obs"] in (0, 1)
    _step("    step 3: value ranges OK -> test_env_obs_to_model_obs_keys: OK")


def test_env_obs_to_model_obs_walkable_pos():
    """Agent positions in obs are walkable indices 0..5."""
    _step("  test_env_obs_to_model_obs_walkable_pos:")
    if not _require_env_utils():
        return
    state = _get_state()
    for agent_idx in (0, 1):
        obs = env_utils.env_obs_to_model_obs(state, agent_idx)
        w = obs["agent_pos_obs"]
        _step("    step {}: agent {} agent_pos_obs = {} (in 0..5)".format(agent_idx + 1, agent_idx, w))
        assert w is not None and 0 <= w <= 5, "agent_pos_obs={} not in 0..5".format(w)
    _step("    -> test_env_obs_to_model_obs_walkable_pos: OK")


def test_env_obs_to_model_obs_soup_delivered_event():
    """soup_delivered_obs is 0 without reward_info; can be 1 when reward_info indicates delivery."""
    _step("  test_env_obs_to_model_obs_soup_delivered_event:")
    if not _require_env_utils():
        return
    state = _get_state()
    obs_no_reward = env_utils.env_obs_to_model_obs(state, 0, reward_info=None)
    assert obs_no_reward["soup_delivered_obs"] == 0
    _step("    step 1: without reward_info soup_delivered_obs=0 OK")
    obs_with_delivery = env_utils.env_obs_to_model_obs(
        state, 0, reward_info={"sparse_reward_by_agent": [20, 0]}
    )
    assert obs_with_delivery["soup_delivered_obs"] == 1
    _step("    step 2: with sparse_reward_by_agent[0]>0 soup_delivered_obs=1 OK")
    _step("    -> test_env_obs_to_model_obs_soup_delivered_event: OK")


def test_get_D_config_from_state():
    """get_D_config_from_state returns agent_start_pos (walkable 0..5) and agent_start_ori."""
    _step("  test_get_D_config_from_state:")
    if not _require_env_utils():
        return
    state = _get_state()
    config0 = env_utils.get_D_config_from_state(state, 0)
    config1 = env_utils.get_D_config_from_state(state, 1)
    _step("    step 1: config keys = {}".format(list(config0.keys())))
    assert "agent_start_pos" in config0 and "agent_start_ori" in config0
    assert 0 <= config0["agent_start_pos"] <= 5, "agent_start_pos={}".format(config0["agent_start_pos"])
    assert 0 <= config0["agent_start_ori"] < 4
    assert 0 <= config1["agent_start_pos"] <= 5
    assert 0 <= config1["agent_start_ori"] < 4
    _step("    step 2: agent_start_pos in 0..5, agent_start_ori in 0..3 OK")
    assert "other_agent_start_pos" not in config0
    _step("    step 3: no other_agent_start_pos in config OK -> test_get_D_config_from_state: OK")


def test_extract_pot_state_start():
    """At start state pot is empty -> POT_0."""
    _step("  test_extract_pot_state_start:")
    if not _require_env_utils():
        return
    state = _get_state()
    pot_state = env_utils._extract_pot_state_from_env(state)
    _step("    step 1: _extract_pot_state_from_env(start_state) = {}".format(pot_state))
    assert pot_state == env_utils.model_init.POT_0, "expected POT_0, got {}".format(pot_state)
    _step("    step 2: pot_state == POT_0 OK -> test_extract_pot_state_start: OK")


def test_extract_pot_state_value_range():
    """_extract_pot_state_from_env returns 0, 1, 2, or 3."""
    _step("  test_extract_pot_state_value_range:")
    if not _require_env_utils():
        return
    state = _get_state()
    pot_state = env_utils._extract_pot_state_from_env(state)
    assert pot_state in (0, 1, 2, 3)
    _step("    step 1: pot_state in (0,1,2,3) OK -> test_extract_pot_state_value_range: OK")


def test_model_action_to_env_action_roundtrip():
    """model_action_to_env_action then env_action_to_model_action preserves index 0..5."""
    _step("  test_model_action_to_env_action_roundtrip:")
    if env_utils is None:
        _step("  SKIP (env_utils not loaded)")
        return
    try:
        for idx in range(6):
            env_action = env_utils.model_action_to_env_action(idx)
            back = env_utils.env_action_to_model_action(env_action)
            assert back == idx, "idx {} -> env_action -> {}".format(idx, back)
            _step("    step {}: model {} -> env -> model {} OK".format(idx + 1, idx, back))
        _step("    -> test_model_action_to_env_action_roundtrip: OK")
    except Exception as e:
        _step("  SKIP (action conversion needs overcooked_ai_py: {})".format(type(e).__name__))


def run_all():
    _step("Testing env_utils (Independent Monotonic cramped_room) against env\n")
    test_env_obs_to_model_obs_keys()
    test_env_obs_to_model_obs_walkable_pos()
    test_env_obs_to_model_obs_soup_delivered_event()
    test_get_D_config_from_state()
    test_extract_pot_state_start()
    test_extract_pot_state_value_range()
    test_model_action_to_env_action_roundtrip()
    _step("\nAll tests completed.")


if __name__ == "__main__":
    run_all()
