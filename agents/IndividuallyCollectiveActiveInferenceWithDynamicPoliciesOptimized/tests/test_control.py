import unittest
import numpy as np
import jax.numpy as jnp
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import directly from the control module to avoid __init__.py issues
from agents.ActiveInference.control2 import get_expected_state, get_expected_states, get_state_sizes, build_decode_table, precompute_lnC, calc_expected_utility, log_softmax_np
from agents.ActiveInference.control import get_expected_obs, joint_from_marginals, calc_surprise, calc_states_info_gain, vanilla_fpi_update_posterior_policies


class TestControl(unittest.TestCase):
    """Test cases for the control module functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Environment parameters
        self.env_params = {
            "width": 3,
            "height": 3,
            "open_success": 0.8,
            "noise": 0.05
        }
        
        # Create a sample belief state
        self.sample_qs = {
            "agent_pos": jnp.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]),  # 3x3 grid, agent likely at center
            "red_door_pos": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # red door at bottom-right
            "blue_door_pos": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # blue door at top-left
            "red_door_state": jnp.array([0.8, 0.2]),  # mostly closed
            "blue_door_state": jnp.array([0.9, 0.1]),  # mostly closed
            "goal_context": jnp.array([0.5, 0.5])  # equal probability for both goals
        }
    
    def test_get_expected_state_single_action(self):
        """Test get_expected_state with a single action."""
        action = 0  # UP action
        
        result = get_expected_state(self.sample_qs, action, self.env_params)
        
        # Check that result is a dictionary with expected keys
        expected_keys = ["agent_pos", "red_door_pos", "blue_door_pos", 
                        "red_door_state", "blue_door_state", "goal_context"]
        self.assertEqual(set(result.keys()), set(expected_keys))
        
        # Check that agent position distribution is normalized
        self.assertAlmostEqual(float(jnp.sum(result["agent_pos"])), 1.0, places=5)
        
        # Check that door positions are unchanged (static)
        np.testing.assert_array_equal(result["red_door_pos"], self.sample_qs["red_door_pos"])
        np.testing.assert_array_equal(result["blue_door_pos"], self.sample_qs["blue_door_pos"])
        
        # Check that door states are normalized
        self.assertAlmostEqual(float(jnp.sum(result["red_door_state"])), 1.0, places=5)
        self.assertAlmostEqual(float(jnp.sum(result["blue_door_state"])), 1.0, places=5)
        
        # Check that goal context is unchanged
        np.testing.assert_array_equal(result["goal_context"], self.sample_qs["goal_context"])
    
    def test_get_expected_state_different_actions(self):
        """Test get_expected_state with different actions."""
        actions = [0, 1, 2, 3, 4, 5]  # UP, DOWN, LEFT, RIGHT, OPEN, NOOP
        
        for action in actions:
            with self.subTest(action=action):
                result = get_expected_state(self.sample_qs, action, self.env_params)
                
                # Basic structure checks
                self.assertIsInstance(result, dict)
                self.assertEqual(len(result), 6)
                
                # All distributions should be normalized
                for key, value in result.items():
                    if key in ["agent_pos", "red_door_state", "blue_door_state"]:
                        self.assertAlmostEqual(float(jnp.sum(value)), 1.0, places=5)
    
    def test_get_expected_states_single_action(self):
        """Test get_expected_states with a single action (scalar)."""
        action = 1  # DOWN action
        
        result = get_expected_states(self.sample_qs, action, self.env_params)
        
        # Should return a list with one element
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # The single result should have the same structure as get_expected_state
        expected_keys = ["agent_pos", "red_door_pos", "blue_door_pos", 
                        "red_door_state", "blue_door_state", "goal_context"]
        self.assertEqual(set(result[0].keys()), set(expected_keys))
    
    def test_get_expected_states_policy_sequence(self):
        """Test get_expected_states with a policy sequence."""
        policy = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        
        result = get_expected_states(self.sample_qs, policy, self.env_params)
        
        # Should return a list with 4 elements (one per action)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        
        # Each element should be a valid state dictionary
        for i, state in enumerate(result):
            with self.subTest(step=i):
                self.assertIsInstance(state, dict)
                self.assertEqual(len(state), 6)
                
                # All distributions should be normalized
                for key, value in state.items():
                    if key in ["agent_pos", "red_door_state", "blue_door_state"]:
                        self.assertAlmostEqual(float(jnp.sum(value)), 1.0, places=5)
    
    def test_get_expected_states_numpy_array_policy(self):
        """Test get_expected_states with numpy array policy."""
        policy = np.array([0, 1, 2, 3])
        
        result = get_expected_states(self.sample_qs, policy, self.env_params)
        
        # Should work the same as list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
    
    def test_get_expected_states_empty_policy(self):
        """Test get_expected_states with empty policy."""
        policy = []
        
        result = get_expected_states(self.sample_qs, policy, self.env_params)
        
        # Should return empty list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_agent_movement_consistency(self):
        """Test that agent movement is consistent with action directions."""
        # Start with agent at position 4 (center of 3x3 grid)
        center_qs = self.sample_qs.copy()
        center_qs["agent_pos"] = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
        # Test UP action (should move to position 1)
        up_result = get_expected_state(center_qs, 0, self.env_params)
        # Due to noise, we can't expect exact movement, but should see some probability at position 1
        self.assertGreater(float(up_result["agent_pos"][1]), 0.0)
        
        # Test DOWN action (should move to position 7)
        down_result = get_expected_state(center_qs, 1, self.env_params)
        self.assertGreater(float(down_result["agent_pos"][7]), 0.0)
        
        # Test LEFT action (should move to position 3)
        left_result = get_expected_state(center_qs, 2, self.env_params)
        self.assertGreater(float(left_result["agent_pos"][3]), 0.0)
        
        # Test RIGHT action (should move to position 5)
        right_result = get_expected_state(center_qs, 3, self.env_params)
        self.assertGreater(float(right_result["agent_pos"][5]), 0.0)
    
    def test_door_opening_mechanism(self):
        """Test that door opening works when agent is adjacent and uses OPEN action."""
        # Place agent adjacent to red door (position 5, red door at position 8)
        adjacent_qs = self.sample_qs.copy()
        adjacent_qs["agent_pos"] = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Use OPEN action
        open_result = get_expected_state(adjacent_qs, 4, self.env_params)
        
        # Red door should have some probability of being open
        self.assertGreater(float(open_result["red_door_state"][1]), float(self.sample_qs["red_door_state"][1]))
    
    def test_noop_action(self):
        """Test that NOOP action doesn't change agent position significantly."""
        noop_result = get_expected_state(self.sample_qs, 5, self.env_params)
        
        # Agent position should be very similar to original (only noise difference)
        original_agent_pos = self.sample_qs["agent_pos"]
        new_agent_pos = noop_result["agent_pos"]
        
        # The difference should be small (mostly due to noise)
        diff = jnp.abs(original_agent_pos - new_agent_pos)
        self.assertLess(float(jnp.max(diff)), 0.1)  # Should be small difference
    
    def test_blue_door_dependency(self):
        """Test that blue door opening depends on red door being open."""
        # Create a scenario where agent is adjacent to blue door
        # Blue door at position 0 (top-left), agent at position 1 (top-center) - adjacent
        # Red door at position 8 (bottom-right) - not adjacent to agent
        
        # Case 1: Red door closed
        closed_red_qs = {
            "agent_pos": jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # top-center
            "red_door_pos": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # bottom-right
            "blue_door_pos": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # top-left
            "red_door_state": jnp.array([1.0, 0.0]),  # closed
            "blue_door_state": jnp.array([1.0, 0.0]),  # closed
            "goal_context": jnp.array([0.5, 0.5])
        }
        
        # Case 2: Red door open
        open_red_qs = closed_red_qs.copy()
        open_red_qs["red_door_state"] = jnp.array([0.0, 1.0])  # open
        
        # Try to open blue door in both cases
        blue_open_with_closed_red = get_expected_state(closed_red_qs, 4, self.env_params)
        blue_open_with_open_red = get_expected_state(open_red_qs, 4, self.env_params)
        
        # Blue door should have higher probability of being open when red is open
        # The dependency is multiplicative, so when red is closed (0.0), blue opening is 0
        # When red is open (1.0), blue opening has full probability
        self.assertGreater(float(blue_open_with_open_red["blue_door_state"][1]), 
                          float(blue_open_with_closed_red["blue_door_state"][1]))


    def test_policy_step_by_step_lengths_1_2_3(self):
        """Show step-by-step rollouts for policy lengths 1, 2, 3."""
        for k in [1, 2, 3]:
            policy = list(range(k))  # [0], [0,1], [0,1,2]
            # Print initial state summary
            init = self.sample_qs
            init_agent_argmax = int(jnp.argmax(init["agent_pos"]))
            init_red_pos = int(jnp.argmax(init["red_door_pos"]))
            init_blue_pos = int(jnp.argmax(init["blue_door_pos"]))
            init_red_open = float(init["red_door_state"][1])
            init_blue_open = float(init["blue_door_state"][1])
            print(
                f"\nPolicy length {k} -> actions {policy}\n"
                f" initial_state: agent_pos*={init_agent_argmax}, "
                f"red_pos*={init_red_pos}, blue_pos*={init_blue_pos}, "
                f"red_open_p={init_red_open:.3f}, blue_open_p={init_blue_open:.3f}"
            )

            # Roll forward step by step using get_expected_state to show input -> output
            qs_t = init
            qs_rollout = []
            for t, action in enumerate(policy):
                in_agent_argmax = int(jnp.argmax(qs_t["agent_pos"]))
                in_red_open = float(qs_t["red_door_state"][1])
                in_blue_open = float(qs_t["blue_door_state"][1])

                qs_next = get_expected_state(qs_t, action, self.env_params)

                out_agent_argmax = int(jnp.argmax(qs_next["agent_pos"]))
                out_red_open = float(qs_next["red_door_state"][1])
                out_blue_open = float(qs_next["blue_door_state"][1])

                print(
                    f" step {t}: action={action}"
                    f" | input: agent_pos*={in_agent_argmax}, red_open_p={in_red_open:.3f}, blue_open_p={in_blue_open:.3f}"
                    f" | output: agent_pos*={out_agent_argmax}, red_open_p={out_red_open:.3f}, blue_open_p={out_blue_open:.3f}"
                )

                qs_rollout.append(qs_next)
                qs_t = qs_next

            # Basic assertions
            self.assertIsInstance(qs_rollout, list)
            self.assertEqual(len(qs_rollout), k)
            for qs in qs_rollout:
                self.assertAlmostEqual(float(jnp.sum(qs["agent_pos"])), 1.0, places=5)
                self.assertAlmostEqual(float(jnp.sum(qs["red_door_state"])), 1.0, places=5)
                self.assertAlmostEqual(float(jnp.sum(qs["blue_door_state"])), 1.0, places=5)


    def test_get_state_sizes(self):
        """Test get_state_sizes function."""
        # Test with 3x3 grid
        sizes = get_state_sizes(3, 3)
        expected = [9, 9, 9, 2, 2, 2]  # agent, red, blue, red_state, blue_state, goal
        self.assertEqual(sizes, expected)
        
        # Test with 2x2 grid
        sizes = get_state_sizes(2, 2)
        expected = [4, 4, 4, 2, 2, 2]
        self.assertEqual(sizes, expected)
        
        # Test with 4x4 grid
        sizes = get_state_sizes(4, 4)
        expected = [16, 16, 16, 2, 2, 2]
        self.assertEqual(sizes, expected)
    
    def test_build_decode_table(self):
        """Test build_decode_table function."""
        # Test with 2x2 grid (smaller for easier verification)
        decode_table = build_decode_table(2, 2)
        
        # Check shape
        self.assertEqual(decode_table.shape, (512, 6))  # 4*4*4*2*2*2 = 512 states, 6 factors
        
        # Check that all indices are valid
        for i in range(decode_table.shape[0]):
            for j in range(decode_table.shape[1]):
                self.assertGreaterEqual(decode_table[i, j], 0)
        
        # Check specific entries
        # Index 0 should decode to [0, 0, 0, 0, 0, 0]
        self.assertTrue(jnp.array_equal(decode_table[0], jnp.array([0, 0, 0, 0, 0, 0])))
        
        # Index 1 should decode to [0, 0, 0, 0, 0, 1] (goal=1)
        self.assertTrue(jnp.array_equal(decode_table[1], jnp.array([0, 0, 0, 0, 0, 1])))
        
        # Index 2 should decode to [0, 0, 0, 0, 1, 0] (blue_state=1)
        self.assertTrue(jnp.array_equal(decode_table[2], jnp.array([0, 0, 0, 0, 1, 0])))
        
        # Index 4 should decode to [0, 0, 0, 1, 0, 0] (red_state=1)
        self.assertTrue(jnp.array_equal(decode_table[4], jnp.array([0, 0, 0, 1, 0, 0])))
        
        # Index 8 should decode to [0, 0, 1, 0, 0, 0] (blue_pos=1)
        self.assertTrue(jnp.array_equal(decode_table[8], jnp.array([0, 0, 1, 0, 0, 0])))
        
        # Index 32 should decode to [0, 1, 0, 0, 0, 0] (red_pos=1)
        self.assertTrue(jnp.array_equal(decode_table[32], jnp.array([0, 1, 0, 0, 0, 0])))
        
        # Index 128 should decode to [1, 0, 0, 0, 0, 0] (agent_pos=1)
        self.assertTrue(jnp.array_equal(decode_table[128], jnp.array([1, 0, 0, 0, 0, 0])))
    
    def test_build_decode_table_3x3(self):
        """Test build_decode_table with 3x3 grid."""
        decode_table = build_decode_table(3, 3)
        
        # Check shape
        self.assertEqual(decode_table.shape, (5832, 6))  # 9*9*9*2*2*2 = 5832 states, 6 factors
        
        # Check that all indices are valid
        for i in range(decode_table.shape[0]):
            for j in range(decode_table.shape[1]):
                self.assertGreaterEqual(decode_table[i, j], 0)
        
        # Check specific entries
        # Index 0 should decode to [0, 0, 0, 0, 0, 0]
        self.assertTrue(jnp.array_equal(decode_table[0], jnp.array([0, 0, 0, 0, 0, 0])))
        
        # Index 1 should decode to [0, 0, 0, 0, 0, 1] (goal=1)
        self.assertTrue(jnp.array_equal(decode_table[1], jnp.array([0, 0, 0, 0, 0, 1])))
    
    def test_get_expected_obs_new_signature(self):
        """Test the new get_expected_obs function with joint state beliefs."""
        # Create a simple belief over joint states (3x3 grid)
        S = 9 * 9 * 9 * 2 * 2 * 2  # Total joint states
        q_state = jnp.zeros(S)
        
        # Set some probability mass on specific states
        # State where agent at pos 4, red at pos 8, blue at pos 0, both doors closed, goal=0
        # This corresponds to index 3168 in the decode table
        state_idx = 3168
        q_state = q_state.at[state_idx].set(0.8)
        
        # State where agent at pos 4, red at pos 8, blue at pos 0, red open, blue closed, goal=0
        # This corresponds to index 3172 in the decode table
        state_idx2 = 3172
        q_state = q_state.at[state_idx2].set(0.2)
        
        # Normalize
        q_state = q_state / jnp.sum(q_state)
        
        # Create decode table
        decode_table = build_decode_table(3, 3)
        
        # Create mock A functions that work with state tuples
        def A_agent_pos(state_tuple, width, height):
            agent_pos = state_tuple[0]
            result = jnp.zeros(width * height)
            return result.at[agent_pos].set(1.0)
        
        def A_on_red_button(state_tuple, width, height):
            agent_pos, red_pos = state_tuple[0], state_tuple[1]
            return jnp.where(agent_pos != red_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_on_blue_button(state_tuple, width, height):
            agent_pos, blue_pos = state_tuple[0], state_tuple[2]
            return jnp.where(agent_pos != blue_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_on_blue_button(state_tuple, width, height):
            agent_pos, blue_pos = state_tuple[0], state_tuple[2]
            return jnp.where(agent_pos != blue_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))

        def A_on_blue_button(state_tuple, width, height):
            agent_pos, blue_pos = state_tuple[0], state_tuple[2]
            return jnp.where(agent_pos != blue_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_red_button_state(state_tuple, width, height):
            red_state = state_tuple[3]
            return jnp.where(red_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_blue_button_state(state_tuple, width, height):
            blue_state = state_tuple[4]
            return jnp.where(blue_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_game_result(state_tuple, width, height):
            red_state = state_tuple[3]
            blue_state = state_tuple[4]
            goal_ctx = state_tuple[5]
            both_pressed = jnp.logical_and(red_state == 1, blue_state == 1)
            neutral = jnp.array([1.0, 0.0, 0.0])
            win = jnp.array([0.0, 1.0, 0.0])
            lose = jnp.array([0.0, 0.0, 1.0])
            when_pressed = jnp.where(goal_ctx == 0, win, lose)
            return jnp.where(both_pressed, when_pressed, neutral)
        
        def A_blue_button_state(state_tuple, width, height):
            blue_state = state_tuple[4]
            return jnp.where(blue_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        def A_game_result(state_tuple, width, height):
            red_state = state_tuple[3]
            blue_state = state_tuple[4]
            goal_ctx = state_tuple[5]
            both_pressed = jnp.logical_and(red_state == 1, blue_state == 1)
            neutral = jnp.array([1.0, 0.0, 0.0])
            win = jnp.array([0.0, 1.0, 0.0])
            lose = jnp.array([0.0, 0.0, 1.0])
            when_pressed = jnp.where(goal_ctx == 0, win, lose)
            return jnp.where(both_pressed, when_pressed, neutral)

        def A_blue_button_state(state_tuple, width, height):
            blue_state = state_tuple[4]
            return jnp.where(blue_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))

        def A_game_result(state_tuple, width, height):
            red_state = state_tuple[3]
            blue_state = state_tuple[4]
            goal_ctx = state_tuple[5]
            both_pressed = jnp.logical_and(red_state == 1, blue_state == 1)
            # neutral=[1,0,0], win=[0,1,0], lose=[0,0,1]; goal 0 => win if both pressed, goal 1 => lose if both pressed
            neutral = jnp.array([1.0, 0.0, 0.0])
            win = jnp.array([0.0, 1.0, 0.0])
            lose = jnp.array([0.0, 0.0, 1.0])
            when_pressed = jnp.where(goal_ctx == 0, win, lose)
            return jnp.where(both_pressed, when_pressed, neutral)
        
        A_funcs = {
            "agent_pos": A_agent_pos,
            "on_red_button": A_on_red_button,
            "on_blue_button": A_on_blue_button,
            "red_button_state": A_red_button_state,
            "blue_button_state": A_blue_button_state,
            "game_result": A_game_result,
        }
        
        # Test the function
        result = get_expected_obs(q_state, A_funcs, decode_table, 3, 3)
        
        # Check structure
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(A_funcs.keys()))
        
        # Check normalization
        for modality, obs_dist in result.items():
            self.assertAlmostEqual(float(jnp.sum(obs_dist)), 1.0, places=5)
            self.assertTrue(jnp.all(obs_dist >= 0.0))
        
        # Check specific values
        # Agent should be at position 4 (both states have agent at 4)
        agent_obs = result["agent_pos"]
        self.assertAlmostEqual(float(agent_obs[4]), 1.0, places=5)
        
        # On red button should be False (agent at 4, red at 8)
        on_red_obs = result["on_red_button"]
        self.assertAlmostEqual(float(on_red_obs[0]), 1.0, places=5)  # 100% False (agent never on red)
        self.assertAlmostEqual(float(on_red_obs[1]), 0.0, places=5)  # 0% True
        
        # Red button state should be mixed (80% closed, 20% open)
        red_state_obs = result["red_button_state"]
        self.assertAlmostEqual(float(red_state_obs[0]), 0.8, places=5)  # 80% closed
        self.assertAlmostEqual(float(red_state_obs[1]), 0.2, places=5)  # 20% open
    
    def test_get_expected_obs_detailed_new_signature(self):
        """Show policy, initial beliefs under policy, and get_expected_obs output step by step."""
        # Define policy
        policy = [0, 1, 2]  # UP, DOWN, LEFT
        print(f"\n=== POLICY ===")
        print(f"Policy: {policy}")
        print(f"Actions: UP, DOWN, LEFT")
        
        # Get beliefs under policy using get_expected_states
        print(f"\n=== BELIEFS UNDER POLICY ===")
        qs_pi = get_expected_states(self.sample_qs, policy, self.env_params)
        
        # Print initial beliefs
        print(f"Initial beliefs:")
        init_qs = self.sample_qs
        print(f"  agent_pos*={int(jnp.argmax(init_qs['agent_pos']))} (belief: {[f'{float(x):.3f}' for x in init_qs['agent_pos']]})")
        print(f"  red_door_pos*={int(jnp.argmax(init_qs['red_door_pos']))} (belief: {[f'{float(x):.3f}' for x in init_qs['red_door_pos']]})")
        print(f"  blue_door_pos*={int(jnp.argmax(init_qs['blue_door_pos']))} (belief: {[f'{float(x):.3f}' for x in init_qs['blue_door_pos']]})")
        print(f"  red_door_state*={int(jnp.argmax(init_qs['red_door_state']))} (belief: {[f'{float(x):.3f}' for x in init_qs['red_door_state']]})")
        print(f"  blue_door_state*={int(jnp.argmax(init_qs['blue_door_state']))} (belief: {[f'{float(x):.3f}' for x in init_qs['blue_door_state']]})")
        print(f"  goal_context*={int(jnp.argmax(init_qs['goal_context']))} (belief: {[f'{float(x):.3f}' for x in init_qs['goal_context']]})")
        
        # Print beliefs at each step
        for t, qs_t in enumerate(qs_pi):
            print(f"\nStep {t} beliefs (after action {policy[t]}):")
            print(f"  agent_pos*={int(jnp.argmax(qs_t['agent_pos']))} (belief: {[f'{float(x):.3f}' for x in qs_t['agent_pos']]})")
            print(f"  red_door_pos*={int(jnp.argmax(qs_t['red_door_pos']))} (belief: {[f'{float(x):.3f}' for x in qs_t['red_door_pos']]})")
            print(f"  blue_door_pos*={int(jnp.argmax(qs_t['blue_door_pos']))} (belief: {[f'{float(x):.3f}' for x in qs_t['blue_door_pos']]})")
            print(f"  red_door_state*={int(jnp.argmax(qs_t['red_door_state']))} (belief: {[f'{float(x):.3f}' for x in qs_t['red_door_state']]})")
            print(f"  blue_door_state*={int(jnp.argmax(qs_t['blue_door_state']))} (belief: {[f'{float(x):.3f}' for x in qs_t['blue_door_state']]})")
            print(f"  goal_context*={int(jnp.argmax(qs_t['goal_context']))} (belief: {[f'{float(x):.3f}' for x in qs_t['goal_context']]})")
        
        # Convert factorized beliefs to joint state beliefs (simplified)
        S = 9 * 9 * 9 * 2 * 2 * 2
        joint_qs_pi = []
        # Strides consistent with decode_table order [agent, red, blue, red_state, blue_state, goal]
        strides = [648, 72, 8, 4, 2, 1]
        
        for qs_t in qs_pi:
            # Create a joint state belief (simplified - just put mass on most likely state)
            q_state = jnp.zeros(S)
            
            # Find most likely state
            agent_pos = int(jnp.argmax(qs_t["agent_pos"]))
            red_pos = int(jnp.argmax(qs_t["red_door_pos"]))
            blue_pos = int(jnp.argmax(qs_t["blue_door_pos"]))
            red_state = int(jnp.argmax(qs_t["red_door_state"]))
            blue_state = int(jnp.argmax(qs_t["blue_door_state"]))
            goal = int(jnp.argmax(qs_t["goal_context"]))
            
            # Encode state using correct mixed-radix strides
            state_idx = (agent_pos * strides[0]
                         + red_pos * strides[1]
                         + blue_pos * strides[2]
                         + red_state * strides[3]
                         + blue_state * strides[4]
                         + goal * strides[5])
            q_state = q_state.at[state_idx].set(1.0)
            joint_qs_pi.append(q_state)
        
        # Create decode table
        decode_table = build_decode_table(3, 3)
        
        # Import noisy A functions
        from generative_models.SA_ActiveInference.RedBlueButton.A_noisy import A_funcs_noisy
        
        # Create wrapper functions that match the expected signature
        def A_agent_pos(state_tuple, width, height):
            S = width * height
            return A_funcs_noisy["agent_pos"](state_tuple, S)
        
        def A_on_red_button(state_tuple, width, height):
            S = 2  # binary
            return A_funcs_noisy["on_red_button"](state_tuple, S)
        
        def A_on_blue_button(state_tuple, width, height):
            S = 2  # binary
            return A_funcs_noisy["on_blue_button"](state_tuple, S)
        
        def A_red_button_state(state_tuple, width, height):
            S = 2  # binary
            return A_funcs_noisy["red_button_state"](state_tuple, S)
        
        def A_blue_button_state(state_tuple, width, height):
            S = 2  # binary
            return A_funcs_noisy["blue_button_state"](state_tuple, S)
        
        def A_game_result(state_tuple, width, height):
            S = 3  # ternary
            return A_funcs_noisy["game_result"](state_tuple, S)
        
        A_funcs = {
            "agent_pos": A_agent_pos,
            "on_red_button": A_on_red_button,
            "on_blue_button": A_on_blue_button,
            "red_button_state": A_red_button_state,
            "blue_button_state": A_blue_button_state,
            "game_result": A_game_result,
        }
        
        # Get expected observations
        print(f"\n=== GET_EXPECTED_OBS OUTPUT ===")
        qo_pi = []
        for t, q_state in enumerate(joint_qs_pi):
            qo_t = get_expected_obs(q_state, A_funcs, decode_table, 3, 3)
            qo_pi.append(qo_t)
            
            print(f"\nStep {t} observations:")
            print(f"  agent_pos: {[f'{float(x):.3f}' for x in qo_t['agent_pos']]} (argmax={int(jnp.argmax(qo_t['agent_pos']))})")
            print(f"  on_red_button: {[f'{float(x):.3f}' for x in qo_t['on_red_button']]} (p(TRUE)={float(qo_t['on_red_button'][1]):.3f})")
            print(f"  on_blue_button: {[f'{float(x):.3f}' for x in qo_t['on_blue_button']]} (p(TRUE)={float(qo_t['on_blue_button'][1]):.3f})")
            print(f"  red_button_state: {[f'{float(x):.3f}' for x in qo_t['red_button_state']]} (p(open)={float(qo_t['red_button_state'][1]):.3f})")
            print(f"  blue_button_state: {[f'{float(x):.3f}' for x in qo_t['blue_button_state']]} (p(open)={float(qo_t['blue_button_state'][1]):.3f})")
            print(f"  game_result: {[f'{float(x):.3f}' for x in qo_t['game_result']]} (neutral/win/lose)")
        
        # Basic assertions
        self.assertIsInstance(qo_pi, list)
        self.assertEqual(len(qo_pi), 3)
        
        for qo_t in qo_pi:
            for modality, obs_dist in qo_t.items():
                self.assertAlmostEqual(float(jnp.sum(obs_dist)), 1.0, places=5)

    def test_precompute_lnC(self):
        """Test precompute_lnC function."""
        from agents.ActiveInference.control import precompute_lnC
        
        # Create mock preference functions
        def C_agent_pos():
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        def C_on_red_button():
            return np.array([0.3, 0.7])  # Prefer being on red button
        
        def C_game_result():
            return np.array([0.1, 0.8, 0.1])  # Prefer winning
        
        C_funcs = {
            "agent_pos": C_agent_pos,
            "on_red_button": C_on_red_button,
            "game_result": C_game_result
        }
        
        lnC = precompute_lnC(C_funcs)
        
        # Check that all modalities are present
        self.assertEqual(set(lnC.keys()), set(C_funcs.keys()))
        
        # Check that log-softmax is applied (sums to 0 in log space)
        for modality, lnC_vec in lnC.items():
            self.assertAlmostEqual(float(np.sum(np.exp(lnC_vec))), 1.0, places=5)
            self.assertTrue(np.allclose(np.log(np.sum(np.exp(lnC_vec))), 0.0, atol=1e-10))
        
        # Check that preferences are preserved (higher values -> higher log-prob)
        agent_lnC = lnC["agent_pos"]
        self.assertTrue(agent_lnC[0] < agent_lnC[-1])  # First element < last element
        
        red_lnC = lnC["on_red_button"]
        self.assertTrue(red_lnC[0] < red_lnC[1])  # Prefer being on red button

    def test_calc_expected_utility(self):
        """Test calc_expected_utility function."""
        from agents.ActiveInference.control import calc_expected_utility
        
        # Create mock observation sequence
        qo_pi = [
            {
                "agent_pos": np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "on_red_button": np.array([0.7, 0.3]),
                "game_result": np.array([0.8, 0.1, 0.1])
            },
            {
                "agent_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0]),
                "on_red_button": np.array([0.6, 0.4]),
                "game_result": np.array([0.7, 0.2, 0.1])
            }
        ]
        
        # Create mock log-preferences
        lnC = {
            "agent_pos": np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])),  # Prefer position 8
            "on_red_button": np.log(np.array([0.3, 0.7])),  # Prefer being on red button
            "game_result": np.log(np.array([0.1, 0.8, 0.1]))  # Prefer winning
        }
        
        expected_util = calc_expected_utility(qo_pi, lnC)
        
        # Should be a finite number
        self.assertTrue(np.isfinite(expected_util))
        
        # Test with 1D preferences
        lnC_1d = {
            "agent_pos": np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]))
        }
        qo_pi_1d = [{"agent_pos": qo_pi[0]["agent_pos"]}]
        util_1d = calc_expected_utility(qo_pi_1d, lnC_1d)
        self.assertTrue(np.isfinite(util_1d))

    def test_calc_surprise(self):
        """Test calc_surprise function."""
        from agents.ActiveInference.control import calc_surprise
        
        # Create mock A functions
        def A_agent_pos(state_tuple, width, height):
            agent_pos = state_tuple[0]
            result = jnp.zeros(width * height)
            return result.at[agent_pos].set(1.0)
        
        def A_on_red_button(state_tuple, width, height):
            agent_pos, red_pos = state_tuple[0], state_tuple[1]
            return jnp.where(agent_pos != red_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        
        A_funcs = {
            "agent_pos": A_agent_pos,
            "on_red_button": A_on_red_button
        }
        
        # Create mock belief state (agent at position 4)
        qs_t = jnp.zeros(9)
        qs_t = qs_t.at[4].set(1.0)
        
        # Create decode table
        decode_table = build_decode_table(3, 3)
        
        surprise = calc_surprise(A_funcs, qs_t, decode_table, 3, 3)
        
        # Should be a finite number
        self.assertTrue(np.isfinite(surprise))
        self.assertGreaterEqual(surprise, 0.0)  # Surprise should be non-negative

    def test_calc_states_info_gain(self):
        """Test calc_states_info_gain function."""
        from agents.ActiveInference.control import calc_states_info_gain
        
        # Create mock A functions
        def A_agent_pos(state_tuple, width, height):
            agent_pos = state_tuple[0]
            result = jnp.zeros(width * height)
            return result.at[agent_pos].set(1.0)
        
        A_funcs = {"agent_pos": A_agent_pos}
        
        # Create mock belief states
        qs_pi = [
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0])
        ]
        
        decode_table = build_decode_table(3, 3)
        
        info_gain = calc_states_info_gain(A_funcs, qs_pi, decode_table, 3, 3)
        
        # Should be a finite number
        self.assertTrue(np.isfinite(info_gain))
        self.assertGreaterEqual(info_gain, 0.0)  # Info gain should be non-negative

    def test_vanilla_fpi_update_posterior_policies(self):
        """Test vanilla_fpi_update_posterior_policies function."""
        from agents.ActiveInference.control import vanilla_fpi_update_posterior_policies
        
        # Import real functions
        from generative_models.SA_ActiveInference.RedBlueButton.B import apply_B
        from generative_models.SA_ActiveInference.RedBlueButton.A_noisy import A_funcs_noisy
        from generative_models.SA_ActiveInference.RedBlueButton.C import make_C_prefs
        
        # Create environment
        env_params = {
            "width": 3,
            "height": 3,
            "B": apply_B
        }
        
        # Use real A functions with wrapper for signature
        def A_agent_pos(state_tuple, width, height):
            S = width * height
            return A_funcs_noisy["agent_pos"](state_tuple, S)
        
        def A_on_red_button(state_tuple, width, height):
            S = 2
            return A_funcs_noisy["on_red_button"](state_tuple, S)
        
        A_funcs = {
            "agent_pos": A_agent_pos,
            "on_red_button": A_on_red_button
        }
        
        # Use real B function
        B = apply_B
        
        # Create mock C functions that return preference vectors (compatible with precompute_lnC)
        def C_agent_pos():
            return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])  # Prefer position 8
        
        def C_on_red_button():
            return np.array([0.3, 0.7])  # Prefer being on red button
        
        C_funcs = {
            "agent_pos": C_agent_pos,
            "on_red_button": C_on_red_button
        }
        
        # Create mock policies
        policies = [
            [0, 1],  # UP, DOWN
            [1, 0],  # DOWN, UP
            [2, 3]   # LEFT, RIGHT
        ]
        
        # Create decode table
        decode_table = build_decode_table(3, 3)
        
        # Convert factorized beliefs to joint state beliefs for get_expected_obs
        def factorized_to_joint(qs_dict):
            S = 9 * 9 * 9 * 2 * 2 * 2
            strides = [648, 72, 8, 4, 2, 1]
            q_state = jnp.zeros(S)
            
            # Find most likely state
            agent_pos = int(jnp.argmax(qs_dict["agent_pos"]))
            red_pos = int(jnp.argmax(qs_dict["red_door_pos"]))
            blue_pos = int(jnp.argmax(qs_dict["blue_door_pos"]))
            red_state = int(jnp.argmax(qs_dict["red_door_state"]))
            blue_state = int(jnp.argmax(qs_dict["blue_door_state"]))
            goal = int(jnp.argmax(qs_dict["goal_context"]))
            
            # Encode state
            state_idx = (agent_pos * strides[0] + red_pos * strides[1] + blue_pos * strides[2] + 
                        red_state * strides[3] + blue_state * strides[4] + goal * strides[5])
            q_state = q_state.at[state_idx].set(1.0)
            return q_state
        
        # Test with utility and info gain
        q_pi, G = vanilla_fpi_update_posterior_policies(
            self.sample_qs, A_funcs, B, C_funcs, policies, 
            decode_table, env_params, use_utility=True, use_states_info_gain=True
        )
        
        # Check that posterior is valid probability distribution
        self.assertAlmostEqual(float(np.sum(q_pi)), 1.0, places=5)
        self.assertTrue(np.all(q_pi >= 0.0))
        self.assertEqual(len(q_pi), len(policies))
        
        # Check that G values are finite
        self.assertTrue(np.all(np.isfinite(G)))
        
        # Test with only utility
        q_pi_util, G_util = vanilla_fpi_update_posterior_policies(
            self.sample_qs, A_funcs, B, C_funcs, policies,
            decode_table, env_params, use_utility=True, use_states_info_gain=False
        )
        
        self.assertAlmostEqual(float(np.sum(q_pi_util)), 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(G_util)))
        
        # Test with only info gain
        q_pi_info, G_info = vanilla_fpi_update_posterior_policies(
            self.sample_qs, A_funcs, B, C_funcs, policies,
            decode_table, env_params, use_utility=False, use_states_info_gain=True
        )
        
        self.assertAlmostEqual(float(np.sum(q_pi_info)), 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(G_info)))

    def test_policy_evaluation_integration(self):
        """Test complete policy evaluation pipeline with manual conversion."""
        from agents.ActiveInference.control2 import get_expected_states
        from agents.ActiveInference.control import get_expected_obs, precompute_lnC, calc_expected_utility, calc_states_info_gain
        
        print(f"\n=== POLICY EVALUATION INTEGRATION TEST ===")
        
        # Setup
        from generative_models.SA_ActiveInference.RedBlueButton.B import apply_B
        from generative_models.SA_ActiveInference.RedBlueButton.A_noisy import A_funcs_noisy
        from generative_models.SA_ActiveInference.RedBlueButton.C import make_C_prefs
        
        env_params = {
            "width": 3,
            "height": 3,
            "B": apply_B
        }
        
        # Create A functions using noisy versions
        def A_agent_pos(state_tuple, width, height):
            S = width * height
            return A_funcs_noisy["agent_pos"](state_tuple, S)
        
        def A_on_red_button(state_tuple, width, height):
            S = 2
            return A_funcs_noisy["on_red_button"](state_tuple, S)
        
        A_funcs = {
            "agent_pos": A_agent_pos,
            "on_red_button": A_on_red_button
        }
        
        # Create mock C functions that return preference vectors
        def C_agent_pos():
            return np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1])  # Prefer center
        
        def C_on_red_button():
            return np.array([0.2, 0.8])  # Prefer being on red button
        
        C_funcs = {
            "agent_pos": C_agent_pos,
            "on_red_button": C_on_red_button
        }
        
        # Create policies
        policies = [
            [0, 1, 2],  # UP, DOWN, LEFT
            [1, 0, 3],  # DOWN, UP, RIGHT
            [2, 3, 0],  # LEFT, RIGHT, UP
            [3, 2, 1],  # RIGHT, LEFT, DOWN
        ]
        
        decode_table = build_decode_table(3, 3)
        
        # Convert factorized beliefs to joint state beliefs
        def factorized_to_joint(qs_dict):
            S = 9 * 9 * 9 * 2 * 2 * 2
            strides = [648, 72, 8, 4, 2, 1]
            q_state = jnp.zeros(S)
            
            # Find most likely state
            agent_pos = int(jnp.argmax(qs_dict["agent_pos"]))
            red_pos = int(jnp.argmax(qs_dict["red_door_pos"]))
            blue_pos = int(jnp.argmax(qs_dict["blue_door_pos"]))
            red_state = int(jnp.argmax(qs_dict["red_door_state"]))
            blue_state = int(jnp.argmax(qs_dict["blue_door_state"]))
            goal = int(jnp.argmax(qs_dict["goal_context"]))
            
            # Encode state
            state_idx = (agent_pos * strides[0] + red_pos * strides[1] + blue_pos * strides[2] + 
                        red_state * strides[3] + blue_state * strides[4] + goal * strides[5])
            q_state = q_state.at[state_idx].set(1.0)
            return q_state
        
        # Evaluate each policy manually
        policy_utilities = []
        policy_posteriors = []
        
        for policy in policies:
            print(f"\nEvaluating policy: {policy}")
            
            # Get expected states
            qs_pi = get_expected_states(self.sample_qs, policy, env_params)
            print(f"  States: {len(qs_pi)} steps")
            
            # Convert to joint states and get observations
            qo_pi = []
            for qs_t in qs_pi:
                q_state = factorized_to_joint(qs_t)
                qo_t = get_expected_obs(q_state, A_funcs, decode_table, 3, 3)
                qo_pi.append(qo_t)
            
            print(f"  Observations: {len(qo_pi)} steps")
            
            # Calculate utility
            lnC = precompute_lnC(C_funcs)
            utility = calc_expected_utility(qo_pi, lnC)
            print(f"  Utility: {utility:.3f}")
            
            # Calculate info gain
            joint_qs_pi = [factorized_to_joint(qs_t) for qs_t in qs_pi]
            info_gain = calc_states_info_gain(A_funcs, joint_qs_pi, decode_table, 3, 3)
            print(f"  Info gain: {info_gain:.3f}")
            
            total_value = utility + info_gain
            policy_utilities.append(total_value)
            print(f"  Total value: {total_value:.3f}")
        
        # Convert to probabilities
        G = np.array(policy_utilities)
        q_pi = np.exp(G)
        q_pi /= np.sum(q_pi)
        
        print(f"\nPolicy utilities (G): {[f'{g:.3f}' for g in G]}")
        print(f"Policy posteriors (q_pi): {[f'{p:.3f}' for p in q_pi]}")
        print(f"Best policy: {policies[np.argmax(q_pi)]} (posterior={q_pi[np.argmax(q_pi)]:.3f})")
        
        # Verify results
        self.assertAlmostEqual(float(np.sum(q_pi)), 1.0, places=5)
        self.assertTrue(np.all(q_pi >= 0.0))
        self.assertTrue(np.all(np.isfinite(G)))


if __name__ == '__main__':
    unittest.main()
