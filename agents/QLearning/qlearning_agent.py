import numpy as np
import random
import json
import os
import ast

class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.985,
        min_epsilon=0.1,
        load_existing=True
    ):
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        
        # Track observed positions for statistics
        self.observed_positions = set()
        
        # Button state labels (matching Active Inference model)
        self.button_state_labels = ["not_pressed", "pressed"]
        self.game_result_labels = ["neutral", "win", "lose"]

        self.q_table = {}
        if load_existing:
            self.load_q_table()

    def get_state(self, obs):
        """
        Convert observation to state representation based purely on observations.
        
        State representation (WITHOUT game_result to allow Q-value transfer):
        (agent_pos, on_red_button, on_blue_button, red_button_state, blue_button_state)
        
        NOTE: We exclude 'game_result' because terminal states (win/lose) should share
        Q-values with their corresponding non-terminal states. Including game_result
        would prevent learned Q-values from transferring between terminal and non-terminal.
        
        Expects observation in the SAME format as Active Inference model:
        - 'agent_pos': int (0-8, flat grid index)
        - 'on_red_button': int (0 or 1)
        - 'on_blue_button': int (0 or 1) 
        - 'red_button_state': int (0 or 1)
        - 'blue_button_state': int (0 or 1)
        """
        if obs is None:
            return None
        
        # Extract observation components (exclude game_result)
        agent_pos = obs["agent_pos"]
        on_red_button = obs.get("on_red_button", 0)
        on_blue_button = obs.get("on_blue_button", 0)
        red_button_state = obs["red_button_state"]
        blue_button_state = obs["blue_button_state"]
        
        # Track observed positions for statistics
        self.observed_positions.add(agent_pos)
        
        # State representation: observations WITHOUT game_result
        return (agent_pos, on_red_button, on_blue_button, 
                red_button_state, blue_button_state)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if state is None or state not in self.q_table:
            return random.randint(0, self.action_space_size - 1)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        return int(np.argmax(self.q_table[state]))

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule."""
        if state is None:
            return
            
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        if next_state is not None and next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size)
        
        # Q-learning update
        current_q = self.q_table[state][action]
        
        if next_state is not None:
            max_next_q = np.max(self.q_table[next_state])
        else:
            max_next_q = 0  # Terminal state
            
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state][action] = new_q
    
    def add_final_state(self, final_state):
        """Add a final state to the Q-table before episode termination."""
        if final_state is not None and final_state not in self.q_table:
            self.q_table[final_state] = np.zeros(self.action_space_size)
            print(f"  Added final state to Q-table: {final_state}")
            return True
        return False

    def decay_exploration(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        """Save Q-table to file."""
        if self.q_table_path:
            serializable_q = {
                str(k): v.tolist() for k, v in self.q_table.items()
            }
            try:
                with open(self.q_table_path, 'w') as file:
                    json.dump(serializable_q, file, indent=2)
            except (IOError, OSError) as e:
                print(f"Error saving Q-table: {e}")

    def load_q_table(self):
        """Load Q-table from file."""
        if self.q_table_path and os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as file:
                    raw = json.load(file)
                    self.q_table = {
                        ast.literal_eval(k): np.array(v) for k, v in raw.items()
                    }
                print(f"Loaded Q-table with {len(self.q_table)} states")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error loading Q-table: {e}. Starting with empty Q-table.")
                self.q_table = {}
            except (FileNotFoundError, PermissionError) as e:
                print(f"File access error: {e}. Starting with empty Q-table.")
                self.q_table = {}

    def get_state_factors(self, state):
        """Decompose state tuple into individual factors."""
        if state is None or len(state) < 5:
            return None
        return {
            "agent_pos": state[0],        # Flat index (0-8)
            "on_red": state[1],           # 0 or 1
            "on_blue": state[2],          # 0 or 1
            "red_state": state[3],        # 0 or 1
            "blue_state": state[4],       # 0 or 1
        }
    
    def state_to_labels(self, state):
        """Convert state to human-readable labels."""
        factors = self.get_state_factors(state)
        if factors is None:
            return None
        
        # Position is a flat index (0-8), convert to (x, y) for label
        agent_pos = factors["agent_pos"]
        x = agent_pos % 3
        y = agent_pos // 3
        pos_label = f"pos_{x}_{y}"
        
        return {
            "agent_pos": pos_label,
            "on_red_button": "on" if factors["on_red"] else "off",
            "on_blue_button": "on" if factors["on_blue"] else "off",
            "red_button_state": self.button_state_labels[factors["red_state"]],
            "blue_button_state": self.button_state_labels[factors["blue_state"]],
        }
    
    def get_stats(self):
        """Get statistics about the Q-table."""
        # Count states by factors
        factor_counts = {
            "positions": set(), 
            "red_states": set(), 
            "blue_states": set(),
            "on_combinations": set(),
        }
        
        for state in self.q_table.keys():
            factors = self.get_state_factors(state)
            if factors:
                factor_counts["positions"].add(factors["agent_pos"])
                factor_counts["red_states"].add(factors["red_state"])
                factor_counts["blue_states"].add(factors["blue_state"])
                factor_counts["on_combinations"].add((factors["on_red"], factors["on_blue"]))
        
        return {
            "num_states": len(self.q_table),
            "epsilon": self.epsilon,
            "total_q_values": sum(len(q_vals) for q_vals in self.q_table.values()),
            "unique_positions": len(factor_counts["positions"]),
            "unique_red_states": len(factor_counts["red_states"]),
            "unique_blue_states": len(factor_counts["blue_states"]),
            "unique_on_combinations": len(factor_counts["on_combinations"]),
            "state_space_coverage": {
                "positions": sorted(list(factor_counts["positions"])),
                "red_states": sorted(list(factor_counts["red_states"])),
                "blue_states": sorted(list(factor_counts["blue_states"])),
                "on_combinations": sorted(list(factor_counts["on_combinations"])),
            }
        }

