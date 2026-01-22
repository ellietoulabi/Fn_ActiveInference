"""
Q-Learning Agent for Overcooked-AI environment.

Adapts Q-Learning to work with Overcooked's state representation.
"""

import numpy as np
import random
import json
import os
import ast
from overcooked_ai_py.mdp.actions import Action, Direction


class OvercookedQLearningAgent:
    """
    Q-Learning agent for Overcooked-AI.
    
    State representation: Simplified features from OvercookedState
    - Player position (x, y)
    - Player orientation (direction index)
    - Held object type (none, onion, tomato, dish, soup)
    - Nearby objects (simplified)
    """
    
    def __init__(
        self,
        agent_idx,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.1,
        load_existing=True
    ):
        self.agent_idx = agent_idx
        self.action_space_size = len(Action.ALL_ACTIONS)  # 6 actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        
        self.q_table = {}
        if load_existing:
            self.load_q_table()
    
    def get_state(self, overcooked_state):
        """
        Convert OvercookedState to a simplified state representation.
        
        State: (pos_x, pos_y, orientation_idx, held_object_type, 
                other_agent_pos_x, other_agent_pos_y)
        
        This is a simplified representation that captures:
        - My position and orientation
        - What I'm holding
        - Other agent's position (for coordination)
        """
        if overcooked_state is None:
            return None
        
        # Get my player state
        my_player = overcooked_state.players[self.agent_idx]
        pos_x, pos_y = my_player.position
        
        # Orientation as index
        orientation_idx = Direction.DIRECTION_TO_INDEX.get(my_player.orientation, 0)
        
        # Held object type
        if my_player.has_object():
            held_obj = my_player.get_object()
            held_type = self._object_type_to_int(held_obj.name)
        else:
            held_type = 0  # Nothing
        
        # Other agent's position
        other_idx = 1 - self.agent_idx
        other_player = overcooked_state.players[other_idx]
        other_pos_x, other_pos_y = other_player.position
        
        # Create state tuple
        state = (pos_x, pos_y, orientation_idx, held_type, other_pos_x, other_pos_y)
        return state
    
    def _object_type_to_int(self, obj_name):
        """Convert object name to integer code."""
        obj_map = {
            "onion": 1,
            "tomato": 2,
            "dish": 3,
            "soup": 4,
        }
        return obj_map.get(obj_name, 0)
    
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
        
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q
        )
        self.q_table[state][action] = new_q
    
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
                os.makedirs(os.path.dirname(self.q_table_path) if os.path.dirname(self.q_table_path) else '.', exist_ok=True)
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
                print(f"Agent {self.agent_idx}: Loaded Q-table with {len(self.q_table)} states")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error loading Q-table: {e}. Starting with empty Q-table.")
                self.q_table = {}
            except (FileNotFoundError, PermissionError) as e:
                print(f"File access error: {e}. Starting with empty Q-table.")
                self.q_table = {}
    
    def get_stats(self):
        """Get statistics about the Q-table."""
        return {
            "num_states": len(self.q_table),
            "epsilon": self.epsilon,
            "total_q_values": sum(len(q_vals) for q_vals in self.q_table.values()),
        }

