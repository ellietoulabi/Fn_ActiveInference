import numpy as np
import json
from pathlib import Path
from agents.QLearning.dynaq_agent import DynaQAgent


class TabularDynaQAgent(DynaQAgent):
    """
    Tabular Dyna-Q Agent with model persistence.
    
    Extends DynaQAgent with the ability to save and load the world model.
    This allows the agent to:
    - Load a pre-trained world model from disk
    - Save the world model for future use
    - Continue learning from a previously built model
    
    The model stores: model[state][action] = (next_state, reward, terminated)
    
    Inherits all DynaQ features:
    - Direct RL (Q-learning)
    - Model Learning
    - Planning steps
    """
    
    def __init__(
        self,
        action_space_size,
        planning_steps=10,
        q_table_path=None,
        model_path=None,  # NEW: Path to save/load world model
        learning_rate=0.3,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.1,
        load_existing_q_table=False,
        load_existing_model=False  # NEW: Whether to load existing model
    ):
        # Initialize parent DynaQAgent
        super().__init__(
            action_space_size=action_space_size,
            planning_steps=planning_steps,
            q_table_path=q_table_path,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            load_existing=load_existing_q_table
        )
        
        # Model persistence
        self.model_path = model_path
        
        # Load existing model if requested
        if load_existing_model and model_path:
            self.load_model()
    
    def save_model(self, path=None):
        """
        Save the world model to a JSON file.
        
        Args:
            path: Optional path to save to (overrides self.model_path)
        """
        save_path = path or self.model_path
        
        if not save_path:
            print("Warning: No model path specified, cannot save model")
            return
        
        # Convert model to JSON-serializable format
        # model[state][action] = (next_state, reward, terminated)
        serializable_model = {}
        
        for state, actions in self.model.items():
            # Convert state tuple to string key
            state_key = str(state)
            serializable_model[state_key] = {}
            
            for action, (next_state, reward, terminated) in actions.items():
                # Convert next_state to string, handle None
                next_state_str = str(next_state) if next_state is not None else "None"
                serializable_model[state_key][str(action)] = {
                    'next_state': next_state_str,
                    'reward': float(reward),
                    'terminated': bool(terminated)
                }
        
        # Convert visited_state_actions list to serializable format
        visited_sa = [
            (str(state), int(action)) 
            for state, action in self.visited_state_actions
        ]
        
        # Create the save data
        save_data = {
            'model': serializable_model,
            'visited_state_actions': visited_sa,
            'model_size': len(self.model),
            'total_transitions': sum(len(actions) for actions in self.model.values()),
            'visited_sa_pairs': len(self.visited_state_actions)
        }
        
        # Save to file
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Model saved to {save_path}")
        print(f"  States: {save_data['model_size']}")
        print(f"  Transitions: {save_data['total_transitions']}")
        print(f"  (state, action) pairs: {save_data['visited_sa_pairs']}")
    
    def load_model(self, path=None):
        """
        Load a world model from a JSON file.
        
        Args:
            path: Optional path to load from (overrides self.model_path)
        """
        load_path = path or self.model_path
        
        if not load_path:
            print("Warning: No model path specified, cannot load model")
            return
        
        if not Path(load_path).exists():
            print(f"Warning: Model file {load_path} not found, starting with empty model")
            return
        
        # Load from file
        with open(load_path, 'r') as f:
            save_data = json.load(f)
        
        # Reconstruct model from JSON
        self.model = {}
        
        for state_key, actions in save_data['model'].items():
            # Convert string key back to tuple
            state = eval(state_key)
            self.model[state] = {}
            
            for action_str, transition_data in actions.items():
                action = int(action_str)
                
                # Reconstruct next_state
                next_state_str = transition_data['next_state']
                if next_state_str == "None":
                    next_state = None
                else:
                    next_state = eval(next_state_str)
                
                reward = transition_data['reward']
                terminated = transition_data['terminated']
                
                self.model[state][action] = (next_state, reward, terminated)
        
        # Reconstruct visited_state_actions
        self.visited_state_actions = [
            (eval(state_str), int(action))
            for state_str, action in save_data['visited_state_actions']
        ]
        
        print(f"Model loaded from {load_path}")
        print(f"  States: {len(self.model)}")
        print(f"  Transitions: {sum(len(actions) for actions in self.model.values())}")
        print(f"  (state, action) pairs: {len(self.visited_state_actions)}")
    
    def get_stats(self):
        """
        Get extended statistics including model persistence info.
        
        Returns:
            Dict with Q-table and model stats plus model path info
        """
        # Get base stats from parent DynaQAgent
        stats = super().get_stats()
        
        # Add TabularDynaQ specific stats
        stats['model_path'] = self.model_path
        stats['model_loaded'] = bool(self.model_path and Path(self.model_path).exists())
        
        return stats
    
    def cleanup(self):
        """
        Save both Q-table and model before cleanup.
        Call this when done training.
        """
        if self.q_table_path:
            self.save_q_table()
        
        if self.model_path:
            self.save_model()
        
        print("\n✓ TabularDynaQ agent cleanup complete")


