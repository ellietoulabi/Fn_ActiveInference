import numpy as np
from agents.QLearning.qlearning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    """
    Dyna-Q Agent with RECENCY BIAS for non-stationary environments.
    
    Extends QLearningAgent with model-based planning and recency-weighted sampling.
    
    Combines:
    - Direct RL: Q-learning from real experience (inherited from QLearningAgent)
    - Model Learning: Stores transitions in a world model
    - Planning: Generates simulated experience from the model
    - RECENCY BIAS: Prioritizes recent experiences during planning
    
    Recency Bias Implementation:
    - Tracks timestamp of each state-action pair
    - Samples from model with weights proportional to recency_decay^age
    - Newer experiences have higher probability of being replayed
    - Helps agent adapt faster when environment dynamics change
    
    Inherits all features from QLearningAgent:
    - Q-table save/load functionality
    - State representation and conversion
    - Statistics tracking
    - Epsilon-greedy action selection
    """
    
    def __init__(
        self,
        action_space_size,
        planning_steps=10,
        recency_decay=0.99,
        q_table_path=None,
        learning_rate=0.3,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.1,
        load_existing=False
    ):
        # Initialize parent QLearningAgent
        super().__init__(
            action_space_size=action_space_size,
            q_table_path=q_table_path,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            load_existing=load_existing
        )
        
        # Dyna-Q specific: planning steps per real step
        self.planning_steps = planning_steps
        
        # RECENCY BIAS: Decay rate for weighting recent experiences
        # Higher (e.g., 0.99) = slow forgetting, Lower (e.g., 0.9) = fast forgetting
        self.recency_decay = recency_decay
        
        # Model stores: model[state][action] = (next_state, reward, terminated)
        self.model = {}
        
        # Track all visited state-action pairs for efficient planning
        self.visited_state_actions = []
        
        # RECENCY BIAS: Track when each state-action was last seen
        self.state_action_timestamps = {}  # (state, action) -> global_step
        self.global_step = 0  # Total steps across all episodes
    
    def update_model(self, state, action, next_state, reward, terminated):
        """
        Store transition in the world model with recency tracking.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting next state (or None if terminal)
            reward: Reward received
            terminated: Whether the episode terminated
        """
        if state is None:
            return
            
        if state not in self.model:
            self.model[state] = {}
        
        self.model[state][action] = (next_state, reward, terminated)
        
        # Track this state-action pair for efficient planning
        # Only add if not already present to avoid duplicates
        state_action = (state, action)
        if state_action not in self.visited_state_actions:
            self.visited_state_actions.append(state_action)
        
        # RECENCY BIAS: Update timestamp for this state-action pair
        self.state_action_timestamps[state_action] = self.global_step
        self.global_step += 1
    
    def planning(self):
        """
        Perform planning steps with RECENCY BIAS sampling.
        
        For each planning step:
        1. Sample a previously visited (state, action) pair with recency weighting
        2. Use the model to predict (next_state, reward)
        3. Update Q-table with this simulated experience
        
        RECENCY BIAS: More recently seen state-action pairs have higher probability
        of being sampled. This helps the agent adapt faster to non-stationary 
        environments where old experiences become irrelevant.
        
        Weight for each (s,a) = recency_decay^age, where age = current_step - last_seen_step
        """
        if not self.visited_state_actions:
            return  # No experience yet to plan from
        
        # Calculate recency weights for all visited state-action pairs
        current_step = self.global_step
        weights = []
        
        for state, action in self.visited_state_actions:
            state_action = (state, action)
            last_seen_step = self.state_action_timestamps.get(state_action, 0)
            age = current_step - last_seen_step
            
            # Exponential decay: more recent = higher weight
            # recency_decay^age gives higher weight to smaller age (recent experiences)
            weight = self.recency_decay ** age
            weights.append(weight)
        
        # Normalize weights to form a probability distribution
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Fallback to uniform if all weights are zero (shouldn't happen)
            weights = np.ones(len(weights)) / len(weights)
        
        # Perform planning steps with recency-biased sampling
        for _ in range(self.planning_steps):
            # Sample a state-action pair with probability proportional to recency
            idx = np.random.choice(len(self.visited_state_actions), p=weights)
            state, action = self.visited_state_actions[idx]
            
            # Get the model's prediction for this state-action pair
            if state in self.model and action in self.model[state]:
                next_state, reward, terminated = self.model[state][action]
                
                # Update Q-table using simulated experience
                # This uses the parent class's update_q_table method
                self.update_q_table(state, action, reward, next_state)
                
                # Note: Parent's update_q_table handles terminal states by
                # checking if next_state is None
    
    def train(self, env, episodes, verbose=True):
        """
        Train the Dyna-Q agent on the given environment.
        
        Args:
            env: The environment to train on
            episodes: Number of episodes to train
            verbose: Whether to print progress
            
        Returns:
            Tuple of (episode_rewards, episode_lengths)
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            # Reset environment (handle both old and new Gym API)
            result = env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
            
            # Convert observation to state using parent's get_state method
            state = self.get_state(obs)
            
            total_reward = 0
            steps = 0
            
            while True:
                # (1) Choose action using epsilon-greedy (inherited method)
                action = self.choose_action(state)
                
                # Step environment (handle both old and new Gym API)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result
                    terminated = done
                
                # Convert next observation to state
                next_state = self.get_state(next_obs) if not done else None
                
                # (2) Direct RL: Update Q-table from real experience
                self.update_q_table(state, action, reward, next_state)
                
                # (3) Model Learning: Store transition in model
                self.update_model(state, action, next_state, reward, terminated)
                
                # (4) Planning: Learn from simulated experience
                self.planning()
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Decay exploration rate
            self.decay_exploration()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards, episode_lengths
    
    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluate the trained agent without exploration.
        
        Args:
            env: The environment to evaluate on
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            List of episode rewards
        """
        eval_rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation
        
        for episode in range(num_episodes):
            result = env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
            
            state = self.get_state(obs)
            total_reward = 0
            
            if render:
                print(f"\n--- Evaluation Episode {episode + 1} ---")
                env.render()
            
            while True:
                action = self.choose_action(state)
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result
                
                if render:
                    action_meaning = getattr(env, 'ACTION_MEANING', {}).get(action, 'unknown')
                    print(f"Action: {action} ({action_meaning})")
                    env.render()
                
                next_state = self.get_state(next_obs) if not done else None
                total_reward += reward
                state = next_state
                
                if done:
                    if render:
                        print(f"Episode reward: {total_reward}, Result: {info.get('result', 'N/A')}")
                    break
            
            eval_rewards.append(total_reward)
        
        self.epsilon = old_epsilon  # Restore exploration rate
        return eval_rewards
    
    def get_stats(self):
        """
        Get extended statistics including model and recency bias info.
        
        Returns:
            Dict with Q-table stats (from parent) plus model and recency stats
        """
        # Get base stats from parent
        stats = super().get_stats()
        
        # Add Dyna-Q specific stats
        stats['planning_steps'] = self.planning_steps
        stats['model_size'] = len(self.model)
        stats['total_transitions'] = sum(len(actions) for actions in self.model.values())
        stats['visited_state_actions'] = len(self.visited_state_actions)
        
        # Add recency bias stats
        stats['recency_decay'] = self.recency_decay
        stats['global_step'] = self.global_step
        
        # Calculate average age of experiences
        if self.state_action_timestamps:
            current_step = self.global_step
            ages = [current_step - step for step in self.state_action_timestamps.values()]
            stats['avg_experience_age'] = np.mean(ages)
            stats['max_experience_age'] = np.max(ages)
            stats['min_experience_age'] = np.min(ages)
        else:
            stats['avg_experience_age'] = 0
            stats['max_experience_age'] = 0
            stats['min_experience_age'] = 0
        
        return stats

