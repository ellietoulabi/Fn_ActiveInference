import numpy as np
from agents.QLearning.qlearning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    """
    Dyna-Q Agent that extends QLearningAgent with model-based planning.
    
    Combines:
    - Direct RL: Q-learning from real experience (inherited from QLearningAgent)
    - Model Learning: Stores transitions in a world model
    - Planning: Generates simulated experience from the model
    
    Planning Modes:
    - Single-step: Traditional Dyna-Q with one-step simulated updates
    - Trajectory Sampling: Multi-step rollouts for faster Q-value propagation
    
    Inherits all features from QLearningAgent:
    - Q-table save/load functionality
    - State representation and conversion
    - Statistics tracking
    - Epsilon-greedy action selection
    
    Trajectory Sampling Parameters:
    - use_trajectory_sampling: Enable multi-step rollout planning
    - n_trajectories: Number of trajectories to simulate per planning step
    - rollout_length: Maximum length of each simulated trajectory
    - planning_epsilon: Exploration rate during simulated rollouts
    """
    
    def __init__(
        self,
        action_space_size,
        planning_steps=10,
        q_table_path=None,
        learning_rate=0.3,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.1,
        load_existing=False,
        use_trajectory_sampling=False,
        n_trajectories=5,
        rollout_length=3,
        planning_epsilon=0.1
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
        
        # Model stores: model[state][action] = (next_state, reward, terminated)
        self.model = {}
        
        # Track all visited state-action pairs for efficient planning
        self.visited_state_actions = []
        
        # Track all visited states for trajectory sampling
        self.visited_states = []
        
        # Trajectory sampling configuration
        self.use_trajectory_sampling = use_trajectory_sampling
        self.n_trajectories = n_trajectories
        self.rollout_length = rollout_length
        self.planning_epsilon = planning_epsilon
        
        # Statistics for trajectory sampling
        self.synthetic_updates_count = 0
    
    def update_model(self, state, action, next_state, reward, terminated):
        """
        Store transition in the world model.
        
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
        
        # Track visited states for trajectory sampling
        if state not in self.visited_states:
            self.visited_states.append(state)
    
    def planning(self):
        """
        Perform planning using either single-step or trajectory sampling mode.
        
        Single-step mode:
        - Sample a previously visited (state, action) pair
        - Use the model to predict (next_state, reward)
        - Update Q-table with this simulated experience
        
        Trajectory sampling mode:
        - Simulate multi-step rollouts starting from visited states
        - Update Q-values at each step of the trajectory
        - More efficient Q-value propagation than single-step
        """
        if self.use_trajectory_sampling:
            self.planning_trajectory_sampling()
        else:
            self.planning_single_step()
    
    def planning_single_step(self):
        """
        Original single-step planning approach.
        
        For each planning step:
        1. Sample a previously visited (state, action) pair
        2. Use the model to predict (next_state, reward)
        3. Update Q-table with this simulated experience
        
        This is more sample-efficient than random sampling because we only
        sample from state-action pairs we've actually encountered.
        """
        if not self.visited_state_actions:
            return  # No experience yet to plan from
        
        for _ in range(self.planning_steps):
            # Sample a previously visited state-action pair
            state, action = self.visited_state_actions[
                np.random.randint(len(self.visited_state_actions))
            ]
            
            # Get the model's prediction for this state-action pair
            if state in self.model and action in self.model[state]:
                next_state, reward, terminated = self.model[state][action]
                
                # Update Q-table using simulated experience
                # This uses the parent class's update_q_table method
                self.update_q_table(state, action, reward, next_state)
                self.synthetic_updates_count += 1
                
                # Note: Parent's update_q_table handles terminal states by
                # checking if next_state is None
    
    def planning_trajectory_sampling(self):
        """
        Multi-step trajectory planning approach.
        
        For each trajectory:
        1. Start from a random visited state
        2. Simulate a rollout of length `rollout_length`
        3. At each step:
           - Select action using epsilon-greedy with planning_epsilon
           - Use model to predict next state and reward
           - Update Q-table
           - Continue from next state
        4. Stop early if terminal state is reached
        
        This allows faster Q-value propagation across multiple steps.
        """
        if not self.visited_states:
            return  # No experience yet to plan from
        
        for _ in range(self.n_trajectories):
            # Start from a random visited state
            state = self.visited_states[np.random.randint(len(self.visited_states))]
            
            # Simulate a trajectory of length rollout_length
            for step in range(self.rollout_length):
                # Check if we have a model for this state
                if state not in self.model or not self.model[state]:
                    break  # No model for this state, stop trajectory
                
                # Select action using epsilon-greedy with planning_epsilon
                if np.random.random() < self.planning_epsilon:
                    # Explore: random action from available actions in model
                    action = np.random.choice(list(self.model[state].keys()))
                else:
                    # Exploit: greedy action based on Q-values
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(self.action_space_size)
                    
                    # Only consider actions we have in the model
                    available_actions = list(self.model[state].keys())
                    q_values = [self.q_table[state][a] for a in available_actions]
                    best_idx = np.argmax(q_values)
                    action = available_actions[best_idx]
                
                # Get the model's prediction
                next_state, reward, terminated = self.model[state][action]
                
                # Update Q-table using simulated experience
                self.update_q_table(state, action, reward, next_state)
                self.synthetic_updates_count += 1
                
                # Stop if we reached a terminal state
                if terminated or next_state is None:
                    break
                
                # Continue from next state
                state = next_state
    
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
        Get extended statistics including model info and trajectory sampling stats.
        
        Returns:
            Dict with Q-table stats (from parent) plus model and trajectory sampling stats
        """
        # Get base stats from parent
        stats = super().get_stats()
        
        # Add Dyna-Q specific stats
        stats['planning_steps'] = self.planning_steps
        stats['model_size'] = len(self.model)
        stats['total_transitions'] = sum(len(actions) for actions in self.model.values())
        stats['visited_state_actions'] = len(self.visited_state_actions)
        stats['visited_states'] = len(self.visited_states)
        
        # Add trajectory sampling stats
        stats['use_trajectory_sampling'] = self.use_trajectory_sampling
        if self.use_trajectory_sampling:
            stats['n_trajectories'] = self.n_trajectories
            stats['rollout_length'] = self.rollout_length
            stats['planning_epsilon'] = self.planning_epsilon
        stats['synthetic_updates_count'] = self.synthetic_updates_count
        
        return stats

