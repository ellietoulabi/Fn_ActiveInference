"""
OPSRL (Optimistic Posterior Sampling for Reinforcement Learning) Agent.

Adapted from rlberry implementation to work with SingleAgentRedBlueButtonEnv.
"""
import numpy as np
import gymnasium.spaces as spaces
import logging

from agents.OPSRL.utils import (
    backward_induction_in_place,
    backward_induction_sd,
)

logger = logging.getLogger(__name__)


class OPSRLAgent:
    """
    OPSRL algorithm adapted for SingleAgentRedBlueButtonEnv.
    
    Uses beta prior for the "Bernoullized" rewards (instead of Gaussian-gamma prior).
    
    Notes
    -----
    The recommended policy after all the episodes is computed without
    exploration bonuses.
    
    Parameters
    ----------
    env : gym.Env
        Environment (SingleAgentRedBlueButtonEnv).
    gamma : double, default: 1.0
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    scale_prior_reward : double, default: 1.0
        scale of the Beta (uniform) prior,
        i.e prior is Beta(scale_prior_reward*(1,1))
    thompson_samples: int, default: 1
        number of thompson samples
    prior_transition : string, default: 'uniform'
        type of Dirichlet prior in {'optimistic', 'uniform'}.
    bernoullized_reward: bool, default: True
        If true the rewards are Bernoullized
    reward_free : bool, default: False
        If true, ignores rewards and uses only 1/n bonuses.
    stage_dependent : bool, default: False
        If true, assume that transitions and rewards can change with the stage h.
    seed : int, optional
        Random seed for reproducibility.
    
    References
    ----------
    .. [1] Osband et al., 2013
        (More) Efficient Reinforcement Learning via Posterior Sampling
        https://arxiv.org/abs/1306.0940
    """
    
    name = "OPSRL"
    
    def __init__(
        self,
        env,
        gamma=1.0,
        horizon=100,
        bernoullized_reward=True,
        scale_prior_reward=1.0,
        thompson_samples=1,
        prior_transition='uniform',
        scale_prior_transition=None,
        reward_free=False,
        stage_dependent=False,
        seed=None,
        **kwargs
    ):
        self.env = env
        self.gamma = gamma
        self.horizon = horizon
        self.bernoullized_reward = bernoullized_reward
        self.scale_prior_reward = scale_prior_reward
        self.thompson_samples = thompson_samples
        assert prior_transition in ['uniform', 'optimistic']
        self.prior_transition = prior_transition
        self.scale_prior_transition = scale_prior_transition
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Check environment
        assert isinstance(self.env.action_space, spaces.Discrete)
        
        # Other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))
        
        # Compute state space size
        # State factors: agent_pos (9), on_red_button (2), on_blue_button (2),
        #                 red_button_pressed (2), blue_button_pressed (2)
        self.width = getattr(env, 'width', 3)
        self.height = getattr(env, 'height', 3)
        self.n_positions = self.width * self.height  # 9
        self.n_states = self.n_positions * 2 * 2 * 2 * 2  # 144
        self.n_actions = self.env.action_space.n  # 6
        
        # Set scale_prior_transition if not provided
        if self.scale_prior_transition is None:
            if self.prior_transition == 'uniform':
                self.scale_prior_transition = 1.0 / self.n_states
            else:
                self.scale_prior_transition = 1.0
        
        # Maximum value - handle reward range
        # Environment rewards: -1 (lose), 0 (neutral), 1 (win)
        r_min = -1.0
        r_max = 1.0
        r_range = r_max - r_min
        if r_range == np.inf or r_range == 0.0:
            logger.warning(
                "{}: Reward range is zero or infinity. ".format(self.name)
                + "Setting it to 2."
            )
            r_range = 2.0
        
        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon - 1)):
            self.v_max[hh] = r_range + self.gamma * self.v_max[hh + 1]
        
        # Initialize
        self.reset()
    
    def _obs_to_state(self, obs):
        """
        Convert observation dict to discrete state integer.
        
        State factors:
        - agent_pos: 0-8 (flat index from (x, y))
        - on_red_button: 0-1
        - on_blue_button: 0-1
        - red_button_pressed: 0-1
        - blue_button_pressed: 0-1
        
        Returns
        -------
        state : int
            Discrete state index (0 to n_states-1)
        """
        if obs is None:
            return None
        
        # Convert position (x, y) to flat index
        position = obs["position"]
        if isinstance(position, np.ndarray):
            position = position.flatten()
            x, y = int(position[0]), int(position[1])
        elif isinstance(position, (list, tuple)):
            x, y = int(position[0]), int(position[1])
        else:
            # Fallback if position is already an integer
            x, y = int(position % self.width), int(position // self.width)
        
        agent_pos = y * self.width + x
        
        # Extract other state factors
        on_red_button = int(obs.get("on_red_button", 0))
        on_blue_button = int(obs.get("on_blue_button", 0))
        red_button_pressed = int(obs.get("red_button_pressed", 0))
        blue_button_pressed = int(obs.get("blue_button_pressed", 0))
        
        # Encode state as integer
        # state = agent_pos * 16 + on_red_button * 8 + on_blue_button * 4 + 
        #         red_button_pressed * 2 + blue_button_pressed
        state = (agent_pos * 16 + 
                 on_red_button * 8 + 
                 on_blue_button * 4 + 
                 red_button_pressed * 2 + 
                 blue_button_pressed)
        
        return state
    
    def reset(self, **kwargs):
        """Reset the agent's internal state."""
        H = self.horizon
        S = self.n_states
        A = self.n_actions
        
        if self.stage_dependent:
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
            if self.prior_transition == 'optimistic':
                shape_hsas = (H, S, A, S + 1)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)
            if self.prior_transition == 'optimistic':
                shape_hsas = (S, A, S + 1)
        
        # Prior transitions
        self.N_sas = self.scale_prior_transition * np.ones(shape_hsas)
        if self.prior_transition == 'optimistic':
            self.N_sas = np.zeros(shape_hsas)
            self.N_sas[..., -1] += self.scale_prior_transition
        
        # Prior rewards
        self.M_sa = self.scale_prior_reward * np.ones(shape_hsa + (2,))
        
        # Value functions
        self.V = np.zeros((H, S))
        if self.prior_transition == 'optimistic':
            self.V = np.zeros((H, S + 1))
        self.Q = np.zeros((H, S, A))
        
        # Init V if needed
        if self.prior_transition == 'optimistic':
            for hh in range(self.horizon):
                self.V[hh, :] = self.v_max[hh]
        
        # for rec. policy
        self.V_policy = np.zeros((H, S))
        self.Q_policy = np.zeros((H, S, A))
        
        # ep counter
        self.episode = 0
    
    def policy(self, observation):
        """
        Get the recommended policy action for the given observation.
        
        Parameters
        ----------
        observation : dict
            Observation from the environment
            
        Returns
        -------
        action : int
            Action to take
        """
        state = self._obs_to_state(observation)
        if state is None or self.Q_policy is None:
            return self.rng.randint(0, self.n_actions)
        return int(self.Q_policy[0, state, :].argmax())
    
    def _get_action(self, state, hh=0):
        """Sampling policy."""
        if state is None or self.Q is None:
            return self.rng.randint(0, self.n_actions)
        q_values = self.Q[hh, state, :]
        # Check for NaN or all equal values - use random action if so
        if np.any(np.isnan(q_values)) or np.all(q_values == q_values[0]):
            return self.rng.randint(0, self.n_actions)
        return int(q_values.argmax())
    
    def _update(self, state, action, next_state, reward, hh):
        """Update posterior distributions."""
        if state is None:
            return
        
        bern_reward = reward
        if self.bernoullized_reward:
            # Normalize reward to [0, 1] for Bernoulli
            # Original rewards: -1, 0, 1 -> map to [0, 1]
            normalized_reward = (reward + 1.0) / 2.0
            bern_reward = self.rng.binomial(1, normalized_reward)
        
        # update posterior
        if self.stage_dependent:
            if next_state is not None:
                self.N_sas[hh, state, action, next_state] += 1
            self.M_sa[hh, state, action, 0] += bern_reward
            self.M_sa[hh, state, action, 1] += 1 - bern_reward
        else:
            if next_state is not None:
                self.N_sas[state, action, next_state] += 1
            self.M_sa[state, action, 0] += bern_reward
            self.M_sa[state, action, 1] += 1 - bern_reward
    
    def _run_episode(self):
        """Run a single episode."""
        # sample reward and transitions from posterior
        B = self.thompson_samples
        
        if self.stage_dependent:
            M_sab_zero = np.repeat(self.M_sa[..., 0, np.newaxis], B, -1)
            M_sab_one = np.repeat(self.M_sa[..., 1, np.newaxis], B, -1)
            N_sasb = np.repeat(self.N_sas[..., np.newaxis], B, axis=-1)
        else:
            M_sab_zero = np.repeat(self.M_sa[..., 0, np.newaxis], B, -1)
            M_sab_one = np.repeat(self.M_sa[..., 1, np.newaxis], B, -1)
            N_sasb = np.repeat(self.N_sas[..., np.newaxis], B, axis=-1)
        
        # Sample rewards from Beta distribution
        self.R_samples = self.rng.beta(M_sab_zero, M_sab_one)
        
        # Sample transitions from Dirichlet (via Gamma)
        self.P_samples = self.rng.gamma(N_sasb)
        # Add small epsilon to avoid zeros
        self.P_samples = self.P_samples + 1e-10
        # Normalize to get probabilities
        if self.stage_dependent:
            sums = self.P_samples.sum(-2, keepdims=True)
            self.P_samples = self.P_samples / sums
        else:
            sums = self.P_samples.sum(-1, keepdims=True)
            self.P_samples = self.P_samples / sums
        
        # Denormalize rewards back to [-1, 1] range
        self.R_samples = 2.0 * self.R_samples - 1.0
        
        # run backward induction
        if self.stage_dependent:
            backward_induction_sd(
                self.Q, self.V, self.R_samples, self.P_samples, self.gamma, self.v_max[0]
            )
        else:
            backward_induction_in_place(
                self.Q,
                self.V,
                self.R_samples,
                self.P_samples,
                self.horizon,
                self.gamma,
                self.v_max[0],
            )
        
        # interact for H steps
        episode_rewards = 0
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        state = self._obs_to_state(obs)
        
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
                terminated = done
            
            episode_rewards += reward  # used for logging only
            
            next_state = self._obs_to_state(next_obs) if not done else None
            
            if self.reward_free:
                reward = 0.0  # set to zero before update if reward_free
            
            self._update(state, action, next_state, reward, hh)
            
            state = next_state
            if done:
                break
        
        # update info
        self.episode += 1
        
        # return sum of rewards collected in the episode
        return episode_rewards
    
    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        
        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            encounters a terminal state in which case it stops early.
        """
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1
        
        # compute Q function for the recommended policy
        R_hat = self.M_sa[..., 0] / (self.M_sa[..., 0] + self.M_sa[..., 1])
        # Denormalize rewards back to [-1, 1] range
        R_hat = 2.0 * R_hat - 1.0
        
        P_hat = self.N_sas / self.N_sas.sum(-1, keepdims=True)
        
        if self.stage_dependent:
            backward_induction_sd(
                self.Q_policy, self.V_policy, R_hat, P_hat, self.gamma, self.v_max[0]
            )
        else:
            backward_induction_in_place(
                self.Q_policy,
                self.V_policy,
                R_hat,
                P_hat,
                self.horizon,
                self.gamma,
                self.v_max[0],
            )

