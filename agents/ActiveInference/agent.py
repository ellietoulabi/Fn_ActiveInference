"""
Functional Active Inference Agent.

This agent uses functional generative models (A_fn, B_fn, C_fn, D_fn) instead of matrices.
Designed to work with the RedBlueButton environment.
"""

import numpy as np
from . import utils
from . import control
from . import inference


class Agent:
    """
    Active Inference agent with functional generative model.
    
    Uses dict-based state beliefs and functional A, B, C, D instead of matrices.
    """
    
    def __init__(
        self,
        A_fn,
        B_fn,
        C_fn,
        D_fn,
        state_factors,
        state_sizes,
        observation_labels,
        env_params,
        policies=None,
        actions=None,
        gamma=4.0,  # Lower gamma = more exploration (was 16.0)
        alpha=16.0,
        policy_len=3,  # Balance between speed and planning
        inference_horizon=3,  # Balance between speed and planning
        action_selection="deterministic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
        num_iter=16,
        dF_tol=0.001,
    ):
        """
        Initialize Functional Active Inference Agent.
        
        Args:
            A_fn: functional observation model (state_indices) → obs_likelihoods
            B_fn: functional transition model (qs, action, **env_params) → next_qs
            C_fn: functional preference model (obs_indices) → preferences
            D_fn: function returning prior beliefs dict (config) → D_dict
            state_factors: list of state factor names (e.g., ['agent_pos', 'red_button_pos', ...])
            state_sizes: dict mapping factor names to sizes (e.g., {'agent_pos': 9, ...})
            observation_labels: dict mapping modality names to observation labels
            env_params: dict with environment parameters (width, height, etc.)
            policies: optional list of policies (will construct if None)
            actions: list of available action indices (e.g., [0, 1, 2, 3, 4, 5])
            gamma: policy precision (inverse temperature for policy selection)
            alpha: action precision (inverse temperature for action selection)
            policy_len: length of each policy
            inference_horizon: planning horizon
            action_selection: "deterministic" or "stochastic"
            sampling_mode: "full" or "marginal"
            inference_algorithm: "VANILLA" (only option currently)
            num_iter: max iterations for state inference
            dF_tol: convergence tolerance for state inference
        
        Examples:
            >>> from generative_models.SA_ActiveInference.RedBlueButton import (
            ...     A_fn, B_fn, C_fn, D_fn, model_init
            ... )
            >>> agent = Agent(
            ...     A_fn=A_fn,
            ...     B_fn=B_fn,
            ...     C_fn=C_fn,
            ...     D_fn=D_fn,
            ...     state_factors=list(model_init.states.keys()),
            ...     state_sizes={f: len(v) for f, v in model_init.states.items()},
            ...     observation_labels=model_init.observations,
            ...     env_params={'width': 3, 'height': 3},
            ...     actions=list(range(6)),
            ... )
        """
        # Functional generative model
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.C_fn = C_fn
        self.D_fn = D_fn
        
        # State and observation space structure
        self.state_factors = state_factors
        self.state_sizes = state_sizes
        self.observation_labels = observation_labels
        self.env_params = env_params

        # Actions and policies u d l r n p
        self.actions = actions if actions is not None else list(range(6))
        
        if policies is not None:
            self.policies = policies
        else:
            self.policies = utils.construct_policies(self.actions, policy_len)

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.policy_len = policy_len
        self.inference_horizon = inference_horizon
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.inference_algorithm = inference_algorithm
        
        # Inference parameters
        self.num_iter = num_iter
        self.dF_tol = dF_tol

        # Policy evaluation settings
        self.use_utility = True
        self.use_states_info_gain = True
        self.E = None  # Prior over policies (uniform if None)

        # Internal state
        self.qs = None  # Posterior beliefs (dict)
        self.q_pi = None  # Policy posterior (array)
        self.action = None  # Current action
        self.prev_actions = []  # Action history
        self.curr_timestep = 0

        # Initialize
        self.reset()

    # =============================================================================
    # Reset
    # =============================================================================
    
    def reset(self, config=None, keep_factors=None):
        """
        Reset agent beliefs and counters.
        
        Args:
            config: optional config dict for D_fn (e.g., custom initial state)
            keep_factors: optional list of factor names to preserve from previous episode
                         (e.g., ['red_button_pos', 'blue_button_pos'])
        """
        self.curr_timestep = 0
        
        # Get prior beliefs from D_fn
        D = self.D_fn(config)
        
        # Build new beliefs, preserving specified factors
        if keep_factors is not None and hasattr(self, 'qs'):
            self.qs = {}
            for factor in self.state_factors:
                if factor in keep_factors:
                    # Keep belief from previous episode
                    self.qs[factor] = self.qs.get(factor, D[factor]).copy()
                else:
                    # Reset to prior
                    self.qs[factor] = D[factor].copy()
        else:
            # Full reset
            self.qs = {factor: D[factor].copy() for factor in self.state_factors}
        
        # Initialize policy posterior (uniform)
        self.q_pi = np.ones(len(self.policies)) / len(self.policies)
        
        # Clear action history
        self.action = None
        self.prev_actions = []
    
    # =============================================================================
    # Inference over States
    # =============================================================================
    
    def infer_states(self, obs_dict):
        """
        Update beliefs over hidden states given observation.
        
        Args:
            obs_dict: dict of observed indices per modality
                e.g., {'agent_pos': 3, 'on_red_button': 1, ...}
                (Use env_utils.env_obs_to_model_obs to convert from env format)
        
        Returns:
            qs: dict of updated posterior beliefs
        """
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        # Get prior: if we took an action, use predicted state; else use D
        if self.action is not None:
            prior_dict = control.get_expected_state(
                self.B_fn, self.qs, self.action, self.env_params
            )
        else:
            # First observation - use initial prior D # NOTE: use the D_fn as the prior
            prior_dict = self.D_fn()

        # Run variational inference
        self.qs = inference.vanilla_fpi_update_posterior_states(
            self.A_fn,
            obs_dict,
            prior_dict,
            self.state_factors,
            self.state_sizes,
            num_iter=16,  # Allow sufficient iterations for convergence
            dF_tol=self.dF_tol,
            debug=False,  # Disable debug output
        )
        
        return self.qs

    # =============================================================================
    # Inference over Policies
    # =============================================================================
    
    def infer_policies(self):
        """
        Compute policy posterior by evaluating Expected Free Energy.
        
        Returns:
            q_pi: array of policy probabilities
            G: array of expected free energies
        """
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        self.q_pi, G = control.vanilla_fpi_update_posterior_policies(
            self.qs,
            self.A_fn,
            self.B_fn,
            self.C_fn,
            self.policies,
            self.env_params,
            self.state_factors,
            self.state_sizes,
            self.observation_labels,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            E=self.E,
            gamma=self.gamma,
        )
        
        return self.q_pi, G

    # =============================================================================
    # Action Selection
    # =============================================================================
    
    def sample_action(self):
        """
        Sample action from policy posterior.
        
        Returns:
            action: int, selected action
        """
        if self.sampling_mode == "marginal":
            # Marginalize over policies to get action distribution
            self.action = control.sample_action(
                self.q_pi, self.policies, self.action_selection, self.alpha, self.actions
            )
        elif self.sampling_mode == "full":
            # Sample complete policy and take first action
            self.action = control.sample_policy(
                self.q_pi, self.policies, self.action_selection, self.alpha
            )
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")
        
        return self.action

    # =============================================================================
    # Time Step
    # =============================================================================
    
    def step_time(self):
        """
        Advance time and update action history.
        
        Returns:
            timestep: current timestep after increment
        """
        if self.action is not None:
            self.prev_actions.append(self.action)
        
        self.curr_timestep += 1
        return self.curr_timestep

    # =============================================================================
    # Full Perception-Action Cycle: infer_states -> infer_policies -> sample_action -> step_time
    # =============================================================================
    
    def step(self, obs_dict):
        """
        Complete perception-action cycle.
        
        Args:
            obs_dict: observation from environment (in model format)
        
        Returns:
            action: selected action
        
        Notes:
            This combines: infer_states → infer_policies → sample_action → step_time
        """
        # 1. Update beliefs about current state
        self.infer_states(obs_dict)
        
        # 2. Evaluate policies
        self.infer_policies()
        
        # 3. Select action
        action = self.sample_action()
        
        # 4. Advance time
        self.step_time()
        
        return action
    
    # =============================================================================
    # Debugging
    # =============================================================================
    
    def get_state_beliefs(self):
        """Get current state beliefs (dict format)."""
        return self.qs.copy()
    
    def get_policy_posterior(self):
        """Get current policy posterior (array format)."""
        return self.q_pi.copy()
    
    def get_map_state(self):
        """Get most likely state configuration."""
        return inference.get_map_state(self.qs)
    
    def get_top_policies(self, top_k=5):
        """Get top-k most likely policies with their probabilities."""
        return control.get_top_policies(self.q_pi, self.policies, top_k)
    
    def evaluate_policy(self, policy):
        """
        Evaluate a specific policy's EFE components.
        
        Args:
            policy: list of actions
        
        Returns:
            dict with 'utility', 'info_gain', 'G_total'
        """
        return control.evaluate_policy_components(
            policy,
            self.qs,
            self.A_fn,
            self.B_fn,
            self.C_fn,
            self.env_params,
            self.state_factors,
            self.state_sizes,
            self.observation_labels,
        )
    
    def get_inference_diagnostics(self):
        """Get diagnostics about current inference state."""
        D = self.D_fn()
        prior_dict = {factor: D[factor] for factor in self.state_factors}
        
        # Create dummy obs_dict (we don't have current observation stored)
        # This is a limitation - consider storing last observation if needed
        obs_dict = {}
        
        return inference.compute_inference_diagnostics(
            self.qs,
            prior_dict,
            obs_dict,
            self.A_fn,
            self.state_factors,
            self.state_sizes,
        )
    
    # =============================================================================
    # Configuration
    # =============================================================================
    
    def set_gamma(self, gamma):
        """Set policy precision parameter."""
        self.gamma = gamma
    
    def set_alpha(self, alpha):
        """Set action precision parameter."""
        self.alpha = alpha
    
    def enable_utility(self, enable=True):
        """Enable/disable utility (pragmatic value) in policy evaluation."""
        self.use_utility = enable
    
    def enable_info_gain(self, enable=True):
        """Enable/disable information gain (epistemic value) in policy evaluation."""
        self.use_states_info_gain = enable
    
    def set_policy_prior(self, E):
        """
        Set prior over policies.
        
        Args:
            E: array of policy prior probabilities (or None for uniform)
        """
        self.E = E
