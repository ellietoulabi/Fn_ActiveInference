"""
Functional Active Inference Agent.

This agent uses functional generative models (A_fn, B_fn, C_fn, D_fn) instead of matrices.
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
        observation_state_dependencies=None,
        policies=None,
        actions=None,
        gamma=4.0,
        alpha=16.0,
        policy_len=3,
        inference_horizon=3,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
        num_iter=16,
        dF_tol=0.001,
        use_action_for_state_inference: bool = False,
        dynamic_policy_generation=True,
        dynamic_destinations=None,
        dynamic_modes=None,
        dynamic_block_other_agent=True,
        dynamic_deduplicate=False,
        dynamic_pad_policies=True,
    ):
        """
        Initialize Functional Active Inference Agent.

        Args:
            A_fn: functional observation model (state_indices) -> obs_likelihoods
            B_fn: functional transition model (qs, action, **env_params) -> next_qs
            C_fn: functional preference model (obs_indices) -> preferences
            D_fn: function returning prior beliefs dict (config) -> D_dict
            state_factors: list of state factor names
            state_sizes: dict mapping factor names to sizes
            observation_labels: dict mapping modality names to observation labels
            env_params: dict with environment parameters
            observation_state_dependencies: optional observation dependency mapping

            policies: optional static list of policies. If provided and
                dynamic_policy_generation=False, these are used directly.
            actions: list of available primitive action indices

            gamma: policy precision
            alpha: action precision
            policy_len: legacy static policy length
            inference_horizon: planning horizon
            action_selection: "deterministic" or "stochastic"
            sampling_mode: "full" or "marginal"
            inference_algorithm: "VANILLA"
            num_iter: max iterations for state inference
            dF_tol: convergence tolerance for state inference

            dynamic_policy_generation: if True, generate policies fresh each step
            dynamic_destinations: optional subset of semantic destinations
            dynamic_modes: optional subset of semantic modes
            dynamic_block_other_agent: whether policy generation treats the other agent tile as blocked
            dynamic_deduplicate: whether to deduplicate generated policies
            dynamic_pad_policies: whether to pad generated policies before inference
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
        self.observation_state_dependencies = observation_state_dependencies
        self.env_params = env_params

        # Primitive actions
        self.actions = actions if actions is not None else list(range(6))

        # Dynamic policy generation config
        self.dynamic_policy_generation = dynamic_policy_generation
        self.dynamic_destinations = dynamic_destinations
        self.dynamic_modes = dynamic_modes
        self.dynamic_block_other_agent = dynamic_block_other_agent
        self.dynamic_deduplicate = dynamic_deduplicate
        self.dynamic_pad_policies = dynamic_pad_policies

        # Static policies, only used if explicitly supplied or if dynamic generation is disabled
        self.static_policies = policies
        self.policy_len = policy_len

        # Current step policy container
        self.policies = []
        self.policy_metadata = []
        self.policy_lengths = None

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.inference_horizon = inference_horizon
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.inference_algorithm = inference_algorithm

        # Inference parameters
        self.num_iter = num_iter
        self.dF_tol = dF_tol
        self.use_action_for_state_inference = bool(use_action_for_state_inference)

        # Policy evaluation settings
        self.use_utility = True
        self.use_states_info_gain = True
        self.E = None  # Prior over policies (uniform if None)

        # Internal state
        self.qs = None
        self.q_pi = None
        self._last_policy_details = None
        self.action = None
        self.prev_actions = []
        self.curr_timestep = 0
        self.last_policy_state = None

        # Initialize
        self.reset()

    # =============================================================================
    # Reset
    # =============================================================================

    def reset(self, config=None, keep_factors=None):
        """
        Reset agent beliefs and counters.

        Args:
            config: optional config dict for D_fn
            keep_factors: optional list of factor names to preserve
        """
        self.curr_timestep = 0

        D = self.D_fn(config)

        if keep_factors is not None and hasattr(self, "qs") and self.qs is not None:
            self.qs = {}
            for factor in self.state_factors:
                if factor in keep_factors:
                    self.qs[factor] = self.qs.get(factor, D[factor]).copy()
                else:
                    self.qs[factor] = D[factor].copy()
        else:
            self.qs = {factor: D[factor].copy() for factor in self.state_factors}

        # Policy-related state resets
        self.policies = []
        self.policy_metadata = []
        self.policy_lengths = None
        self.q_pi = None
        self._last_policy_details = None
        self.last_policy_state = None

        # If using static policies only, initialize them now
        if not self.dynamic_policy_generation:
            if self.static_policies is not None:
                self.policies = [list(p) for p in self.static_policies]
            else:
                self.policies = utils.construct_policies(self.actions, self.policy_len)
            self.q_pi = np.ones(len(self.policies), dtype=float) / len(self.policies)

        self.action = None
        self.prev_actions = []

    # =============================================================================
    # Dynamic policy generation
    # =============================================================================

    def update_policies(self, policy_state):
        """
        Generate and store the current timestep's policy set.

        Args:
            policy_state: dict describing the current Overcooked state for policy generation

        Returns:
            policies: list of current policies
        """
        self.last_policy_state = policy_state

        if not self.dynamic_policy_generation:
            if self.policies is None:
                if self.static_policies is not None:
                    self.policies = [list(p) for p in self.static_policies]
                else:
                    self.policies = utils.construct_policies(self.actions, self.policy_len)
            self.policy_metadata = []
            self.policy_lengths = [len(p) for p in self.policies]
            self.q_pi = np.ones(len(self.policies), dtype=float) / len(self.policies)
            return self.policies

        result = utils.generate_policies_from_state(
            state=policy_state,
            destinations=self.dynamic_destinations,
            modes=self.dynamic_modes,
            block_other_agent=self.dynamic_block_other_agent,
            deduplicate=self.dynamic_deduplicate,
            pad=self.dynamic_pad_policies,
            pad_action=utils.STAY,
            return_metadata=True,
        )

        if self.dynamic_pad_policies:
            self.policies, self.policy_lengths, self.policy_metadata = result
        else:
            self.policies, self.policy_metadata = result
            self.policy_lengths = [len(p) for p in self.policies]

        if len(self.policies) == 0:
            raise ValueError("Dynamic policy generation produced zero policies.")

        self.q_pi = np.ones(len(self.policies), dtype=float) / len(self.policies)
        return self.policies

    # =============================================================================
    # Inference over States
    # =============================================================================

    def infer_states(self, obs_dict):
        """
        Update beliefs over hidden states given observation.

        Args:
            obs_dict: dict of observed indices per modality

        Returns:
            qs: dict of updated posterior beliefs
        """
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        if self.use_action_for_state_inference and self.action is not None:
            prior_dict = control.get_expected_state(
                self.B_fn, self.qs, self.action, self.env_params
            )
        else:
            prior_dict = {factor: self.qs[factor].copy() for factor in self.state_factors}

        map_idx = {f: int(np.argmax(prior_dict[f])) for f in self.state_factors}
        qs_fast = {f: prior_dict[f].copy() for f in self.state_factors}

        direct_modalities = []
        for modality in obs_dict.keys():
            if not modality.endswith("_obs"):
                continue
            factor = modality[: -len("_obs")]
            if factor in self.state_sizes and modality in self.observation_labels:
                if len(self.observation_labels[modality]) == self.state_sizes[factor]:
                    direct_modalities.append((factor, modality))

        for factor, modality in direct_modalities:
            obs_idx = int(obs_dict[modality])
            S = self.state_sizes[factor]
            like = np.zeros(S, dtype=float)
            s_idx = map_idx.copy()
            for v in range(S):
                s_idx[factor] = int(v)
                p_o = self.A_fn(s_idx)[modality]
                if 0 <= obs_idx < len(p_o):
                    like[v] = float(p_o[obs_idx])
            post = qs_fast[factor] * like
            z = float(post.sum())
            qs_fast[factor] = post / (z if z > 0 else 1.0)

        informative_modalities = []
        for m in obs_dict.keys():
            if m.endswith("_obs") and m[: -len("_obs")] in self.state_sizes:
                informative_modalities.append(m)

        if informative_modalities and all(
            any(m == dm for _, dm in direct_modalities) for m in informative_modalities
        ):
            self.qs = qs_fast
        else:
            self.qs = inference.vanilla_fpi_update_posterior_states(
                self.A_fn,
                obs_dict,
                prior_dict,
                self.state_factors,
                self.state_sizes,
                num_iter=self.num_iter,
                dF_tol=self.dF_tol,
                debug=False,
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

        if self.policies is None or len(self.policies) == 0:
            raise ValueError(
                "No policies available for policy inference. "
                "Call update_policies(policy_state) first, or disable dynamic_policy_generation."
            )

        result = control.vanilla_fpi_update_posterior_policies(
            self.qs,
            self.A_fn,
            self.B_fn,
            self.C_fn,
            self.policies,
            self.env_params,
            self.state_factors,
            self.state_sizes,
            self.observation_labels,
            observation_state_dependencies=self.observation_state_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            E=self.E,
            gamma=self.gamma,
            return_policy_details=True,
        )

        if len(result) == 3:
            self.q_pi, G, self._last_policy_details = result
        else:
            self.q_pi, G = result
            self._last_policy_details = None

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
        if self.q_pi is None or self.policies is None:
            raise ValueError("Cannot sample action before policies have been inferred.")

        if self.sampling_mode == "marginal":
            self.action = control.sample_action(
                self.q_pi, self.policies, self.action_selection, self.alpha, self.actions
            )
        elif self.sampling_mode == "full":
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
    # Full Perception-Action Cycle
    # =============================================================================

    def step(self, obs_dict, policy_state=None):
        """
        Complete perception-action cycle.

        Args:
            obs_dict: observation from environment (in model format)
            policy_state: current world-state dict used for dynamic policy generation.
                Required when dynamic_policy_generation=True.

        Returns:
            action: selected action
        """
        self.infer_states(obs_dict)

        if self.dynamic_policy_generation:
            if policy_state is None:
                raise ValueError(
                    "policy_state must be provided when dynamic_policy_generation=True."
                )
            self.update_policies(policy_state)
        elif self.policies is None:
            self.update_policies(policy_state=None)

        self.infer_policies()
        action = self.sample_action()
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
        return None if self.q_pi is None else self.q_pi.copy()

    def get_map_state(self):
        """Get most likely state configuration."""
        return inference.get_map_state(self.qs)

    def get_top_policies(self, top_k=5):
        """Get top-k most likely current-step policies with their probabilities."""
        if self.q_pi is None or self.policies is None:
            return []
        return control.get_top_policies(self.q_pi, self.policies, top_k)

    def get_last_policy_details(self):
        """
        Get utility and info_gain for each policy from the last infer_policies() call.
        Returns list of dicts with keys: policy_idx, policy, utility, info_gain, G, prob.
        """
        if self._last_policy_details is None:
            return []
        return self._last_policy_details

    def get_policy_metadata(self):
        """Get metadata for the current step's generated policies."""
        return list(self.policy_metadata)

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
            observation_state_dependencies=self.observation_state_dependencies,
        )

    def get_inference_diagnostics(self):
        """Get diagnostics about current inference state."""
        D = self.D_fn()
        prior_dict = {factor: D[factor] for factor in self.state_factors}

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