import numpy as np
from typing import Union, Optional

# from pymdp import utils, inference, control, learning
from . import utils
from . import control
from . import inference

# from utils import construct_policies, format_observations
# from control import get_expected_states


class Agent:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        policies: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        gamma: float = 16.0,
        alpha: float = 16.0,
        policy_len: int = 1,
        inference_horizon: int = 1,
        action_selection: str = "deterministic",
        sampling_mode: str = "full",
        inference_algorithm: str = "VANILLA",
    ):
        """
        Initialize all model matrices, priors, and hyper-parameters.
        """
        # Store raw A, B, C, D
        self.A = A
        self.B = B
        self.C = C if C is not None else self._make_uniform_C()
        self.D = D if D is not None else self._make_uniform_D()

        # Matrix dimensions
        self.num_modalities = len(A)
        self.num_obs = [Am.shape[0] for Am in self.A]
        self.num_factors = self.B.shape[0]
        self.num_states = [self.B[f].shape[0] for f in range(len(self.B))]
        # Control factors: 1 = controllable (all actions affect), 0 = non-controllable (only noop affects)
        self.control_factors = [1 if B[f].shape[2] > 1 else 0 for f in range(len(B))]

        # Policy library and selection params
        self.gamma = gamma
        self.alpha = alpha
        self.policy_len = policy_len
        
        self.actions = list(range(len(actions)))

        if policies:
            self.policies = np.array(policies, dtype=object)
        else:
            self.policies = self._construct_policies()

        # Inference settings
        self.inference_horizon = inference_horizon
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.inference_algorithm = inference_algorithm
        self.inference_params = self._get_default_params()
        
        # Policy evaluation settings
        self.use_utility = True  # Use preference-based utility
        self.use_states_info_gain = True  # Use state information gain
        self.E = None  # Prior beliefs over policies (if None, uniform)

        # Internal state
        self.qs = None  # posterior over hidden states
        self.q_pi = None  # posterior over policies
        self.action = None  # chosen action
        self.prev_actions = None  # previous action

        # Reset into a fresh starting state
        self.curr_timestep = 0
        self.reset()

    def reset(self):
        """
        Reset posterior beliefs (qs) to the uniform prior D, clear any history,
        and reset time counters.
        """
        self.curr_timestep = 0
        # Initialize qs with uniform beliefs over states
        self.qs = [np.ones(d.shape[0]) / d.shape[0] for d in self.D]
        # Initialize policy posterior uniformly
        self.q_pi = np.ones(len(self.policies)) / len(self.policies)
        self.action = None
        self.prev_actions = None

    def infer_states(self, observation):

        if self.inference_algorithm == "VANILLA":
            if self.action is not None:
                prior = control.get_expected_state(
                    self.B, self.qs, self.action, self.control_factors
                )
            else:
                prior = np.array(self.D, dtype=object)
            one_hot_obs = utils.format_observations(observation, self.num_obs)
            qs = inference.vanilla_fpi_update_posterior_states(
                self.A,
                one_hot_obs,
                prior,
                self.num_obs,
                self.num_states,
                **self.inference_params,
            )
            
            # Update the agent's state beliefs
            self.qs = qs
        else:
            raise NotImplementedError(
                "Only VANILLA inference algorithm is supported for now"
            )

        return qs

    def infer_policies(self):
        """
        Compute expected free energies G for each policy and update self.q_pi
        via softmax(-G * gamma + log E).
        Returns (self.q_pi, G).
        """

        if self.inference_algorithm == "VANILLA":
            print(f"    Debug: qs shape: {[qs_f.shape for qs_f in self.qs]}")
            print(f"    Debug: policies shape: {self.policies.shape}")
            print(f"    Debug: control_factors: {self.control_factors}")
            
            result = control.vanilla_fpi_update_posterior_policies(
                self.qs,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                E=self.E,
                gamma=self.gamma,
                control_factors=self.control_factors,
            )
            
            print(f"    Debug: result type: {type(result)}")
            print(f"    Debug: result: {result}")
            
            # The function returns (q_pi, G), so we need the first element
            if isinstance(result, tuple):
                self.q_pi = result[0]
            else:
                self.q_pi = result
                
            return self.q_pi, result[1] if isinstance(result, tuple) else None
        else:
            raise NotImplementedError(
                "Only VANILLA inference algorithm is supported for now"
            )

    def sample_action(self):
        """
        Draw or deterministically select an action (or action‐factor vector)
        from self.q_pi and self.policies, then advance time.
        Returns the chosen action.

        Alpha: the "inverse temperature" parameter that controls how deterministic vs random the agent's action selection is.
               Alpha scales the probabilities before sampling, making the agent more or less "confident" in its choices.
               
               Temperature Analogy:
                - High temperature = more random, "hot" behavior
                - Low temperature = more deterministic, "cold" behavior
               Alpha = 1/temperature, so:
                - High alpha = low temperature = more deterministic
                - Low alpha = high temperature = more random
        """
        if self.sampling_mode == "marginal":
            self.action = control.sample_action(
                self.q_pi, self.policies, self.action_selection, self.alpha, self.actions
            )
        elif self.sampling_mode == "full":
            self.action = control.sample_policy(
                self.q_pi, self.policies, self.action_selection, self.alpha
            )
        else:
            raise NotImplementedError(
                f"Sampling mode {self.sampling_mode} not implemented"
            )

        # Return the sampled action
        return self.action

    def step_time(self):
        # NOTE: some chaages need to be made for MMP inference
        if self.prev_actions is None:
            self.prev_actions = [self.action]
        else:
            self.prev_actions.append(self.action)
        self.curr_timestep += 1
        return self.curr_timestep

    # def set_latest_beliefs(self):
    #     """
    #     Shift the history of post-dictive beliefs forward in time, so that the penultimate belief before the beginning of the horizon is correctly indexed.
    #     """
    #     self.qs = self.prev_actions[-self.inference_horizon :]
    #     self.prev_actions = self.prev_actions[-self.inference_horizon :]

    # ──────────────────────────────────────────────────────────────────────────

    # Helper constructors (you can customize these)
    def _make_uniform_C(self):
        """Build a uniform preference array C for each modality."""
        return np.array([np.ones(dim) / dim for dim in self.num_obs], dtype=object)

    def _make_uniform_D(self):
        """Build a uniform prior D over each hidden-state factor."""
        return np.array([np.ones(dim) / dim for dim in self.num_states], dtype=object)

    def _construct_policies(self):
        """Generate all possible action‐sequences up to self.policy_len."""
        # Use the actions provided, or default to 6 actions if not provided
        num_actions = len(self.actions) if self.actions is not None else 6
        return utils.construct_policies(
            range(num_actions), self.policy_len
        )

    def _get_default_params(self):
        default_params = None
        if self.inference_algorithm == "VANILLA":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}

        return default_params
