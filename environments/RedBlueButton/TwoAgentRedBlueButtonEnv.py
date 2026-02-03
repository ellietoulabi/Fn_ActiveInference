import numpy as np
try:
    from gymnasium import spaces
    import gymnasium as gym
except ImportError:
    # Fallback to old gym
    from gym import spaces
    import gym


class TwoAgentRedBlueButtonEnv(gym.Env):
    """
    Two-Agent Red-Blue Button Environment (Ordinal Task)

    RedBlueButtons is an ordinal task where two agents must collaborate to press both buttons in the correct order.
    The environment consists of a red button and a blue button, both initially unpressed. The task is to
    press both buttons, but the order of actions matters. The red button must be pressed first, followed
    by the blue button. Either agent can press either button.

    Win/Lose/Neutral Conditions:
    - **WIN**: Blue button is pressed after red button has been pressed (by either agent)
    - **LOSE**: Episode reaches max steps OR blue button is pressed before red button (by either agent)
    - **NEUTRAL**: All other cases (episode continues)

    SPARSE REWARD STRUCTURE:
    - Reward = 0.0 for all intermediate steps (movement, pressing red button, etc.)
    - Reward = +1.0 only when episode ends with WIN (blue pressed after red)
    - Reward = -1.0 only when episode ends with LOSS (blue pressed before red, or timeout)
    - Both agents receive the same shared reward.
    
    Action format: dict with keys 'agent_0' and 'agent_1', each containing an action (0-5)
    Observation format: dict with keys 'agent_0' and 'agent_1', each containing agent-specific observations
    """

    metadata = {"render.modes": ["human"]}

    ACTION_MEANING = {0: "up", 1: "down", 2: "left", 3: "right", 4: "press", 5: "noop"}

    def __init__(
        self,
        width=3,
        height=3,
        red_button_pos=(0, 2),
        blue_button_pos=(2, 0),
        agent_0_start_pos=(0, 0),
        agent_1_start_pos=(2, 2),
        max_steps=50,
        allow_same_position=False,
    ):
        """
        Initialize Two-Agent Red-Blue Button Environment.
        
        Args:
            width: Grid width
            height: Grid height
            red_button_pos: (x, y) position of red button
            blue_button_pos: (x, y) position of blue button
            agent_0_start_pos: Starting position for agent 0
            agent_1_start_pos: Starting position for agent 1
            max_steps: Maximum steps per episode
            allow_same_position: If True, agents can occupy the same cell (default: False)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.red_button = red_button_pos
        self.blue_button = blue_button_pos
        self.agent_0_start_pos = agent_0_start_pos
        self.agent_1_start_pos = agent_1_start_pos
        self.max_steps = max_steps
        self.allow_same_position = allow_same_position
        self.step_count = 0
        self.cumulative_reward = 0.0

        # Agent positions (will be set in reset)
        self.agent_0_position = None
        self.agent_1_position = None

        self._initialize_common_attributes()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.red_button_pressed = False
        self.blue_button_pressed = False
        self.button_just_pressed = None  # Track which button was just pressed this step (and by which agent)
        self.step_count = 0
        self.cumulative_reward = 0.0

        # Reset agents to start positions
        self.agent_0_position = self.agent_0_start_pos
        self.agent_1_position = self.agent_1_start_pos

        # Generate observations for both agents
        observations = self._get_observations()
        return observations, {}

    def step(self, actions):
        """
        Execute one step for both agents.
        
        Args:
            actions: dict with keys 'agent_0' and 'agent_1', each containing an action (0-5)
        
        Returns:
            observations: dict with keys 'agent_0' and 'agent_1' containing observations
            rewards: dict with keys 'agent_0' and 'agent_1' (both receive same reward)
            terminated: bool, whether episode terminated
            truncated: bool, whether episode was truncated
            info: dict with episode information
        """
        # REWARD LOGIC (same as single agent):
        # if action == press:
        #   - if on red button and red not pressed: press it, neutral (reward 0)
        #   - if on blue button and blue not pressed:
        #     - if red pressed: press blue, win (reward 1), terminate
        #     - if red not pressed: press blue, lose (reward -1), terminate
        # if step > maxstep: lose (reward -1), truncate
        # else: neutral (reward 0)
        
        # Validate actions format
        if not isinstance(actions, dict) or 'agent_0' not in actions or 'agent_1' not in actions:
            raise ValueError("Actions must be a dict with keys 'agent_0' and 'agent_1'")
        
        action_0 = actions['agent_0']
        action_1 = actions['agent_1']
        
        reward = 0.0  # Default reward (neutral) - shared by both agents
        terminated = False
        truncated = False
        self.button_just_pressed = None  # Reset button_just_pressed at start of each step
        win_lose_neutral = 0
        button_pressed_by = None  # Track which agent pressed the button

        info = {
            "step": self.step_count,
            "action_0": action_0,
            "action_1": action_1,
            "action_0_meaning": self.ACTION_MEANING.get(action_0, "unknown"),
            "action_1_meaning": self.ACTION_MEANING.get(action_1, "unknown"),
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "button_just_pressed": self.button_just_pressed,
            "button_pressed_by": button_pressed_by,
            "result": win_lose_neutral,
        }

        # Process agent 0's movement first
        if action_0 in [0, 1, 2, 3]:  # Movement actions
            x, y = self.agent_0_position
            dx, dy = 0, 0
            if action_0 == 0:  # up
                dy = -1
            elif action_0 == 1:  # down
                dy = 1
            elif action_0 == 2:  # left
                dx = -1
            elif action_0 == 3:  # right
                dx = 1

            new_pos = (x + dx, y + dy)
            if self._valid_move(new_pos):
                # Check if position is already occupied by other agent
                if self.allow_same_position or new_pos != self.agent_1_position:
                    self.agent_0_position = new_pos

        # Process agent 1's movement
        if action_1 in [0, 1, 2, 3]:  # Movement actions
            x, y = self.agent_1_position
            dx, dy = 0, 0
            if action_1 == 0:  # up
                dy = -1
            elif action_1 == 1:  # down
                dy = 1
            elif action_1 == 2:  # left
                dx = -1
            elif action_1 == 3:  # right
                dx = 1

            new_pos = (x + dx, y + dy)
            if self._valid_move(new_pos):
                # Check if position is already occupied by other agent
                if self.allow_same_position or new_pos != self.agent_0_position:
                    self.agent_1_position = new_pos

        # Process press actions (agent 0 first, then agent 1)
        # If both try to press, agent 0's action takes precedence
        for agent_id, agent_pos, agent_action in [('agent_0', self.agent_0_position, action_0),
                                                    ('agent_1', self.agent_1_position, action_1)]:
            if agent_action == 4:  # Press action
                x, y = agent_pos

                # If on red button and red button not pressed: press it, neutral
                # SPARSE REWARD: No reward for intermediate steps (pressing red button)
                if (x, y) == self.red_button and not self.red_button_pressed:
                    self.red_button_pressed = True
                    self.button_just_pressed = "red"
                    button_pressed_by = agent_id
                    info["button_just_pressed"] = "red"
                    info["button_pressed_by"] = agent_id
                    reward = 0.0  # Sparse: no reward for intermediate steps
                    win_lose_neutral = 0
                    break  # Only one button press per step

                # If on blue button and blue not pressed
                elif (x, y) == self.blue_button and not self.blue_button_pressed:
                    self.blue_button_pressed = True
                    self.button_just_pressed = "blue"
                    button_pressed_by = agent_id
                    info["button_just_pressed"] = "blue"
                    info["button_pressed_by"] = agent_id

                    # SPARSE REWARD: Only give reward at episode termination
                    # If red pressed: press blue, win (reward +1), terminate
                    if self.red_button_pressed:
                        reward = 1.0  # Sparse: reward only at win
                        terminated = True
                        win_lose_neutral = 1
                    # If red not pressed: press blue, lose (reward -1), terminate
                    else:
                        reward = -1.0  # Sparse: reward only at loss
                        terminated = True
                        win_lose_neutral = 2
                    break  # Only one button press per step

        self.step_count += 1

        # SPARSE REWARD: Only give reward at episode termination
        # Check termination conditions: if step > maxstep: lose (reward -1), truncate
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            reward = -1.0  # Sparse: reward only at timeout/loss
            win_lose_neutral = 2

        self.cumulative_reward += reward
        info["reward"] = reward
        info["cumulative_reward"] = self.cumulative_reward
        info["map"] = self.render(mode="silent")
        
        if win_lose_neutral == 0:
            info["result"] = "neutral"
        elif win_lose_neutral == 1:
            info["result"] = "win"
        elif win_lose_neutral == 2:
            info["result"] = "lose"

        # Generate observations for both agents
        observations = self._get_observations(win_lose_neutral)

        # Both agents receive the same shared reward
        rewards = {
            'agent_0': reward,
            'agent_1': reward
        }

        return observations, rewards, terminated, truncated, info

    def _get_observations(self, win_lose_neutral=0):
        """
        Generate observations for both agents.
        
        Args:
            win_lose_neutral: 0=neutral, 1=win, 2=lose
        """
        obs_0 = {
            "position": np.array(self.agent_0_position, dtype=int),
            "on_red_button": int(self.is_adjacent_to_red(*self.agent_0_position)),
            "on_blue_button": int(self.is_adjacent_to_blue(*self.agent_0_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,
            "button_just_pressed": self.button_just_pressed,  # None, "red", or "blue"
            "other_agent_position": np.array(self.agent_1_position, dtype=int),  # Additional info for agent 0
        }
        
        obs_1 = {
            "position": np.array(self.agent_1_position, dtype=int),
            "on_red_button": int(self.is_adjacent_to_red(*self.agent_1_position)),
            "on_blue_button": int(self.is_adjacent_to_blue(*self.agent_1_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,
            "button_just_pressed": self.button_just_pressed,  # None, "red", or "blue"
            "other_agent_position": np.array(self.agent_0_position, dtype=int),  # Additional info for agent 1
        }
        
        return {
            'agent_0': obs_0,
            'agent_1': obs_1
        }

    def render(self, mode="human"):
        grid = np.full((self.height, self.width), ".", dtype=str)

        # Add buttons first (so agents can overwrite them if on same cell)
        if self.agent_0_position != self.red_button and self.agent_1_position != self.red_button:
            grid[self.red_button[1], self.red_button[0]] = (
                "r" if not self.red_button_pressed else "R"
            )
        if self.agent_0_position != self.blue_button and self.agent_1_position != self.blue_button:
            grid[self.blue_button[1], self.blue_button[0]] = (
                "b" if not self.blue_button_pressed else "B"
            )

        # Add agents (agent_0 first, agent_1 can overwrite if same position allowed)
        x0, y0 = self.agent_0_position
        x1, y1 = self.agent_1_position
        
        if self.allow_same_position and (x0, y0) == (x1, y1):
            # Both agents on same cell
            grid[y0, x0] = "X"  # Mark as both agents
        else:
            grid[y0, x0] = "A"  # Agent 0
            if (x1, y1) != (x0, y0):
                grid[y1, x1] = "B"  # Agent 1

        if mode == "human":
            s = "\n".join([" ".join(row) for row in grid])
            print(s)
            print(f"Agent 0 (A): {self.agent_0_position}, Agent 1 (B): {self.agent_1_position}")
            print(f"Red: {'pressed' if self.red_button_pressed else 'unpressed'}, "
                  f"Blue: {'pressed' if self.blue_button_pressed else 'unpressed'}")
            print()

        return grid

    def close(self):
        pass

    def _initialize_common_attributes(self):
        """Initialize shared attributes."""
        self.red_button_pressed = False
        self.blue_button_pressed = False

        # Define action and observation spaces
        # Each agent has the same action space
        self.action_space = spaces.Dict({
            'agent_0': spaces.Discrete(6),  # 0-3: movement, 4: press, 5: noop
            'agent_1': spaces.Discrete(6),
        })

        # Each agent has its own observation space
        agent_obs_space = spaces.Dict({
            "position": spaces.Box(
                low=0,
                high=max(self.width - 1, self.height - 1),
                shape=(2,),
                dtype=int,
            ),  # (x, y) coordinates
            "on_red_button": spaces.Discrete(2),
            "on_blue_button": spaces.Discrete(2),
            "red_button_pressed": spaces.Discrete(2),
            "blue_button_pressed": spaces.Discrete(2),
            "win_lose_neutral": spaces.Discrete(3),  # 0=neutral, 1=win, 2=lose
            "other_agent_position": spaces.Box(
                low=0,
                high=max(self.width - 1, self.height - 1),
                shape=(2,),
                dtype=int,
            ),  # Position of other agent
        })

        self.observation_space = spaces.Dict({
            'agent_0': agent_obs_space,
            'agent_1': agent_obs_space,
        })

    def _valid_move(self, pos):
        """Check if a position is valid for movement."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_adjacent_to_red(self, x, y):
        """Returns True if (x, y) is on the red button."""
        return (x, y) == self.red_button

    def is_adjacent_to_blue(self, x, y):
        """Returns True if (x, y) is on the blue button."""
        return (x, y) == self.blue_button

    def get_state(self):
        """Get full state information (useful for planning)."""
        return {
            "agent_0_position": self.agent_0_position,
            "agent_1_position": self.agent_1_position,
            "red_button": self.red_button,
            "blue_button": self.blue_button,
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

