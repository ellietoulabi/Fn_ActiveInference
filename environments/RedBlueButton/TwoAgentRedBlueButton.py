import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TwoAgentRedBlueButtonEnv(gym.Env):
    """
    Two-Agent Red-Blue Button Environment (Ordinal Task)

    RedBlueButtons is an ordinal task where two agents must cooperatively press both buttons in the correct order.
    The environment consists of a red button and a blue button, both initially unpressed. The task is to
    press both buttons, but the order of actions matters. The red button must be pressed first, followed
    by the blue button. Either agent can press either button.

    Win/Lose/Neutral Conditions:
    - **WIN**: Blue button is pressed after red button has been pressed (by either agent)
    - **LOSE**: Episode reaches max steps OR blue button is pressed before red button
    - **NEUTRAL**: All other cases (episode continues)

    No intermediate rewards are given during the episode.
    """

    metadata = {"render.modes": ["human"]}

    ACTION_MEANING = {0: "up", 1: "down", 2: "left", 3: "right", 4: "press", 5: "noop"}

    def __init__(
        self,
        width=3,
        height=3,
        red_button_pos=(0, 2),
        blue_button_pos=(2, 0),
        agent1_start_pos=(0, 0),
        agent2_start_pos=(2, 2),
        max_steps=50,
        allow_overlap=False,  # Whether agents can occupy the same cell
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.red_button = red_button_pos
        self.blue_button = blue_button_pos
        self.agent1_start_pos = agent1_start_pos
        self.agent2_start_pos = agent2_start_pos
        self.agent1_position = agent1_start_pos
        self.agent2_position = agent2_start_pos
        self.max_steps = max_steps
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.allow_overlap = allow_overlap
        self.n_agents = 2

        self._initialize_common_attributes()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.red_button_pressed = False
        self.blue_button_pressed = False
        self.button_just_pressed = None  # Track which button was just pressed this step
        self.button_pressed_by = None  # Track which agent pressed a button this step
        self.step_count = 0
        self.cumulative_reward = 0.0

        # Reset agents to start positions
        self.agent1_position = self.agent1_start_pos
        self.agent2_position = self.agent2_start_pos

        # Generate observation directly in reset function
        position1_coords = np.array(self.agent1_position, dtype=int)
        position2_coords = np.array(self.agent2_position, dtype=int)

        # At reset, everything is neutral (step_count is 0, no buttons pressed)
        win_lose_neutral = 0  # neutral - episode just started

        observation = {
            "agent1_position": position1_coords,
            "agent2_position": position2_coords,
            "agent1_on_red_button": int(self.is_on_red(*self.agent1_position)),
            "agent1_on_blue_button": int(self.is_on_blue(*self.agent1_position)),
            "agent2_on_red_button": int(self.is_on_red(*self.agent2_position)),
            "agent2_on_blue_button": int(self.is_on_blue(*self.agent2_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,  # 0=neutral, 1=win, 2=lose
            "button_just_pressed": self.button_just_pressed,  # None, "red", or "blue"
            "button_pressed_by": self.button_pressed_by,  # None, 1, or 2
        }

        return observation, {}

    def step(self, actions):
        """
        Execute a step with actions from both agents.
        Agent 1 acts first (move + press), then Agent 2 acts.
        
        Args:
            actions: tuple or list of (action_agent1, action_agent2)
                     Each action is an integer from 0-5
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        action1, action2 = actions
        
        reward = 0.0  # Default reward (neutral)
        terminated = False
        truncated = False
        self.button_just_pressed = None
        self.button_pressed_by = None
        win_lose_neutral = 0

        info = {
            "step": self.step_count,
            "action1": action1,
            "action2": action2,
            "action1_meaning": self.ACTION_MEANING.get(action1, "unknown"),
            "action2_meaning": self.ACTION_MEANING.get(action2, "unknown"),
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "button_just_pressed": self.button_just_pressed,
            "button_pressed_by": self.button_pressed_by,
            "result": "neutral",
        }

        # Sequential action processing: Agent 1 acts first, then Agent 2
        
        # --- Agent 1's turn ---
        # Agent 1 movement
        new_pos1 = self._compute_new_position(self.agent1_position, action1)
        if self._valid_move(new_pos1):
            # Check collision with agent 2
            if not self.allow_overlap and new_pos1 == self.agent2_position:
                pass  # Blocked by agent 2, stay in place
            else:
                self.agent1_position = new_pos1

        # Agent 1 press action
        if action1 == 4 and not terminated:
            result = self._handle_press(1)
            if result is not None:
                reward, terminated, win_lose_neutral = result

        # --- Agent 2's turn (only if game hasn't terminated) ---
        if not terminated:
            # Agent 2 movement
            new_pos2 = self._compute_new_position(self.agent2_position, action2)
            if self._valid_move(new_pos2):
                # Check collision with agent 1's NEW position
                if not self.allow_overlap and new_pos2 == self.agent1_position:
                    pass  # Blocked by agent 1, stay in place
                else:
                    self.agent2_position = new_pos2

            # Agent 2 press action
            if action2 == 4 and not terminated:
                result = self._handle_press(2)
                if result is not None:
                    reward, terminated, win_lose_neutral = result

        self.step_count += 1

        # Check truncation: if step >= maxstep: lose (reward -1), truncate
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            reward = -1
            win_lose_neutral = 2

        self.cumulative_reward += reward
        
        # Update info
        info["reward"] = reward
        info["cumulative_reward"] = self.cumulative_reward
        info["button_just_pressed"] = self.button_just_pressed
        info["button_pressed_by"] = self.button_pressed_by
        info["map"] = self.render(mode="silent")

        if win_lose_neutral == 0:
            info["result"] = "neutral"
        elif win_lose_neutral == 1:
            info["result"] = "win"
        elif win_lose_neutral == 2:
            info["result"] = "lose"

        # Generate observation
        position1_coords = np.array(self.agent1_position, dtype=int)
        position2_coords = np.array(self.agent2_position, dtype=int)

        observation = {
            "agent1_position": position1_coords,
            "agent2_position": position2_coords,
            "agent1_on_red_button": int(self.is_on_red(*self.agent1_position)),
            "agent1_on_blue_button": int(self.is_on_blue(*self.agent1_position)),
            "agent2_on_red_button": int(self.is_on_red(*self.agent2_position)),
            "agent2_on_blue_button": int(self.is_on_blue(*self.agent2_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,  # 0=neutral, 1=win, 2=lose
            "button_just_pressed": self.button_just_pressed,
            "button_pressed_by": self.button_pressed_by,
        }

        return observation, reward, terminated, truncated, info

    def _compute_new_position(self, current_pos, action):
        """Compute new position based on action (movement only)."""
        if action not in [0, 1, 2, 3]:  # Not a movement action
            return current_pos
            
        x, y = current_pos
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -1
        elif action == 1:  # down
            dy = 1
        elif action == 2:  # left
            dx = -1
        elif action == 3:  # right
            dx = 1

        return (x + dx, y + dy)

    def _handle_press(self, agent_id):
        """
        Handle press action for a specific agent.
        
        Returns:
            None if no button interaction, otherwise (reward, terminated, win_lose_neutral)
        """
        if agent_id == 1:
            pos = self.agent1_position
        else:
            pos = self.agent2_position

        x, y = pos

        # If on red button and red button not pressed: press it, neutral
        if (x, y) == self.red_button and not self.red_button_pressed:
            self.red_button_pressed = True
            self.button_just_pressed = "red"
            self.button_pressed_by = agent_id
            return (0, False, 0)  # neutral

        # If on blue button and blue not pressed
        elif (x, y) == self.blue_button and not self.blue_button_pressed:
            self.blue_button_pressed = True
            self.button_just_pressed = "blue"
            self.button_pressed_by = agent_id

            # If red pressed: press blue, win (reward 1), terminate
            if self.red_button_pressed:
                return (1, True, 1)  # win
            # If red not pressed: press blue, lose (reward -1), terminate
            else:
                return (-1, True, 2)  # lose

        return None

    def render(self, mode="human"):
        grid = np.full((self.height, self.width), ".", dtype=str)

        # Add buttons first
        grid[self.red_button[1], self.red_button[0]] = (
            "r" if not self.red_button_pressed else "R"
        )
        grid[self.blue_button[1], self.blue_button[0]] = (
            "b" if not self.blue_button_pressed else "B"
        )

        # Add agents (overwrite buttons if on same position)
        x1, y1 = self.agent1_position
        x2, y2 = self.agent2_position
        
        # Handle overlapping agents
        if self.agent1_position == self.agent2_position:
            grid[y1, x1] = "X"  # Both agents on same cell
        else:
            grid[y1, x1] = "1"  # Agent 1
            grid[y2, x2] = "2"  # Agent 2

        if mode == "human":
            s = "\n".join([" ".join(row) for row in grid])
            print(s)
            print()

        return grid

    def close(self):
        pass

    def _initialize_common_attributes(self):
        """Initialize shared attributes across both map and specs parsing."""
        self.red_button_pressed = False
        self.blue_button_pressed = False

        # Define action and observation spaces
        # Each agent has 6 actions: 0-3: movement, 4: press, 5: noop
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # Agent 1 actions
            spaces.Discrete(6),  # Agent 2 actions
        ))

        max_coord = max(self.width - 1, self.height - 1)
        self.observation_space = spaces.Dict(
            {
                "agent1_position": spaces.Box(
                    low=0, high=max_coord, shape=(2,), dtype=int
                ),
                "agent2_position": spaces.Box(
                    low=0, high=max_coord, shape=(2,), dtype=int
                ),
                "agent1_on_red_button": spaces.Discrete(2),
                "agent1_on_blue_button": spaces.Discrete(2),
                "agent2_on_red_button": spaces.Discrete(2),
                "agent2_on_blue_button": spaces.Discrete(2),
                "red_button_pressed": spaces.Discrete(2),
                "blue_button_pressed": spaces.Discrete(2),
                "win_lose_neutral": spaces.Discrete(3),  # 0=neutral, 1=win, 2=lose
            }
        )

    def _valid_move(self, pos):
        """Check if a position is valid for movement."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_on_red(self, x, y):
        """Returns True if (x, y) is on the red button."""
        return (x, y) == self.red_button

    def is_on_blue(self, x, y):
        """Returns True if (x, y) is on the blue button."""
        return (x, y) == self.blue_button

    def get_state(self):
        """Get full state information (useful for planning)."""
        return {
            "agent1_position": self.agent1_position,
            "agent2_position": self.agent2_position,
            "red_button": self.red_button,
            "blue_button": self.blue_button,
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    def get_agent_observation(self, agent_id):
        """
        Get observation from a specific agent's perspective.
        Useful for decentralized learning where each agent has its own observation.
        
        Args:
            agent_id: 1 or 2
            
        Returns:
            dict: Observation from the agent's perspective
        """
        if agent_id == 1:
            my_pos = np.array(self.agent1_position, dtype=int)
            other_pos = np.array(self.agent2_position, dtype=int)
            my_on_red = int(self.is_on_red(*self.agent1_position))
            my_on_blue = int(self.is_on_blue(*self.agent1_position))
        else:
            my_pos = np.array(self.agent2_position, dtype=int)
            other_pos = np.array(self.agent1_position, dtype=int)
            my_on_red = int(self.is_on_red(*self.agent2_position))
            my_on_blue = int(self.is_on_blue(*self.agent2_position))

        return {
            "my_position": my_pos,
            "other_position": other_pos,
            "my_on_red_button": my_on_red,
            "my_on_blue_button": my_on_blue,
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
        }

