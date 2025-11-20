import numpy as np
from gym import spaces
import gym


class SingleAgentRedBlueButtonEnv(gym.Env):
    """
    Single-Agent Red-Blue Button Environment (Ordinal Task)

    RedBlueButtons is an ordinal task where a single agent must press both buttons in the correct order.
    The environment consists of a red button and a blue button, both initially unpressed. The task is to
    press both buttons, but the order of actions matters. The red button must be pressed first, followed
    by the blue button.

    Win/Lose/Neutral Conditions:
    - **WIN**: Blue button is pressed after red button has been pressed
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
        agent_start_pos=(0, 0),
        max_steps=50,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.red_button = red_button_pos
        self.blue_button = blue_button_pos
        self.agent_start_pos = agent_start_pos
        self.agent_position = agent_start_pos
        self.max_steps = max_steps
        self.step_count = 0
        self.cumulative_reward = 0.0

        self._initialize_common_attributes()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.red_button_pressed = False
        self.blue_button_pressed = False
        self.button_just_pressed = None  # Track which button was just pressed this step
        self.step_count = 0
        self.cumulative_reward = 0.0

        # Reset agent to start position
        self.agent_position = self.agent_start_pos

        # Generate observation directly in reset function
        position_coords = np.array(self.agent_position, dtype=int)

        # At reset, everything is neutral (step_count is 0, no buttons pressed)
        win_lose_neutral = 0  # neutral - episode just started

        observation = {
            "position": position_coords,  # Return position as (x, y) coordinates
            "on_red_button": int(self.is_adjacent_to_red(*self.agent_position)),
            "on_blue_button": int(self.is_adjacent_to_blue(*self.agent_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,  # 0=neutral, 1=win, 2=lose
            "button_just_pressed": self.button_just_pressed,  # None, "red", or "blue"
        }

        return observation, {}

    def step(self, action):
        # REWARD LOGIC:
        # if action == press:
        #   - if on red button and red not pressed: press it, neutral (reward 0)
        #   - if on blue button and blue not pressed:
        #     - if red pressed: press blue, win (reward 1), terminate
        #     - if red not pressed: press blue, lose (reward -1), terminate
        # if step > maxstep: lose (reward -1), truncate
        # else: neutral (reward 0)
        
        reward = 0.0  # Default reward (neutral)
        terminated = False
        truncated = False
        self.button_just_pressed = None  # Reset button_just_pressed at start of each step
        win_lose_neutral = 0

        info = {
            "step": self.step_count,
            "action": action,
            "action_meaning": self.ACTION_MEANING.get(action, "unknown"),
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "button_just_pressed": self.button_just_pressed,
            "result": win_lose_neutral,
        }

        if action in [0, 1, 2, 3]:  # Movement actions
            x, y = self.agent_position
            dx, dy = 0, 0
            if action == 0:  # up
                dy = -1
            elif action == 1:  # down
                dy = 1
            elif action == 2:  # left
                dx = -1
            elif action == 3:  # right
                dx = 1

            new_pos = (x + dx, y + dy)
            if self._valid_move(new_pos):
                self.agent_position = new_pos
            # Movement: neutral (reward 0) - already set as default

        elif action == 4:  # Press action
            x, y = self.agent_position

            # If on red button and red button not pressed: press it, neutral
            if (x, y) == self.red_button and not self.red_button_pressed:
                self.red_button_pressed = True
                self.button_just_pressed = "red"
                info["button_just_pressed"] = "red"
                reward = 0  # neutral
                win_lose_neutral = 0

            # If on blue button and blue not pressed
            elif (x, y) == self.blue_button and not self.blue_button_pressed:
                self.blue_button_pressed = True
                self.button_just_pressed = "blue"
                info["button_just_pressed"] = "blue"

                # If red pressed: press blue, win (reward 1), terminate
                if self.red_button_pressed:
                    reward = 1
                    terminated = True
                    win_lose_neutral = 1
                # If red not pressed: press blue, lose (reward -1), terminate
                else:
                    reward = -1
                    terminated = True
                    win_lose_neutral = 2

        elif action == 5:  # Noop action - do nothing
            pass  # Neutral (reward 0) - already set as default

        self.step_count += 1

        # Check termination conditions: if step > maxstep: lose (reward -1), truncate
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            reward = -1
            win_lose_neutral = 2
        # elif terminated:
        #     # Termination already handled in button pressing logic
        #     if self.red_button_pressed and self.blue_button_pressed:
        #         info["result"] = "win"  # Blue pressed after red = win
        #     else:
        #         info["result"] = "lose"  # Blue pressed before red = lose
        # else:
        #     info["result"] = "neutral"  # Episode continues

        self.cumulative_reward += reward
        info["reward"] = reward
        info["cumulative_reward"] = self.cumulative_reward
        info["map"] = self.render(mode="silent")

        # Generate observation directly in step function
        position_coords = np.array(self.agent_position, dtype=int)

        # # Determine win/lose/neutral observation
        # if self.step_count >= self.max_steps:
        #     win_lose_neutral = 2  # lose - max steps reached
        # elif self.blue_button_pressed and not self.red_button_pressed:
        #     win_lose_neutral = 2  # lose - blue pressed before red
        # elif self.red_button_pressed and self.blue_button_pressed:
        #     win_lose_neutral = 1  # win - blue pressed after red
        # else:
        #     win_lose_neutral = 0  # neutral - episode continues
        if win_lose_neutral == 0:
            info["result"] = "neutral"
        elif win_lose_neutral == 1:
            info["result"] = "win"
        elif win_lose_neutral == 2:
            info["result"] = "lose"

        observation = {
            "position": position_coords,  # Return position as (x, y) coordinates
            "on_red_button": int(self.is_adjacent_to_red(*self.agent_position)),
            "on_blue_button": int(self.is_adjacent_to_blue(*self.agent_position)),
            "red_button_pressed": int(self.red_button_pressed),
            "blue_button_pressed": int(self.blue_button_pressed),
            "win_lose_neutral": win_lose_neutral,  # 0=neutral, 1=win, 2=lose
            "button_just_pressed": self.button_just_pressed,  # None, "red", or "blue"
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        grid = np.full((self.height, self.width), ".", dtype=str)

        # Add agent first
        x, y = self.agent_position
        grid[y, x] = "A"

        # Add buttons (only if agent is not on them)
        if self.agent_position != self.red_button:
            grid[self.red_button[1], self.red_button[0]] = (
                "r" if not self.red_button_pressed else "R"
            )
        if self.agent_position != self.blue_button:
            grid[self.blue_button[1], self.blue_button[0]] = (
                "b" if not self.blue_button_pressed else "B"
            )

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

        # Agent can move anywhere on the map (3x3 grid)
        # Position is represented as (x, y) coordinates directly
        # No need for predefined valid positions - anywhere on the grid is valid

        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)  # 0-3: movement, 4: press, 5: noop

        self.observation_space = spaces.Dict(
            {
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
            }
        )

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
            "agent_position": self.agent_position,
            "red_button": self.red_button,
            "blue_button": self.blue_button,
            "red_button_pressed": self.red_button_pressed,
            "blue_button_pressed": self.blue_button_pressed,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }
