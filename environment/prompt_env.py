import gymnasium as gym
import numpy as np

from prompts.base_prompt import BASE_PROMPT
from prompts.transformations import (
    PROMPT_TRANSFORMATIONS,
    apply_transformation,
)


class PromptOptimizationEnv(gym.Env):
    """
    Gymnasium environment for optimizing prompts using reinforcement learning.

    The agent selects prompt transformations in order to improve the
    performance of a frozen language model on GSM8K.

    Parameters
    ----------
    evaluator : GSM8KEvaluator
        Evaluator used to measure the performance of the current prompt.

    max_steps : int, default=5
        Maximum number of transformations that can be applied during
        one episode.

    initial_prompt : str, default=BASE_PROMPT
        Prompt used to initialize each episode.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        evaluator,
        max_steps=5,
        initial_prompt=BASE_PROMPT,
    ):
        super().__init__()

        self.evaluator = evaluator
        self.max_steps = max_steps
        self.initial_prompt = initial_prompt

        # Actions
        self.stop_action = len(PROMPT_TRANSFORMATIONS)

        self.n_actions = self.stop_action + 1

        self.action_space = gym.spaces.Discrete(
            self.n_actions
        )

        # Observation

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32,
        )

        # State variables
        self.prompt = None
        self.current_accuracy = None
        self.selected_actions = []
        self.steps = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Returns
        -------
        observation : np.ndarray
            Initial state.

        info : dict
            Additional information about the initial state.
        """

        super().reset(seed=seed)

        self.prompt = self.initial_prompt

        self.selected_actions = []

        self.steps = 0

        # Evaluate the initial prompt.
        self.current_accuracy = self.evaluator.evaluate(
            self.prompt
        )

        observation = self._get_observation()

        info = {
            "prompt": self.prompt,
            "accuracy": self.current_accuracy,
            "selected_actions": [],
        }

        return observation, info

    def step(self, action):
        """
        Apply an action and transition to the next state.

        Parameters
        ----------
        action : int
            Selected prompt transformation or STOP action.

        Returns
        -------
        observation : np.ndarray
            New state.

        reward : float
            Reward associated with the selected action.

        terminated : bool
            Whether the episode has ended naturally.

        truncated : bool
            Whether the episode was truncated.

        info : dict
            Additional transition information.
        """

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        # sTOP action
        if action == self.stop_action:

            observation = self._get_observation()

            info = {
                "prompt": self.prompt,
                "accuracy": self.current_accuracy,
                "selected_actions": (
                    self.selected_actions.copy()
                ),
                "last_action": action,
            }

            return (
                observation,
                0.0,
                True,
                False,
                info,
            )

        # prevent duplicate transformations
        if action in self.selected_actions:

            observation = self._get_observation()

            info = {
                "prompt": self.prompt,
                "accuracy": self.current_accuracy,
                "selected_actions": (
                    self.selected_actions.copy()
                ),
                "last_action": action,
                "invalid_action": True,
            }

            return (
                observation,
                -1.0,
                False,
                False,
                info,
            )

        # current performance

        old_accuracy = self.current_accuracy

        # apply transformation

        self.prompt = apply_transformation(
            self.prompt,
            action,
        )

        self.selected_actions.append(action)

        self.steps += 1

        # evaluate new prompt

        self.current_accuracy = self.evaluator.evaluate(
            self.prompt
        )

        # reward

        reward = (
            self.current_accuracy
            - old_accuracy
        )

        # episode termination

        terminated = (
            self.steps >= self.max_steps
        )

        truncated = False

        observation = self._get_observation()

        info = {
            "prompt": self.prompt,
            "accuracy": self.current_accuracy,
            "previous_accuracy": old_accuracy,
            "selected_actions": (
                self.selected_actions.copy()
            ),
            "last_action": action,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_observation(self):
        """
        Build the numerical representation of the current state.

        Returns
        -------
        np.ndarray
            Binary vector indicating which transformations have
            already been selected.
        """

        observation = np.zeros(
            self.n_actions,
            dtype=np.float32,
        )

        for action in self.selected_actions:
            observation[action] = 1.0

        return observation

    def render(self):
        """
        Display the current state of the environment.
        """

        print("=" * 60)
        print(f"Step: {self.steps}")
        print(
            f"Accuracy: {self.current_accuracy:.4f}"
        )
        print(
            f"Selected actions: "
            f"{self.selected_actions}"
        )

        print("\nCurrent prompt:")
        print(self.prompt)

        print("=" * 60)