import gymnasium as gym
import numpy as np

from prompts.transformations import (
    PROMPT_TRANSFORMATIONS,
    apply_transformation,
)


class PromptOptimizationEnv(gym.Env):
    """
    Reinforcement Learning environment for prompt optimization.

    The agent learns to select prompt transformations that improve
    the performance of a frozen LLM on GSM8K.

    At each step, the agent:

    1. observes the current prompt state;
    2. selects an unused prompt transformation;
    3. evaluates the resulting prompt;
    4. receives a reward based on the improvement in accuracy.

    The reward is defined as:

        reward = current_accuracy - previous_accuracy

    Parameters
    ----------
    evaluator : object
        GSM8KEvaluator used to evaluate prompts.

    base_prompt : str
        Initial prompt before transformations.

    max_steps : int, default=5
        Maximum number of transformations per episode.
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        evaluator,
        base_prompt,
        max_steps=5,
    ):
        super().__init__()

        self.evaluator = evaluator
        self.base_prompt = base_prompt
        self.max_steps = max_steps

        # ---------------------------------------------------------
        # Transformations
        # ---------------------------------------------------------

        self.transformations = PROMPT_TRANSFORMATIONS

        self.action_space = gym.spaces.Discrete(
            len(self.transformations)
        )

        # ---------------------------------------------------------
        # Observation space
        # ---------------------------------------------------------

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    0.0,  # accuracy
                    0.0,  # normalized step
                    0.0,  # normalized prompt length
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Internal state
        # ---------------------------------------------------------

        self.current_prompt = None
        self.current_accuracy = None

        self.step_count = 0

        self.selected_actions = []

        self.used_actions = set()

    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset the environment.

        Returns
        -------
        observation : np.ndarray
            Initial state.

        info : dict
            Information about the initial prompt.
        """

        super().reset(seed=seed)

        self.current_prompt = self.base_prompt

        self.step_count = 0

        self.selected_actions = []

        self.used_actions = set()

        # ---------------------------------------------------------
        # Evaluate base prompt
        # ---------------------------------------------------------

        self.current_accuracy = (
            self.evaluator.evaluate(
                self.current_prompt
            )
        )

        observation = self._get_observation()

        info = {
            "prompt": self.current_prompt,
            "accuracy": self.current_accuracy,
            "available_actions": self.get_available_actions(),
        }

        return observation, info

    # =============================================================
    # STEP
    # =============================================================

    def step(self, action):
        """
        Apply a prompt transformation.

        Parameters
        ----------
        action : int
            ID of the transformation to apply.

        Returns
        -------
        observation : np.ndarray
            New state.

        reward : float
            Improvement in accuracy.

        terminated : bool
            Whether the episode naturally terminated.

        truncated : bool
            Whether the maximum number of steps was reached.

        info : dict
            Transition information.
        """

        # ---------------------------------------------------------
        # Validate action
        # ---------------------------------------------------------

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        if action in self.used_actions:
            raise ValueError(
                f"Action {action} has already been used."
            )

        # ---------------------------------------------------------
        # Previous accuracy
        # ---------------------------------------------------------

        previous_accuracy = (
            self.current_accuracy
        )

        # ---------------------------------------------------------
        # Apply transformation
        # ---------------------------------------------------------

        self.current_prompt = (
            apply_transformation(
                self.current_prompt,
                action,
            )
        )

        # ---------------------------------------------------------
        # Mark action as used
        # ---------------------------------------------------------

        self.used_actions.add(action)

        self.selected_actions.append(action)

        # ---------------------------------------------------------
        # Evaluate new prompt
        # ---------------------------------------------------------

        self.current_accuracy = (
            self.evaluator.evaluate(
                self.current_prompt
            )
        )

        # ---------------------------------------------------------
        # Differential reward
        # ---------------------------------------------------------

        reward = (
            self.current_accuracy
            - previous_accuracy
        )

        # ---------------------------------------------------------
        # Update step
        # ---------------------------------------------------------

        self.step_count += 1

        # ---------------------------------------------------------
        # Termination
        # ---------------------------------------------------------

        terminated = (
            len(self.used_actions)
            == len(self.transformations)
        )

        truncated = (
            self.step_count >= self.max_steps
            and not terminated
        )

        # ---------------------------------------------------------
        # Observation
        # ---------------------------------------------------------

        observation = self._get_observation()

        info = {
            "prompt": self.current_prompt,
            "accuracy": self.current_accuracy,
            "previous_accuracy": previous_accuracy,
            "improvement": reward,
            "action": action,
            "action_name": self.transformations[action]["name"],
            "step": self.step_count,
            "available_actions": self.get_available_actions(),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # =============================================================
    # AVAILABLE ACTIONS
    # =============================================================

    def get_available_actions(self):
        """
        Return transformations that have not yet been used.

        Returns
        -------
        list[int]
            IDs of available actions.
        """

        return [
            action
            for action in self.transformations
            if action not in self.used_actions
        ]

    # =============================================================
    # OBSERVATION
    # =============================================================

    def _get_observation(self):
        """
        Build the current normalized observation.

        Returns
        -------
        np.ndarray
            Current environment state.
        """

        normalized_step = (
            self.step_count
            / self.max_steps
        )

        normalized_length = min(
            len(self.current_prompt)
            / 1000.0,
            1.0,
        )

        return np.array(
            [
                self.current_accuracy,
                normalized_step,
                normalized_length,
            ],
            dtype=np.float32,
        )
    def get_action_mask(self):
        """
        Return a boolean mask indicating which actions are available.

        Returns
        -------
        np.ndarray
            Boolean action mask.
        """

        mask = np.zeros(
            self.action_space.n,
            dtype=bool,
        )

        for action in self.get_available_actions():
            mask[action] = True

        return mask