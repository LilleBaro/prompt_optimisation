import gymnasium as gym
import numpy as np

from prompts.transformations import apply_transformation


class PromptOptimizationEnv(gym.Env):
    """
    Gymnasium environment for prompt optimization.

    The agent sequentially selects prompt transformations.
    The LLM remains frozen and is evaluated after each transformation.

    Parameters
    ----------
    evaluator : GSM8KEvaluator
        Evaluator used to measure prompt performance.

    base_prompt : str
        Initial prompt before any transformation.

    max_steps : int, default=5
        Maximum number of transformations per episode.

    final_reward_coef : float, default=0.5
        Weight applied to the final improvement relative to
        the base prompt.
    """

    def __init__(
        self,
        evaluator,
        base_prompt,
        max_steps=5,
        final_reward_coef=0.5,
    ):
        super().__init__()

        self.evaluator = evaluator
        self.base_prompt = base_prompt
        self.max_steps = max_steps
        self.final_reward_coef = final_reward_coef

        # Number of available transformations
        self.action_dim = 8

        # ---------------------------------------------------------
        # Action space
        # ---------------------------------------------------------

        self.action_space = gym.spaces.Discrete(
            self.action_dim
        )

        # ---------------------------------------------------------
        # Observation space
        # ---------------------------------------------------------

        # Current accuracy
        # Previous accuracy
        # Step progress
        # Number of transformations already selected
        #
        # Plus one binary feature per action indicating whether
        # the transformation has already been selected.
        #
        # Total:
        # 4 + 8 = 12
        # ---------------------------------------------------------

        self.observation_dim = 12

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Episode state
        # ---------------------------------------------------------

        self.current_prompt = None
        self.current_accuracy = 0.0
        self.base_accuracy = 0.0
        self.previous_accuracy = 0.0

        self.step_count = 0

        self.selected_actions = []

        self.used_actions = set()

    # =============================================================
    # OBSERVATION
    # =============================================================

    def _get_observation(self):
        """
        Build the current environment observation.

        Returns
        -------
        np.ndarray
            Current environment state.
        """

        step_progress = (
            self.step_count / self.max_steps
        )

        num_selected = (
            len(self.selected_actions)
            / self.action_dim
        )

        used_actions = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        for action in self.used_actions:
            used_actions[action] = 1.0

        observation = np.concatenate(
            [
                np.array(
                    [
                        self.current_accuracy,
                        self.previous_accuracy,
                        step_progress,
                        num_selected,
                    ],
                    dtype=np.float32,
                ),
                used_actions,
            ]
        )

        return observation.astype(
            np.float32
        )

    # =============================================================
    # ACTION MASK
    # =============================================================

    def get_action_mask(self):
        """
        Return the mask of currently available actions.

        Returns
        -------
        np.ndarray
            Boolean mask where True means that the action
            can still be selected.
        """

        mask = np.ones(
            self.action_dim,
            dtype=bool,
        )

        for action in self.used_actions:
            mask[action] = False

        return mask

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
            Initial observation.

        info : dict
            Initial environment information.
        """

        super().reset(seed=seed)

        self.current_prompt = (
            self.base_prompt
        )

        # ---------------------------------------------------------
        # Evaluate base prompt
        # ---------------------------------------------------------

        self.base_accuracy = (
            self.evaluator.evaluate(
                self.base_prompt
            )
        )

        self.current_accuracy = (
            self.base_accuracy
        )

        self.previous_accuracy = (
            self.base_accuracy
        )

        self.step_count = 0

        self.selected_actions = []

        self.used_actions = set()

        observation = (
            self._get_observation()
        )

        info = {
            "prompt": self.current_prompt,
            "accuracy": self.current_accuracy,
            "base_accuracy": self.base_accuracy,
            "actions": self.selected_actions,
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
            Transformation ID.

        Returns
        -------
        observation : np.ndarray
            New environment state.

        reward : float
            Reward obtained after the transformation.

        terminated : bool
            Whether the episode naturally ended.

        truncated : bool
            Whether the episode was truncated.

        info : dict
            Environment information.
        """

        # ---------------------------------------------------------
        # Validate action
        # ---------------------------------------------------------

        if action in self.used_actions:
            raise ValueError(
                f"Action {action} has already been selected."
            )

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        # ---------------------------------------------------------
        # Previous state
        # ---------------------------------------------------------

        self.previous_accuracy = (
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

        self.used_actions.add(action)

        self.selected_actions.append(
            action
        )

        self.step_count += 1

        # ---------------------------------------------------------
        # Evaluate transformed prompt
        # ---------------------------------------------------------

        self.current_accuracy = (
            self.evaluator.evaluate(
                self.current_prompt
            )
        )

        # ---------------------------------------------------------
        # Local reward
        # ---------------------------------------------------------

        reward = (
            self.current_accuracy
            - self.previous_accuracy
        )

        # ---------------------------------------------------------
        # Episode termination
        # ---------------------------------------------------------

        terminated = (
            self.step_count
            >= self.max_steps
        )

        truncated = False

        # ---------------------------------------------------------
        # Final reward
        # ---------------------------------------------------------

        final_reward = 0.0

        if terminated:

            final_improvement = (
                self.current_accuracy
                - self.base_accuracy
            )

            final_reward = (
                self.final_reward_coef
                * final_improvement
            )

            reward += final_reward

        # ---------------------------------------------------------
        # Observation
        # ---------------------------------------------------------

        observation = (
            self._get_observation()
        )

        # ---------------------------------------------------------
        # Information
        # ---------------------------------------------------------

        info = {
            "prompt": self.current_prompt,
            "accuracy": self.current_accuracy,
            "base_accuracy": self.base_accuracy,
            "improvement": (
                self.current_accuracy
                - self.base_accuracy
            ),
            "previous_accuracy": (
                self.previous_accuracy
            ),
            "local_reward": (
                self.current_accuracy
                - self.previous_accuracy
            ),
            "final_reward": final_reward,
            "actions": list(
                self.selected_actions
            ),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )