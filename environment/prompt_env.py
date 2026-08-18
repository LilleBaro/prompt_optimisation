import gymnasium as gym
import numpy as np

from prompts.base_prompt import BASE_PROMPT
from prompts.transformations import PROMPT_TRANSFORMATIONS, apply_transformation

class PromptOptimizationEnv(gym.Env):
    """
    Gymnasium environment for prompt optimization using reinforcement leanrning.

    The agent does not generate answer to GSMK8K problems. Instead, it learns 
    to select prompt transformations that improve the performances of a frozen LLM

    Params
    ------
    evaluator : object
        Object responsible for evaluating a prompt on GSM8K and returning its accuracy.
    max steps : int, default=5
        Maximum number of prompt transformations that can be applied during on episode
    initial prompt: str, default=BASE_PROMPT
        Initial prompt from which the agent starts each episode.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
            self, 
            evaluator,
            max_steps=5,
            initial_prompt=BASE_PROMPT
    ):
        super().__init__()
        self.evaluator = evaluator
        self.max_steps = max_steps
        self.initial_prompt = initial_prompt

        # number of available prompt transformations
        self.n_actions = len(PROMPT_TRANSFORMATIONS)
        # discrete action space:
        # each integer corresponds to one prompt transformation
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # the actual prompt is textual, so it can't directly be represneted 
        # by a standard numerical gymnasium observation space.
        # We temporarily expose the number of transformation already used
        # a richer state representation will be introduced later.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.n_actions,),
            dtype=np.float32
        )
        self.prompt = None
        self.current_accuracy = None
        self.selected_actions = None
        self.steps = None
    def reset(self, seed=None, options=None):
        """
        Reset the environement to its initial state.

        Returns
        ------
        observation : np.ndarray
            Initial state representation
        info : dict
            Additional information about the environment state. 
        """
        super().reset(seed=seed)
        self.prompt = self.initial_prompt
        self.selected_actions = []
        self.steps = 0

        # Evaluates the initial prompt
        self.current_accuracy = self.evaluator.evaluate(
            self.prompt
        ) 
        observation = self.get_observation()

        info = {
            "prompt":self.prompt,
            "accuracy": self.current_accuracy,
            "selected_actions": self.selected_actions.copy()
        }
        return observation, info

    def step(self, action):
        """
        Apply a prompt transformation and evaluate the resulting prompt

        Parameters
        ------
        action : int 
            ID of the prompt transformation to apply.
        
        Returns
        ------
        observation : np.ndarray
            New state representation 

        reward : float
            Improvement on GSM8K accuracy.

        terminated : bool
            Wheter the episode has naturally terminated.

        truncated : bool
            Whether the episode was truncated because the maximum number pf steps was reached

        info : dict
            Additional information about the transition.
        """
        action = int(action)
        # validate the action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalide action: {action}")
        # prevent duplicated transformations 
        if action in self.selected_actions:
            reward = -1.0
            observation = self.get_observation()
            info = {
                "prompt": self.prompt,
                "accuracy": self.current_accuracy,
                "selected_actions": self.selected_actions.copy(),
                "invalid_actions": True
            }
            return (
                observation,
                reward,
                False,
                False,
                info
            )
        # apply transformation
        old_accuracy = self.current_accuracy
        self.prompt = apply_transformation(
            self.prompt,
            action
        )
        self.selected_actions.append(action)
        self.steps += 1
        # evaluate new prompt
        self.current_accuracy = self.evaluator.evaluate(
            self.prompt
        )
        # compute reward 
        reward = self.current_accuracy - old_accuracy
        # determine termination
        terminated = (
            self.steps >= self.max_steps
            or len(self.selected_actions) >= self.n_actions
        )
        truncated = False
        observation = self._get_observation()
        info = {
            "prompt":self.prompt,
            "accuracy": self.current_accuracy,
            "previous_accuracy": old_accuracy,
            "selected_action": self.selected_actions.copy(),
            "last_action": action
        }
        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    def _get_observation(self):
        """
        Build the numerical represen tation of the current state

        Returns
        ------
        np.ndarray
            Binary vector indicating which transformations have already
            been selected.
        """
        observation = np.zeros(
            self.n_actions,
            dtype=np.float32
        )
        for action in self.selected_actions:
            observation[action] = 1.0
        return observation

    def render(self):
        """
        Display the current environment state.
        """
        print("="*60)
        print(f"Step: {self.steps}")
        print(f"Accuracy: {self.current_accuracy}")
        print(f"Selected actions: {self.selected_actions}")
        print(f"\nCurrent prompt: ")
        print(self.prompt)
        print("="*60)

