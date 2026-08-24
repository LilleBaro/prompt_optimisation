import numpy as np


class RolloutBuffer:
    """
    Buffer used to store transitions collected during PPO rollouts.

    The buffer can contain transitions from multiple episodes.

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor.

    gae_lambda : float, default=0.95
        GAE parameter.
    """

    def __init__(
        self,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clear()

    def clear(self):
        """Remove all stored transitions."""

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.advantages = None
        self.returns = None

    def add(
        self,
        state,
        action,
        reward,
        done,
        log_prob,
        value,
    ):
        """
        Add a transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            Environment state.

        action : int
            Selected action.

        reward : float
            Received reward.

        done : bool
            Whether the transition terminated the episode.

        log_prob : float
            Log probability of the selected action under
            the old policy.

        value : float
            Value estimate produced by the old policy.
        """

        self.states.append(
            np.asarray(state, dtype=np.float32)
        )

        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self):
        """
        Compute advantages and returns using GAE.

        Returns
        -------
        advantages : np.ndarray
            Estimated advantages.

        returns : np.ndarray
            Estimated returns.
        """

        rewards = np.asarray(
            self.rewards,
            dtype=np.float32,
        )

        values = np.asarray(
            self.values,
            dtype=np.float32,
        )

        dones = np.asarray(
            self.dones,
            dtype=np.float32,
        )

        advantages = np.zeros_like(
            rewards,
            dtype=np.float32,
        )

        gae = 0.0

        for t in reversed(range(len(rewards))):

            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma
                * next_value
                * non_terminal
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * gae
            )

            advantages[t] = gae

        returns = advantages + values

        self.advantages = advantages
        self.returns = returns

        return advantages, returns

    def get_data(self):
        """
        Return all collected transitions.

        Returns
        -------
        dict
            Dictionary containing rollout data.
        """

        if self.advantages is None:
            raise RuntimeError(
                "Call compute_gae() before get_data()."
            )

        return {
            "states": np.asarray(
                self.states,
                dtype=np.float32,
            ),
            "actions": np.asarray(
                self.actions,
                dtype=np.int64,
            ),
            "old_log_probs": np.asarray(
                self.log_probs,
                dtype=np.float32,
            ),
            "returns": self.returns,
            "advantages": self.advantages,
        }

    def __len__(self):
        """Return the number of stored transitions."""

        return len(self.rewards)