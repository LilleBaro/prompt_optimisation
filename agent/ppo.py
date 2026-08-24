import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from agent.policy import PolicyNetwork
from agent.rollout_buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization agent.

    Parameters
    ----------
    env : gymnasium.Env
        Prompt optimization environment.

    learning_rate : float, default=3e-4
        Learning rate of the optimizer.

    gamma : float, default=0.99
        Discount factor.

    gae_lambda : float, default=0.95
        GAE parameter.

    clip_epsilon : float, default=0.2
        PPO clipping parameter.

    value_coef : float, default=0.5
        Weight of the value loss.

    entropy_coef : float, default=0.01
        Weight of the entropy bonus.

    update_epochs : int, default=4
        Number of optimization epochs.

    minibatch_size : int, default=32
        Number of transitions used in each PPO mini-batch.

    rollout_episodes : int, default=8
        Number of episodes collected before each PPO update.

    hidden_dim : int, default=128
        Number of neurons in the policy hidden layers.

    device : str or None, default=None
        Device used for training.
    """

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=4,
        minibatch_size=32,
        rollout_episodes=8,
        hidden_dim=128,
        device=None,
    ):
        self.env = env

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_episodes = rollout_episodes

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # ---------------------------------------------------------
        # Environment dimensions
        # ---------------------------------------------------------

        observation_dim = (
            env.observation_space.shape[0]
        )

        action_dim = (
            env.action_space.n
        )

        # ---------------------------------------------------------
        # Policy
        # ---------------------------------------------------------

        self.policy = PolicyNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )

    # =============================================================
    # ACTION SELECTION
    # =============================================================

    def select_action(
        self,
        state,
        deterministic=False,
    ):
        """
        Select an action using the current policy.
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            action, log_probability, value = (
                self.policy.get_action(
                    state_tensor,
                    deterministic=deterministic,
                )
            )

        return (
            action.item(),
            log_probability.item(),
            value.item(),
        )

    # =============================================================
    # ONE EPISODE
    # =============================================================

    def collect_episode(self, buffer):
        """
        Collect one complete trajectory and add it to a buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Buffer receiving the collected transitions.

        Returns
        -------
        dict
            Episode statistics.
        """

        state, info = self.env.reset()

        done = False
        episode_reward = 0.0
        steps = 0

        while not done:

            action, log_prob, value = (
                self.select_action(state)
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            done = terminated or truncated

            buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            episode_reward += reward
            steps += 1

            state = next_state

        return {
            "reward": episode_reward,
            "steps": steps,
            "info": info,
        }

    # =============================================================
    # ROLLOUT COLLECTION
    # =============================================================

    def collect_rollouts(self):
        """
        Collect several episodes before a PPO update.

        Returns
        -------
        buffer : RolloutBuffer
            Collected transitions.

        episode_statistics : list[dict]
            Statistics for each episode.
        """

        buffer = RolloutBuffer(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        episode_statistics = []

        for _ in range(self.rollout_episodes):

            statistics = self.collect_episode(
                buffer
            )

            episode_statistics.append(
                statistics
            )

        buffer.compute_gae()

        return (
            buffer,
            episode_statistics,
        )

    # =============================================================
    # PPO UPDATE
    # =============================================================

    def update(self, buffer):
        """
        Update the policy using collected rollouts.

        Parameters
        ----------
        buffer : RolloutBuffer
            Collected rollout data.

        Returns
        -------
        dict
            Training statistics.
        """

        data = buffer.get_data()

        states = torch.tensor(
            data["states"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            data["actions"],
            dtype=torch.long,
            device=self.device,
        )

        old_log_probs = torch.tensor(
            data["old_log_probs"],
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.tensor(
            data["returns"],
            dtype=torch.float32,
            device=self.device,
        )

        advantages = torch.tensor(
            data["advantages"],
            dtype=torch.float32,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Advantage normalization
        # ---------------------------------------------------------

        if len(advantages) > 1:

            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False)
                + 1e-8
            )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        update_count = 0

        n_samples = len(states)

        # ---------------------------------------------------------
        # PPO epochs
        # ---------------------------------------------------------

        for _ in range(self.update_epochs):

            indices = torch.randperm(
                n_samples,
                device=self.device,
            )

            # -----------------------------------------------------
            # Mini-batches
            # -----------------------------------------------------

            for start in range(
                0,
                n_samples,
                self.minibatch_size,
            ):

                batch_indices = indices[
                    start:start + self.minibatch_size
                ]

                batch_states = states[
                    batch_indices
                ]

                batch_actions = actions[
                    batch_indices
                ]

                batch_old_log_probs = old_log_probs[
                    batch_indices
                ]

                batch_returns = returns[
                    batch_indices
                ]

                batch_advantages = advantages[
                    batch_indices
                ]

                # -------------------------------------------------
                # Evaluate current policy
                # -------------------------------------------------

                (
                    new_log_probs,
                    entropy,
                    values,
                ) = self.policy.evaluate_actions(
                    batch_states,
                    batch_actions,
                )

                # -------------------------------------------------
                # Probability ratio
                # -------------------------------------------------

                ratios = torch.exp(
                    new_log_probs
                    - batch_old_log_probs
                )

                # -------------------------------------------------
                # PPO clipped objective
                # -------------------------------------------------

                unclipped_objective = (
                    ratios * batch_advantages
                )

                clipped_ratios = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )

                clipped_objective = (
                    clipped_ratios
                    * batch_advantages
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                # -------------------------------------------------
                # Value loss
                # -------------------------------------------------

                value_loss = nn.functional.mse_loss(
                    values,
                    batch_returns,
                )

                # -------------------------------------------------
                # Entropy
                # -------------------------------------------------

                entropy_loss = entropy.mean()

                # -------------------------------------------------
                # Total loss
                # -------------------------------------------------

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # -------------------------------------------------
                # Gradient update
                # -------------------------------------------------

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=0.5,
                )

                self.optimizer.step()

                total_policy_loss += (
                    policy_loss.item()
                )

                total_value_loss += (
                    value_loss.item()
                )

                total_entropy += (
                    entropy_loss.item()
                )

                update_count += 1

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        return {
            "policy_loss": (
                total_policy_loss
                / update_count
            ),
            "value_loss": (
                total_value_loss
                / update_count
            ),
            "entropy": (
                total_entropy
                / update_count
            ),
            "buffer_size": len(buffer),
        }

    # =============================================================
    # TRAINING
    # =============================================================

    def train(self, n_episodes=100):
        """
        Train the PPO agent.

        Parameters
        ----------
        n_episodes : int, default=100
            Total number of environment episodes.

        Returns
        -------
        history : list[dict]
            Training statistics.
        """

        history = []

        completed_episodes = 0
        update_index = 0

        while completed_episodes < n_episodes:

            remaining_episodes = (
                n_episodes
                - completed_episodes
            )

            current_rollout_episodes = min(
                self.rollout_episodes,
                remaining_episodes,
            )

            # -----------------------------------------------------
            # Temporarily collect the required number of episodes
            # -----------------------------------------------------

            original_rollout_episodes = (
                self.rollout_episodes
            )

            self.rollout_episodes = (
                current_rollout_episodes
            )

            buffer, episode_statistics = (
                self.collect_rollouts()
            )

            self.rollout_episodes = (
                original_rollout_episodes
            )

            # -----------------------------------------------------
            # PPO update
            # -----------------------------------------------------

            update_info = self.update(
                buffer
            )

            # -----------------------------------------------------
            # Store episode statistics
            # -----------------------------------------------------

            for statistics in episode_statistics:

                completed_episodes += 1

                history.append({
                    "episode": completed_episodes,
                    "reward": statistics["reward"],
                    "steps": statistics["steps"],
                    "buffer_size": len(buffer),
                    **update_info,
                })

            update_index += 1

            # -----------------------------------------------------
            # Logging
            # -----------------------------------------------------

            mean_reward = np.mean([
                item["reward"]
                for item in episode_statistics
            ])

            print(
                f"Update {update_index} | "
                f"Episodes "
                f"{completed_episodes}/{n_episodes} | "
                f"Mean Reward: "
                f"{mean_reward:.4f} | "
                f"Policy Loss: "
                f"{update_info['policy_loss']:.4f} | "
                f"Value Loss: "
                f"{update_info['value_loss']:.4f}"
            )

        return history

    # =============================================================
    # EVALUATION
    # =============================================================

    def evaluate(self):
        """
        Evaluate the learned policy deterministically.

        Returns
        -------
        dict
            Final prompt, accuracy and selected actions.
        """

        state, info = self.env.reset()

        selected_actions = []

        done = False

        while not done:

            action, _, _ = self.select_action(
                state,
                deterministic=True,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            selected_actions.append(action)

            done = terminated or truncated

            state = next_state

        return {
            "prompt": info["prompt"],
            "accuracy": info["accuracy"],
            "actions": selected_actions,
        }