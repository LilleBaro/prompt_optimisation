import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.policy import PolicyNetwork


class PPO:
    """
    Proximal Policy Optimization agent for prompt optimization.

    The agent learns to select prompt transformations that improve
    the performance of a frozen LLM on GSM8K.

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
        Number of optimization epochs per trajectory.

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
    # ACTION MASK
    # =============================================================

    def _get_action_mask(self):
        """
        Get the current action mask from the environment.

        Returns
        -------
        torch.Tensor
            Boolean action mask.
        """

        mask = self.env.get_action_mask()

        return torch.tensor(
            mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

    # =============================================================
    # ACTION SELECTION
    # =============================================================

    def select_action(
        self,
        state,
        action_mask=None,
        deterministic=False,
    ):
        """
        Select an action using the current policy.

        Parameters
        ----------
        state : np.ndarray
            Current environment state.

        action_mask : np.ndarray or None
            Boolean mask of available actions.

        deterministic : bool, default=False
            Whether to select the most probable valid action.

        Returns
        -------
        action : int
            Selected action.

        log_probability : float
            Log probability of the selected action.

        value : float
            Estimated state value.
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        if action_mask is not None:

            action_mask = torch.tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)

        with torch.no_grad():

            (
                action,
                log_probability,
                value,
            ) = self.policy.get_action(
                state_tensor,
                action_mask=action_mask,
                deterministic=deterministic,
            )

        return (
            action.item(),
            log_probability.item(),
            value.item(),
        )

    # =============================================================
    # TRAJECTORY COLLECTION
    # =============================================================

    def collect_episode(self):
        """
        Collect one complete trajectory.

        Returns
        -------
        trajectory : dict
            States, actions, rewards, log probabilities,
            values, masks and terminal flags.
        """

        state, info = self.env.reset()

        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
            "action_masks": [],
        }

        done = False

        while not done:

            # -----------------------------------------------------
            # Get available actions
            # -----------------------------------------------------

            action_mask = self.env.get_action_mask()

            # -----------------------------------------------------
            # Select action
            # -----------------------------------------------------

            (
                action,
                log_prob,
                value,
            ) = self.select_action(
                state,
                action_mask=action_mask,
            )

            # -----------------------------------------------------
            # Environment transition
            # -----------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            done = (
                terminated
                or truncated
            )

            # -----------------------------------------------------
            # Store transition
            # -----------------------------------------------------

            trajectory["states"].append(state)

            trajectory["actions"].append(action)

            trajectory["rewards"].append(reward)

            trajectory["log_probs"].append(log_prob)

            trajectory["values"].append(value)

            trajectory["dones"].append(done)

            trajectory["action_masks"].append(
                action_mask
            )

            state = next_state

        return trajectory

    # =============================================================
    # ADVANTAGE ESTIMATION
    # =============================================================

    def compute_gae(self, trajectory):
        """
        Compute Generalized Advantage Estimation.

        Parameters
        ----------
        trajectory : dict
            Collected trajectory.

        Returns
        -------
        advantages : np.ndarray
            Estimated advantages.

        returns : np.ndarray
            Estimated returns.
        """

        rewards = np.asarray(
            trajectory["rewards"],
            dtype=np.float32,
        )

        values = np.asarray(
            trajectory["values"],
            dtype=np.float32,
        )

        dones = np.asarray(
            trajectory["dones"],
            dtype=np.float32,
        )

        advantages = np.zeros_like(
            rewards
        )

        gae = 0.0

        for t in reversed(
            range(len(rewards))
        ):

            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            non_terminal = (
                1.0 - dones[t]
            )

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

        returns = (
            advantages + values
        )

        return advantages, returns

    # =============================================================
    # PPO UPDATE
    # =============================================================

    def update(self, trajectory):
        """
        Update the policy using PPO.

        Parameters
        ----------
        trajectory : dict
            Collected trajectory.

        Returns
        -------
        dict
            Training statistics.
        """

        advantages, returns = (
            self.compute_gae(
                trajectory
            )
        )

        # ---------------------------------------------------------
        # Convert trajectory to tensors
        # ---------------------------------------------------------

        states = torch.tensor(
            np.asarray(
                trajectory["states"]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            trajectory["actions"],
            dtype=torch.long,
            device=self.device,
        )

        old_log_probs = torch.tensor(
            trajectory["log_probs"],
            dtype=torch.float32,
            device=self.device,
        )

        action_masks = torch.tensor(
            np.asarray(
                trajectory["action_masks"]
            ),
            dtype=torch.bool,
            device=self.device,
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Advantage normalization
        # ---------------------------------------------------------

        if len(advantages) > 1:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # ---------------------------------------------------------
        # PPO optimization epochs
        # ---------------------------------------------------------

        for _ in range(
            self.update_epochs
        ):

            (
                new_log_probs,
                entropy,
                values,
            ) = self.policy.evaluate_actions(
                states,
                actions,
                action_mask=action_masks,
            )

            # -----------------------------------------------------
            # Probability ratio
            # -----------------------------------------------------

            ratios = torch.exp(
                new_log_probs
                - old_log_probs
            )

            # -----------------------------------------------------
            # Surrogate objectives
            # -----------------------------------------------------

            unclipped_objective = (
                ratios * advantages
            )

            clipped_ratios = torch.clamp(
                ratios,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )

            clipped_objective = (
                clipped_ratios
                * advantages
            )

            policy_loss = -torch.min(
                unclipped_objective,
                clipped_objective,
            ).mean()

            # -----------------------------------------------------
            # Value loss
            # -----------------------------------------------------

            value_loss = (
                nn.functional.mse_loss(
                    values,
                    returns,
                )
            )

            # -----------------------------------------------------
            # Entropy
            # -----------------------------------------------------

            entropy_loss = (
                entropy.mean()
            )

            # -----------------------------------------------------
            # Total loss
            # -----------------------------------------------------

            loss = (
                policy_loss
                + self.value_coef
                * value_loss
                - self.entropy_coef
                * entropy_loss
            )

            # -----------------------------------------------------
            # Gradient update
            # -----------------------------------------------------

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

        return {
            "policy_loss": (
                total_policy_loss
                / self.update_epochs
            ),
            "value_loss": (
                total_value_loss
                / self.update_epochs
            ),
            "entropy": (
                total_entropy
                / self.update_epochs
            ),
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
            Number of training episodes.

        Returns
        -------
        history : list[dict]
            Training statistics for each episode.
        """

        history = []

        for episode in range(
            n_episodes
        ):

            trajectory = (
                self.collect_episode()
            )

            update_info = self.update(
                trajectory
            )

            episode_reward = sum(
                trajectory["rewards"]
            )

            final_accuracy = (
                self.env.current_accuracy
            )

            history.append({
                "episode": episode + 1,
                "reward": episode_reward,
                "accuracy": final_accuracy,
                "steps": len(
                    trajectory["rewards"]
                ),
                **update_info,
            })

            if (
                (episode + 1) % 10 == 0
            ):
                print(
                    f"Episode "
                    f"{episode + 1}/{n_episodes} | "
                    f"Reward: "
                    f"{episode_reward:.4f} | "
                    f"Accuracy: "
                    f"{final_accuracy:.4f} | "
                    f"Policy Loss: "
                    f"{update_info['policy_loss']:.4f}"
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

        state, info = (
            self.env.reset()
        )

        selected_actions = []

        done = False

        while not done:

            action_mask = (
                self.env.get_action_mask()
            )

            (
                action,
                _,
                _,
            ) = self.select_action(
                state,
                action_mask=action_mask,
                deterministic=True,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            selected_actions.append(
                action
            )

            done = (
                terminated
                or truncated
            )

            state = next_state

        return {
            "prompt": info["prompt"],
            "accuracy": info["accuracy"],
            "actions": selected_actions,
        }