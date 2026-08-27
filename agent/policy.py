import torch
import torch.nn as nn

from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Actor-Critic neural network used by the RL agent.

    The network receives the current environment state and produces:

    - action logits for the policy;
    - a state-value estimate for the critic.

    The network also supports action masking, allowing the environment
    to indicate which actions are currently available.

    Parameters
    ----------
    observation_dim : int
        Dimension of the environment observation.

    action_dim : int
        Number of available actions.

    hidden_dim : int, default=128
        Number of neurons in the hidden layers.
    """

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=128,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Shared representation
        # ---------------------------------------------------------

        self.shared_network = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Actor
        # ---------------------------------------------------------

        self.policy_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        # ---------------------------------------------------------
        # Critic
        # ---------------------------------------------------------

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(self, state):
        """
        Compute policy logits and state value.

        Parameters
        ----------
        state : torch.Tensor
            Current environment state.

        Returns
        -------
        logits : torch.Tensor
            Unnormalized action scores.

        value : torch.Tensor
            Estimated state value.
        """

        features = self.shared_network(state)

        logits = self.policy_head(features)

        value = self.value_head(features)

        return logits, value

    # =============================================================
    # ACTION MASK
    # =============================================================

    @staticmethod
    def apply_action_mask(logits, action_mask):
        """
        Mask unavailable actions.

        Parameters
        ----------
        logits : torch.Tensor
            Action logits produced by the policy.

        action_mask : torch.Tensor
            Boolean tensor where:

            True  = action available
            False = action unavailable

        Returns
        -------
        torch.Tensor
            Masked action logits.
        """

        if action_mask is None:
            return logits

        # Convert mask to boolean if necessary.
        action_mask = action_mask.bool()

        # Invalid actions receive a very negative logit.
        masked_logits = logits.masked_fill(
            ~action_mask,
            torch.finfo(logits.dtype).min,
        )

        return masked_logits

    # =============================================================
    # DISTRIBUTION
    # =============================================================

    def get_distribution(
        self,
        state,
        action_mask=None,
    ):
        """
        Build the action probability distribution.

        Parameters
        ----------
        state : torch.Tensor
            Current environment state.

        action_mask : torch.Tensor or None
            Boolean mask indicating available actions.

        Returns
        -------
        Categorical
            Probability distribution over available actions.
        """

        logits, _ = self.forward(state)

        logits = self.apply_action_mask(
            logits,
            action_mask,
        )

        return Categorical(
            logits=logits
        )

    # =============================================================
    # ACTION SELECTION
    # =============================================================

    def get_action(
        self,
        state,
        action_mask=None,
        deterministic=False,
    ):
        """
        Select an action according to the current policy.

        Parameters
        ----------
        state : torch.Tensor
            Current environment state.

        action_mask : torch.Tensor or None
            Boolean mask indicating available actions.

        deterministic : bool, default=False
            If True, select the action with the highest probability.

        Returns
        -------
        action : torch.Tensor
            Selected action.

        log_probability : torch.Tensor
            Log probability of the selected action.

        value : torch.Tensor
            Estimated state value.
        """

        logits, value = self.forward(state)

        logits = self.apply_action_mask(
            logits,
            action_mask,
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            action = torch.argmax(
                logits,
                dim=-1,
            )
        else:
            action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action,
            log_probability,
            value.squeeze(-1),
        )

    # =============================================================
    # EVALUATE ACTIONS
    # =============================================================

    def evaluate_actions(
        self,
        state,
        actions,
        action_mask=None,
    ):
        """
        Evaluate actions under the current policy.

        This method is used during PPO optimization.

        Parameters
        ----------
        state : torch.Tensor
            Environment states.

        actions : torch.Tensor
            Actions previously selected.

        action_mask : torch.Tensor or None
            Boolean mask indicating available actions for each state.

        Returns
        -------
        log_probabilities : torch.Tensor
            Log probabilities of the selected actions.

        entropy : torch.Tensor
            Policy entropy.

        values : torch.Tensor
            Estimated state values.
        """

        logits, values = self.forward(state)

        logits = self.apply_action_mask(
            logits,
            action_mask,
        )

        distribution = Categorical(
            logits=logits
        )

        log_probabilities = (
            distribution.log_prob(actions)
        )

        entropy = distribution.entropy()

        return (
            log_probabilities,
            entropy,
            values.squeeze(-1),
        )