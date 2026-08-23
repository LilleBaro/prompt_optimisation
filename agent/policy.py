import torch 
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Actor-Critic neural network used by the RL agent.
    
    The network receive the current environment state and produces:
    
    - action logits for the policy;
    - a state-value estimate for the critic.

    Parameters
    ----------
    observation_dim : int
        Dimension of the environment observation.

    action_dim : int
        Number of available actions.
    
    hidden_dim : int:
        Number of neurons in the hidden layers.
    
    """

    def __init__(
            self, 
            observation_dim,
            action_dim,
            hidden_dim=128
    ):
        super().__init__()

        # shared representation
        self.shared_network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor
        self.policy_head = nn.Linear(
            hidden_dim,
            action_dim
        )

        # critic
        self.value_head = nn.Linear(
            hidden_dim, 
            1
        )

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
            Estimated action scores.
        """

        features = self.shared_network(state)
        logits = self.policy_head(features)
        value = self.value_head(features)

        return logits, value

    def get_distribution(self, state):
        """
        Build the action probability distribution.
        
        Parameters
        ----------
        state : torch.Tensor
            Current environment state.
        
        Returns 
        -------
        torch.distrubutions.Categorical
            Categorical distribution over actions.
        """

        logits, _ = self.forward(state)
        return Categorical(logits=logits)

    def get_action(self, state, deterministic=False):
        """
        Select an action according to the current polocy.
        
        Parameters
        ----------
        action : torch.Tensor
            Selected action.
        
        log_probability : torch.Tensor
            Log probability og the selected action.
        
        value : torch.Tensor
            Estimated state value.
        """

        logits, value = self.forward(state)

        distributions = Categorical(
            logits=logits
        )
        if deterministic:
            action = torch.argmax(
                logits,
                dim=-1
            )
        else:
            action = distributions.sample()
        log_probability = distributions.log_prob(
            action
        )
        return (
            action,
            log_probability,
            value.squeeze(-1)
        )

    def evaluate_actions(self, state, actions):
        """
        Evaluate actions under the current policy.
        
        This method is particularly useful for PPO.
        
        Parameters
        ----------
        state : torch.Tensor
            Environment state.
        
        actions : torch.Tensor
            Actions taken in those states.

        Returns
        -------
        log_probabilities : torch.Tensor
            Log probabilities of the selected actions
        
        entropy : torch.Tensor
            Policy entropy.
        
        value : torch.Tensor
            Extimated state values.
        """

        logits, values = self.forward(state)

        distribution = Categorical(
            logits=logits
        )
        log_probabilities = distribution.log_prob(
            actions
        )
        entropy = distribution.entropy()

        return (
            log_probabilities,
            entropy,
            values.squeeze(-1),
        )
    