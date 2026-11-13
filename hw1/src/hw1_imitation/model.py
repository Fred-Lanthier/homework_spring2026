"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        
        super().__init__(state_dim, action_dim, chunk_size)

        layers_list = []
        in_dim = state_dim
        for hid_dim in hidden_dims:
            Layer = nn.Linear(in_dim, hid_dim)
            layers_list.append(Layer)
            activation = nn.ReLU()
            layers_list.append(activation)
            in_dim = hid_dim

        layers_list.append(nn.Linear(in_dim, chunk_size*action_dim))

        self.net = nn.Sequential(
            *layers_list                    # * Permets de passer la liste layers_list de forme [layer1, layer2, ...] sous plusieurs arguments layer1, layer2, ... Cela désempile
        )
        self.loss = nn.MSELoss()

    def _compute_action(
            self,
            state: torch.Tensor,
    ) -> torch.Tensor:

        flatten_action = self.net(state)
        unflatten_actions  = flatten_action.reshape(state.shape[0], self.chunk_size, self.action_dim)
        return unflatten_actions
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        predicted_actions = self._compute_action(state)
        return (self.loss(predicted_actions, action_chunk))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        predicted_actions = self._compute_action(state)
        return predicted_actions


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = self.state_dim + self.action_dim * self.chunk_size + 1 # size of state, size of action for chunk actions (chunk is the number of actions), size of flow matching timestep

        for hid_dim in hidden_dims:
            layer = nn.Linear(in_dim, hid_dim)
            in_dim = hid_dim
            layers.append(layer)
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_dim, self.action_dim * self.chunk_size))

        self.net = nn.Sequential(
            *layers,
        )

        self.loss = nn.MSELoss()

    def compute_velocity_field(
            self, 
            state: torch.Tensor, 
            noisy_action: torch.Tensor,
            tau: torch.Tensor,
            ) -> torch.Tensor:
        
        
        
        A_t_flat = noisy_action.flatten(start_dim=1)
        state_flat = state
        tau_flat = tau.flatten(start_dim=1)
        input_net = torch.cat([state_flat, A_t_flat, tau_flat], dim=1)

        output_net = self.net(input_net)
        output_net_reshape = output_net.reshape(state.shape[0], self.chunk_size, self.action_dim)

        return output_net_reshape
        
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        tau = torch.rand(state.shape[0], 1, 1, device=state.device)
        A_0 = torch.randn_like(action_chunk)
        A_t = tau * action_chunk + (1-tau) * A_0

        net_output = self.compute_velocity_field(state, A_t, tau)
        return self.loss(net_output, (action_chunk - A_0))


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        A_t = torch.randn((state.shape[0], self.chunk_size, self.action_dim)).to(state.device)
        tau = torch.zeros(state.shape[0], 1, 1, device=state.device)

        for _ in range(num_steps):
            vf = self.compute_velocity_field(state, A_t, tau)
            A_t = A_t + 1/num_steps*vf
            tau += 1/num_steps

        return A_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
