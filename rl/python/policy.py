"""Fully convolutional actor-critic for MaskablePPO.

Instead of squeezing the map into a vector and predicting 2881 actions with a dense layer,
the network predicts one logit per cell per building type ("drill here", "conveyor here facing up", ...).

The policy is hierarchical: p(action) = p(type) * p(cell | type), with type in {wait, drill, conveyor} and
p(cell | type) normalized over valid cells only (the last two observation planes are the validity masks).
Otherwise ~1750 of ~1800 valid actions are conveyors, and an untrained agent burns all its copper on
random conveyors before it ever places a drill.
Dilated residual convolutions at half resolution give every cell a view of the whole map, so it can find the core.
"""
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

VALUE_FEATURES = 128


class ResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class MapEncoder(BaseFeaturesExtractor):
    """Outputs [action logits (1 + 5*S*S) | value features] in one flat vector.

    U-Net-like: the global reasoning (where is the core, where is the ore) runs at half resolution,
    where it is 4x cheaper, and a light full-resolution layer picks the exact cell.
    """

    def __init__(self, observation_space, width=32, dilations=(1, 2, 4)):
        c, h, w = observation_space.shape
        self.cells = h * w
        super().__init__(observation_space, features_dim=1 + 5 * self.cells + VALUE_FEATURES)
        wide = width * 2
        self.stem = nn.Sequential(nn.Conv2d(c, width, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(width, wide, 3, stride=2, padding=1), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(wide, d) for d in dilations])
        self.up = nn.Upsample(size=(h, w), mode="nearest")
        self.merge = nn.Sequential(nn.Conv2d(width + wide, width, 3, padding=1), nn.ReLU())
        self.cell_head = nn.Conv2d(width, 5, 1)  # drill, conveyor right/up/left/down
        self.type_head = nn.Linear(2 * wide, 3)  # wait, drill, conveyor
        self.value_head = nn.Sequential(nn.Linear(2 * wide, VALUE_FEATURES), nn.ReLU())

    def forward(self, obs):
        fine = self.stem(obs)
        coarse = self.body(self.down(fine))
        x = self.merge(torch.cat([fine, self.up(coarse)], 1))
        pooled = torch.cat([coarse.mean((2, 3)), coarse.amax((2, 3))], 1)
        cell = self.cell_head(x)  # (B, 5, H, W)
        drill = cell[:, 0].flatten(1)  # index y * W + x, matches the env layout
        conveyor = cell[:, 1:].permute(0, 2, 3, 1).flatten(1)  # index (y * W + x) * 4 + rotation

        drill_ok = obs[:, -2].flatten(1) > 0.5
        conveyor_ok = (obs[:, -1].flatten(1) > 0.5).repeat_interleave(4, dim=1)
        drill = torch.log_softmax(drill.masked_fill(~drill_ok, -1e8), 1)
        conveyor = torch.log_softmax(conveyor.masked_fill(~conveyor_ok, -1e8), 1)
        kind = torch.log_softmax(self.type_head(pooled), 1)
        logits = torch.cat([kind[:, :1], kind[:, 1:2] + drill, kind[:, 2:] + conveyor], 1)
        return torch.cat([logits, self.value_head(pooled)], 1)


class SplitHead(nn.Module):
    """Routes the logits to the actor and the value features to the critic."""

    def __init__(self, n_logits):
        super().__init__()
        self.n_logits = n_logits
        self.latent_dim_pi = n_logits
        self.latent_dim_vf = VALUE_FEATURES

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return features[:, :self.n_logits]

    def forward_critic(self, features):
        return features[:, self.n_logits:]


class MapPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", MapEncoder)
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SplitHead(self.action_space.n)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # logits already come from the conv head; drop SB3's dense action layer
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
