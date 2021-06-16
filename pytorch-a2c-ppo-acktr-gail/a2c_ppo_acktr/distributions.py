import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.model_based.dreamer.actor_models import OneHotDist, SplitDist

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class FixedSplitDist(SplitDist):
    def mode(self):
        return torch.cat((self._dist1.mode().float(), self._dist2.mode().float()), -1)

    def entropy(self):
        return self._dist1.entropy().sum(-1) + self._dist2.entropy()

    def log_prob(self, actions):
        return (
            self._dist1.log_prob(actions[:, :self.split_dim])
            + self._dist2.log_prob(actions[:, self.split_dim:]).sum(dim=-1)
        ).reshape(-1, 1)

    def sample(self):
        return self.rsample()

    def log_probs(self, actions):
        return self.log_prob(actions)


class DiscreteContinuousDist(nn.Module):
    def __init__(self, num_inputs, num_outputs, discrete_action_dim):
        super(DiscreteContinuousDist, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs - discrete_action_dim))
        self.discrete_action_dim = discrete_action_dim
        self.num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(
            (action_mean.shape[0], action_mean.shape[1] - self.discrete_action_dim)
        )
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        out = action_mean.split(
            self.discrete_action_dim, -1
        )
        if len(out) == 2:
            discrete_logits, continuous_action_mean = out
        else:
            discrete_logits, continuous_action_mean, extra = out
            continuous_action_mean = torch.cat((continuous_action_mean, extra), -1)
        dist1 = OneHotDist(logits=discrete_logits)
        dist2 = FixedNormal(continuous_action_mean, action_logstd.exp())
        dist = FixedSplitDist(dist1, dist2, self.discrete_action_dim)
        return dist
