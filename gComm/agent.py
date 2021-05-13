from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.gumbel import Gumbel

class Agent(ABC):
    """
    An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action.
    """

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """
        Propose an action based on observation.
        Returns a dict, with 'action` entry containing the proposed action,
        and optionally other entries containing auxiliary information
        (e.g. value function).
        """
        pass

    @abstractmethod
    def transmit(self, concept, train=True):
        """
        Use the communication channel to communicate
        to the other agent (listener).
        """
        # if train: use communication channel
        # else:     use argmax
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


# class CommChannel:
#     """
#     Communication channel:
#     choices = ['continuous', 'binary', 'categorical', 'random', 'fixed']
#     continuous: continuous messages
#     discrete: binary/categorical messages
#
#     # baselines:
#     random: noisy messages to distract the listening agent
#     fixed: set of ones/zeros
#
#     :params
#     num_messages: number of messages to be transmitted
#     message_dim: dimension of each message
#     num_agents: number of agents interacting with each other
#     """
#
#     def __init__(self, num_messages, message_dim, choice, num_agents):
#         self.num_messages = num_messages
#         self.message_dim = message_dim
#         self.choice = choice
#         self.num_agents = num_agents
#
#     @staticmethod
#     def categorical_softmax(probs, tau=1, hard=True, sample=False, dim=-1):
#         cat_distr = RelaxedOneHotCategorical(tau, probs=probs)
#         if sample:
#             y_soft = cat_distr.sample()
#         else:  # differentiable sample
#             y_soft = cat_distr.rsample()
#
#         if hard:  # Straight-Through
#             index = y_soft.max(dim, keepdim=True)[1]
#             y_hard = torch.zeros_like(probs).scatter_(dim, index, 1.0)
#             out = y_hard - y_soft.detach() + y_soft
#         else:  # Re-parameterization trick
#             out = y_soft
#         return out
#
#     @staticmethod
#     def binary_softmax(probs, tau=0.5, eps=1e-20, hard=True, sample=False):
#         log_probs = torch.log(probs)
#
#         if sample:  # sample from Gumbel distribution
#             u = torch.rand(log_probs.size(), names='')
#             u = -torch.log(-torch.log(u + eps) + eps)
#             y = log_probs + u
#         else:
#             y = log_probs
#
#         y_gumbel = torch.softmax(y / tau, dim=-1)
#         y_hat, ind = y_gumbel.max(dim=-1)
#
#         if hard:
#             y_hard = ind.float()
#             out = (y_hard - y_hat).detach() + y_hat
#             return out
#         else:
#             return y_hat
#
#     def forward(self, probs):
#         for i in range(self.num_messages):
#             if self.choice == 'categorical':
#                 pass

class CommChannel:
    """
    xyz
    """

    def __init__(self, msg_len, num_msgs, comm_type, temp, device):
        self.msg_len = msg_len
        self.num_msgs = num_msgs
        self.comm_type = comm_type
        self.temp = temp
        self.device = device

        self.gumbel = Gumbel(loc=torch.tensor([0.], device=self.device), scale=torch.tensor([1.], device=self.device))
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def binary(self, logits):
        """
        generates binary messages using logits

        :param: logits: [batch_size, msg_len, 2]
        """
        y = self.log_softmax(logits)
        y_hat = y + self.gumbel.sample(y.size()).squeeze(-1)
        y_hat = self.softmax(y_hat * self.temp)

        # Straight-Through Trick
        y_hard = torch.max(y_hat, dim=2)[1].to(self.device)
        y_soft = torch.zeros(y.size(0), y.size(1)).to(self.device)
        for i in range(y_soft.shape[0]):
            y_soft[i, :] = y_hat[i, :, :].gather(1, y_hard[i, :].view(-1, 1)).view(-1)
        ret = (y_hard.float() - y_soft).detach() + y_soft
        return ret

    def one_hot(self, probs, sample=False, dim=-1):
        """
        generates one-hot messages using sampling probabilities

        :param: sample: differentiable sample when True
        :param: probs: sampling probabilities [batch_size, msg_len]
        """
        cat_distr = RelaxedOneHotCategorical(self.temp, probs=probs)
        if sample:
            y_soft = cat_distr.sample()
        else:  # differentiable sample
            y_soft = cat_distr.rsample()

        # Straight-Through Trick
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft, device=self.device).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def random(self):
        '''
        communicate random values
        '''
        message = torch.randn(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def fixed(self):
        '''
        communicate fixed values
        '''
        message = torch.ones(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def perfect(self, concept_representation):
        '''
        @concept_representation :
        '''
        message = np.zeros(12, dtype=np.float32)
        message[:8] = concept_representation[4:12]  # size, color
        message[-4:] = concept_representation[-4:]  # task
        message = torch.tensor(message, dtype=torch.float32).unsqueeze(0).to(self.device)
        return message

    def oracle(self):
        '''
        we don't need this function
        '''
        pass

