from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.gumbel import Gumbel
from models import TargetEncoder, GridEncoder

class Agent(ABC):
    """
    An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action.
    """

    def on_reset(self):
        pass

    def act(self, state, validation=False):
        """
        Propose an action based on observation.
        Returns a dict, with 'action` entry containing the proposed action,
        and optionally other entries containing auxiliary information
        (e.g. value function).
        """
        pass

    def transmit(self, concept, validation=False):
        """
        Use the communication channel to communicate
        to the other agent (listener).
        """
        # if train: use communication channel
        # else:     use argmax

        pass

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

# ========================Speaker Module============================ #
# class SpeakerBot(nn.Module):
#     """
#     Speaker has access to the 18-d concept input (size + shape + color + weight + task)
#     and transmits 'num_msgs' messages of length 'msg_len'
#     input_size: 18-d
#     """
#
#     def __init__(self, comm_type, input_size, hidden_size, output_size, num_msgs, temp, device):
#         super().__init__()
#         # model params
#         self.comm_type = comm_type
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size  # msg_len
#         self.num_msgs = num_msgs
#         self.device = device
#
#         self.rnn_cell = nn.RNNCell(self.input_size, self.hidden_size)
#
#         if comm_type == 'categorical':
#             self.output_layer = nn.Linear(self.hidden_size, self.output_size)
#         elif comm_type == 'binary':
#             self.output_layer = nn.Linear(self.hidden_size, self.output_size*2)
#
#         self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=self.device), requires_grad=True)
#         self.comm_channel = CommChannel(msg_len=output_size, num_msgs=num_msgs,
#                                         comm_type=comm_type, temp=temp, device=device)
#
#     def forward(self, data_input, validation=False):
#         """
#         :param data_input: concept representation
#         :param validation: if True, take argmax(.) over logits/probs
#         """
#         # model code
#         batch_size = data_input.size(0)
#         decoder_hidden = self.init_input
#         message = []
#         entropy = torch.zeros((batch_size,)).to(self.device)
#         log_probs = torch.tensor(0.).to(self.device)
#
#         for ind in range(self.num_msgs):
#             # split_input = data_input[:, ind * 4: (ind + 1) * 4]
#             decoder_hidden = self.rnn_cell(data_input, decoder_hidden)
#             logits = self.output_layer(decoder_hidden)
#
#             if self.comm_type == 'categorical':
#                 probs = F.softmax(logits, dim=-1)
#                 if not validation:
#                     predict = self.comm_channel.one_hot(probs, sample=False)
#                 else:
#                     predict = F.one_hot(torch.argmax(probs, dim=1), num_classes=self.output_size).float()
#
#             elif self.comm_type == 'binary':
#                 logits = logits.view(batch_size, self.output_size, -1)
#                 binary_probs = torch.softmax(logits, dim=-1)
#                 probs = torch.zeros(batch_size, self.output_size).to(self.device)
#                 if not validation:
#                     predict = self.comm_channel.binary(logits)
#                 else:
#                     predict = torch.max(logits, dim=2)[1].to(self.device)
#                 for i in range(batch_size):
#                     probs[i, :] = binary_probs[i, :, :].gather(1, predict[i, :].view(-1, 1)).view(-1)
#
#             elif self.comm_type == 'continuous':
#                 probs = F.softmax(logits, dim=-1)
#                 predict = logits
#
#             log_probs += torch.log((probs * predict).sum(dim=1)).squeeze(0)
#             message.extend(predict)
#             decoder_hidden = predict.detach()
#
#         message = torch.stack(message)
#         return message, log_probs, entropy



class SpeakerAgent(Agent):
    def __init__(self, num_msgs, msg_len, speaker_model=None, comm_type=None, device=None):
        '''
        pass a nn.module object for speaker
        '''
        assert comm_type in ['simple', 'random', 'fixed', 'perfect']
        self.comm_type = comm_type

        if comm_type=='simple':
            self.speaker_model = speaker_model
        else:
            assert device is not None, 'pass the device type'
            self.comm = CommChannel(msg_len=msg_len, num_msgs=num_msgs, device=device)


    def transmit(self, concept=None, validation=False):
        if self.comm_type == 'random':
            return self.comm.random()
        elif self.comm_type == 'fixed':
            return self.comm.fixed()
        elif self.comm_type == 'perfect':
            return self.comm.perfect(concept_representation=concept)
        else:
            return self.speaker_model(data_input = concept, validation= validation)

    def act(self, state, validation=False):
        print('not a valid function for speaker')


class ListenerAgent(Agent):
    def __init__(self,listener_model,
                 listener_type='None'):
        '''
        pass a nn.module object for listener
        '''

        self.grid_encoder = GridEncoder()
        self.listener_model = listener_model
        if listener_type !='oracle':
            self.target_encoder = TargetEncoder()

        self.listener_type='oracle' if listener_type=='oracle' else 'None'


    def act(self, state, validation=False):

        return self.listener_model(state=state, validate=validation)

    def derive_state(self, speaker_messages, grid_representation):
        '''
        target encoder forward based in if the listener is oracle or not
        grid_Encoder forward
        '''
        if self.listener_type == 'oracle':
            grid_representation = grid_representation.contiguous().permute(2, 0, 1).unsqueeze(0)
        else:
            grid_representation = self.target_encoder(speaker_messages, grid_representation)
        listener_state = self.grid_encoder(grid_representation)
        return listener_state



    def transmit(self, concept, validation=False):
        print('not a valid function for listener')


class Communication :
    def __init__(self, speaker_agent, target_encode, grid_encoder,
                 listener_agent, comm_channel):
        self.speaker_agent = speaker_agent
        pass

