from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical, Categorical
from torch.distributions.gumbel import Gumbel
from gComm.helpers import action_IND_to_STR


class Agent(ABC):
    """
    An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action.
    """

    def on_reset(self):
        pass

    def act(self, state):
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


class CommChannel:
    """
    Communication Channel (refer gComm paper for more info)
    """

    def __init__(self, msg_len, num_msgs, comm_type, temp, device):
        """
        num_msgs: number of messages transmitted by the speaker n_m (int)
        msg_len: message length d_m (int)
        comm_type: communication baseline (str) ['categorical', 'binary', 'continuous',
        random', 'fixed', 'perfect', 'oracle']
        temp: temperature parameter for discrete messages (float)
        device: torch.device("cpu") / torch.device("cuda")
        """
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

    def one_hot(self, probs, dim=-1):
        """
        generates one-hot messages using sampling probabilities

        :param: probs: sampling probabilities [batch_size, msg_len]
        """
        cat_distr = RelaxedOneHotCategorical(self.temp, probs=probs)
        y_soft = cat_distr.rsample()  # differentiable sample

        # Straight-Through Trick
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft, device=self.device).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def continuous(self, logits):
        """
        generates continuous messages

        :param: logits: hidden state of speaker RNN
        """
        return logits

    def random(self):
        """
        communicate random messages
        """
        message = torch.randn(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def fixed(self):
        """
        communicate fixed messages (ones)
        """
        message = torch.ones(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def perfect(self, concept_representation):
        """
        @concept_representation: [1 2 3 4 square cylinder circle diamond r b y g light heavy walk push pull pickup]
        """
        message = np.zeros(12, dtype=np.float32)
        message[:8] = concept_representation[4:12]  # size, color
        message[-4:] = concept_representation[-4:]  # task
        message = torch.tensor(message, dtype=torch.float32).unsqueeze(0).to(self.device)
        return message

    def oracle(self):
        """
        Oracle is implemented on the listener's end
        """
        pass


class SpeakerAgent(Agent):
    def __init__(self, num_msgs, msg_len, temp=1.0, speaker_model=None, comm_type=None, device=None):
        """
        num_msgs: number of messages transmitted by the speaker n_m (int)
        msg_len: message length d_m (int)
        temp: temperature parameter for discrete messages (float)
        speaker_model: user-defined model, pass a nn.module object for speaker
        comm_type: communication baseline (str) ['categorical', 'binary', 'continuous',
        random', 'fixed', 'perfect', 'oracle']
        device: torch.device("cpu") / torch.device("cuda")
        """
        assert comm_type in ['categorical', 'binary', 'random', 'fixed', 'perfect', 'oracle', 'continuous']
        self.comm_type = comm_type

        if comm_type in ['categorical', 'binary', 'continuous']:
            self.speaker_model = speaker_model
        assert device is not None, 'pass the device type'
        self.comm = CommChannel(msg_len=msg_len, num_msgs=num_msgs,
                                comm_type=comm_type, temp=temp, device=device)
        self.device = device

    def transmit(self, concept=None, validation=False):
        if self.comm_type == 'random':
            return self.comm.random()
        elif self.comm_type == 'fixed':
            return self.comm.fixed()
        elif self.comm_type == 'perfect':
            return self.comm.perfect(concept_representation=concept)
        elif self.comm_type == 'oracle':
            return None
        else:
            concept = torch.tensor(concept[:12], dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.speaker_model(data_input=concept, comm_channel=self.comm,
                                      validation=validation).view(1, -1)

    def act(self, state, validation=False):
        raise Exception('Not a valid function for speaker')


class ListenerAgent(Agent):
    def __init__(self, listener_model):
        """
        pass a nn.module object for listener
        """
        self.listener_model = listener_model

    def act(self, state, validate=False):
        """
        :state: tuple(grid_image, received_messages)
        :validate: if True use argmax(.), if False use sampling
        """
        policy_logits = self.listener_model(grid_image=state[0], speaker_out=state[1])
        policy_dist = torch.softmax(policy_logits, dim=-1)

        if validate:
            action = torch.argmax(policy_dist, dim=1)
        else:
            action = Categorical(probs=policy_dist).sample()

        log_prob = Categorical(probs=policy_dist).log_prob(action)
        entropy = -(torch.log(policy_dist + 1e-8) * policy_dist).sum()
        return log_prob, entropy, action_IND_to_STR(action.item())

    def transmit(self, concept, validation=False):
        raise Exception('Not a valid function for listener')


class Communication:
    def __init__(self):
        pass
